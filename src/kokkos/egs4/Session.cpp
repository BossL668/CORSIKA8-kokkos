/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/egs4/Session.hpp>
#include <corsika/detail/kokkos/egs4/Egs4C8AirWavefront.inl>
#include <corsika/detail/kokkos/egs4/Egs4C8DeviceRadio.inl>
#include <corsika/detail/kokkos/egs4/Egs4C8DeviceProfile.inl>
#include <corsika/detail/kokkos/egs4/Egs4C8HostOutput.inl>
#include <array>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace corsika::kokkos::egs4::application {
  namespace {
    struct RuntimeLease {
      bool ownsRuntime_{};
      RuntimeLease() {
        if (Kokkos::is_finalized())
          throw std::runtime_error(
              "Cannot create EGS4 session after Kokkos finalization");
        ownsRuntime_ = !Kokkos::is_initialized();
        if (ownsRuntime_) Kokkos::initialize();
      }
      ~RuntimeLease() {
        if (ownsRuntime_ && Kokkos::is_initialized() && !Kokkos::is_finalized())
          Kokkos::finalize();
      }
    };
    // Sessions are managed on the application's main host thread. Share runtime
    // ownership so destroying the first session cannot invalidate another one.
    std::shared_ptr<RuntimeLease> acquire_runtime() {
      if (Kokkos::is_finalized())
        throw std::runtime_error("Cannot create EGS4 session after Kokkos finalization");
      static std::weak_ptr<RuntimeLease> live;
      auto lease = live.lock();
      if (!lease) {
        lease = std::make_shared<RuntimeLease>();
        live = lease;
      }
      return lease;
    }
    RareMaterial rare_material(Configuration const& c) {
      return {c.electronMassMeV_,
              c.muonMassMeV_,
              c.chargedPionMassMeV_,
              c.protonMassMeV_,
              c.averageAtomicWeight_,
              {c.airComposition_[0], c.airComposition_[1], c.airComposition_[2]}};
    }
    TransportSettings session_transport_settings(Tables const& tables,
                                                 Configuration const& c) {
      auto const model =
          model_view(tables, {c.electronMassMeV_, c.electronuclearThresholdMeV_,
                              c.photonuclearThresholdMeV_, c.muonPairThresholdMeV_});
      // Apply HATCH once at session setup, including the table's rounded AE/AP.
      return transport_settings(model, c.electronTotalCutMeV_ - c.electronMassMeV_,
                                c.photonCutMeV_, c.stepfc_);
    }
    c8_adapter::PhotonuclearPolicy photonuclear_policy(Configuration const& c) {
      return {c.nativePhotonuclearVertices_,
              {c.protonMassMeV_, c.neutronMassMeV_, c.neutralPionMassMeV_,
               c.chargedPionMassMeV_, c.phiMassMeV_, c.omegaMassMeV_, c.rhoMassMeV_},
              make_transverse_momentum_table(),
              c.nativePromptRhoDecay_,
              c.nativePromptResonanceDecay_,
              {c.omegaMassMeV_, c.phiMassMeV_, c.chargedPionMassMeV_,
               c.neutralPionMassMeV_, c.muonMassMeV_, c.chargedKaonMassMeV_,
               c.longKaonMassMeV_, c.shortKaonMassMeV_, c.etaMassMeV_}};
    }
    struct Stream {
      bool failed_{};
      virtual ~Stream() = default;
      virtual void append(std::vector<::corsika::kokkos::ParticleState> const&) = 0;
      virtual std::size_t active() const = 0;
      virtual WaveResult advance(std::uint64_t, std::uint64_t) = 0;
      virtual RunStatistics statistics() const = 0;
      virtual void finish() = 0;
      virtual void restart(std::uint64_t, std::uint64_t, OutputCallbacks const&) = 0;
    };
    template <class TExec>
    struct NativeStream final : Stream {
      c8_adapter::AirWavefront<TExec> queue_;
      OutputCallbacks output_;
      DeviceRadio<TExec> radio_;
      DeviceProfile<TExec> profile_;
      HostOutput<TExec> hostOutput_;
      RunStatistics stats_;
      int maximumWaves_;
      bool needsPromptDecay_;
      NativeStream(Tables const& tables, Configuration const& c,
                   OutputCallbacks const& callbacks)
          : queue_(tables,
                   {c.electronMassMeV_, c.electronuclearThresholdMeV_,
                    c.photonuclearThresholdMeV_, c.muonPairThresholdMeV_},
                   c8_adapter::make_air_environment(
                       c.environment_, c.earthRadiusM_, c.sterncor_,
                       c8_adapter::AirConvention::c7_egs4_four_exponentials),
                   session_transport_settings(tables, c), c.queueCapacity_,
                   c.resolveRareVertices_, rare_material(c), photonuclear_policy(c),
                   c.thinning_, c.reuseQueueWorkspace_, c.splitRareKernels_)
          , output_(callbacks)
          , radio_(c.radio_, output_)
          , profile_(c.profile_, output_)
          , maximumWaves_(c.maximumWaves_)
          , needsPromptDecay_(c.nativePromptRhoDecay_ || c.nativePromptResonanceDecay_) {
        if (maximumWaves_ < 1 || !output_.step_ || !output_.radio_ ||
            !output_.observation_ || !output_.discarded_ ||
            ((c.nativePromptRhoDecay_ || c.nativePromptResonanceDecay_) &&
             !output_.promptDecay_))
          throw std::invalid_argument(
              "Native EGS4 stream requires all output consumers and bounded waves");
        queue_.beginShower({}, c.seed_, c.showerId_, 1);
        stats_.executionSpace_ = TExec::name();
      }
      void append(std::vector<::corsika::kokkos::ParticleState> const& input) override {
        queue_.append(input);
        stats_.injections_ += input.size();
      }
      std::size_t active() const override { return queue_.activeCount(); }
      RunStatistics statistics() const override { return stats_; }
      void restart(std::uint64_t seed, std::uint64_t shower,
                   OutputCallbacks const& next) override {
        if (!next.step_ || !next.radio_ || !next.observation_ || !next.discarded_ ||
            (needsPromptDecay_ && !next.promptDecay_) ||
            (radio_.enabled() && !next.radioWaveforms_) ||
            (profile_.enabled() && !next.profile_))
          throw std::invalid_argument(
              "Persistent native EGS4 event is missing an output consumer");
        queue_.resetForShower(seed, shower);
        output_ = next;
        stats_ = {};
        stats_.executionSpace_ = TExec::name();
        radio_.reset();
        profile_.reset();
      }
      void finish() override {
        profile_.finish(stats_);
        radio_.finish(stats_);
        stats_.queueWorkspaceAllocations_ = queue_.workspaceAllocations();
        stats_.queueWorkspaceReuses_ = queue_.workspaceReuses();
        stats_.queueProcessedRecords_ = queue_.processedRecords();
        stats_.queuePeakActive_ = queue_.peakActive();
        stats_.queueAdvanceWallMs_ = queue_.advanceWallMilliseconds();
        stats_.queueAppendCalls_ = queue_.appendCalls();
        stats_.queueAppendRecords_ = queue_.appendRecords();
        stats_.queueAppendCopiedActiveRecords_ = queue_.appendCopiedActiveRecords();
        stats_.queueAppendWallMs_ = queue_.appendWallMilliseconds();
        stats_.queueSplitRareWaves_ = queue_.splitRareWaves();
        stats_.queueSplitRareRecords_ = queue_.splitRareRecords();
        output_ = {}; // never retain references to the completed event's writers
      }
      WaveResult advance(std::uint64_t first, std::uint64_t count) override {
        if (!active()) return {};
        if (stats_.waves_ >= std::uint64_t(maximumWaves_))
          throw std::runtime_error("Native EGS4 stream wave limit reached");
        auto batch = queue_.advance(first, count);
        ++stats_.waves_;
        stats_.children_ += batch.created_;
        radio_.accumulate(batch.outputs_, stats_);
        profile_.accumulate(batch.outputs_);
        auto records =
            hostOutput_.collect(batch, profile_.enabled(), radio_.enabled(), stats_);
        WaveResult result;
        result.activeParticles_ = active();
        for (std::size_t i = 0; i < records.extent(0); ++i) {
          auto const& transfer = records(i).transfer_;
          result.targetRestEnergyGeV_ +=
              transfer.targetRestEnergyMeV_ * .001 * records(i).weight_;
          for (int j = 0; j < transfer.count_; ++j) {
            auto const& source = transfer.vertices_[j];
            HostRequest request;
            switch (source.kind_) {
              case c8_adapter::HostVertexKind::muon_transport:
                request.kind_ = HostRequestKind::muon_transport;
                break;
              case c8_adapter::HostVertexKind::photonuclear:
                request.kind_ = HostRequestKind::photonuclear;
                break;
              case c8_adapter::HostVertexKind::hadron_transport:
                request.kind_ = HostRequestKind::hadron_transport;
                break;
              case c8_adapter::HostVertexKind::c7_vector_meson_decay:
                request.kind_ = HostRequestKind::c7_vector_meson_decay;
                break;
              case c8_adapter::HostVertexKind::photonuclear_many_hadrons:
                request.kind_ = HostRequestKind::photonuclear_many_hadrons;
                break;
              case c8_adapter::HostVertexKind::native_em:
                continue; // already resident in native queue
            }
            request.particle_ = source.particle_;
            request.virtualPhoton_ = source.virtualPhoton_;
            request.targetAtomicNumber_ = source.targetAtomicNumber_;
            request.polarizationCosine_ = source.polarizationCosine_;
            request.polarizationAzimuth_ = source.polarizationAzimuth_;
            request.photonuclearBranch_ = source.photonuclearBranch_;
            request.targetPdg_ = source.targetPdg_;
            request.randomKey_ = source.random_.key_;
            request.hadronRandomKey_ = source.hadronRandom_.key_;
            result.hostRequests_.push_back(request);
          }
        }
        for (std::size_t i = 0; i < records.extent(0); ++i) {
          auto const& r = records(i).output_;
          auto const& transfer = records(i).transfer_;
          if (transfer.hasRhoDecay_ || transfer.hasResonanceDecay_) {
            PromptDecayRecord decay;
            decay.parent_ =
                transfer.hasRhoDecay_ ? transfer.rhoParent_ : transfer.resonanceParent_;
            decay.count_ = transfer.hasRhoDecay_ ? 2 : transfer.resonanceChildren_;
            for (int j = 0; j < decay.count_; ++j) {
              decay.daughters_[j] = transfer.vertices_[j].particle_;
              decay.polarizationCosine_[j] = transfer.vertices_[j].polarizationCosine_;
              decay.polarizationAzimuth_[j] = transfer.vertices_[j].polarizationAzimuth_;
            }
            output_.promptDecay_(decay);
            ++stats_.promptDecays_;
          }
          if (r.hasStep_ && !profile_.enabled()) output_.step_(r.step_);
          if (r.hasRadio_ && !radio_.enabled()) output_.radio_(r.radio_);
          if (r.hasObservation_) output_.observation_(r.observation_);
          if (r.discard_ != c8_adapter::AirDiscard::none) {
            output_.discarded_(r.discardedEnergyGeV_ * records(i).weight_,
                               static_cast<unsigned>(r.discard_));
          }
        }
        stats_.hostRequests_ += result.hostRequests_.size();
        stats_.targetRestEnergyGeV_ += result.targetRestEnergyGeV_;
        return result;
      }
    };
  } // namespace
  struct Session::Impl {
    Tables tables_;
    std::shared_ptr<RuntimeLease> runtime_;
    std::unique_ptr<Stream> stream_;
    std::unique_ptr<Stream> retained_;
    bool retainAcrossShowers_{};
    explicit Impl(std::string const& path)
        : tables_(read_air_tables(path))
        , runtime_(acquire_runtime()) {}
    explicit Impl(Tables value)
        : tables_(std::move(value))
        , runtime_(acquire_runtime()) {}
  };
  Session::Session(std::string const& path)
      : impl_(std::make_unique<Impl>(path)) {}
  Session::Session(Tables tables)
      : impl_(std::make_unique<Impl>(std::move(tables))) {}
  Session::~Session() = default;
  double Session::sternheimerCorrection() const {
    double const cutoff = impl_->tables_.medium_.ap_;
    constexpr std::array<std::pair<double, double>, 6> corrections{
        {{3., 0.}, {1., 6.}, {.4, 10.}, {.25, 11.}, {.15, 12.5}, {.05, 15.}}};
    for (auto const& pair : corrections) {
      if (std::abs(cutoff - pair.first) < 1.e-7) return pair.second;
    }
    throw std::invalid_argument(
        "No C7 EGSIN2 Sternheimer correction for this AIR-NTP table");
  }
  GpuMemoryBudget Session::gpuMemoryBudget(
      double fraction, ::corsika::kokkos::radio::RadioConfig const& radio) const {
    if (impl_->stream_ || !Kokkos::is_initialized() || Kokkos::is_finalized())
      throw std::logic_error(
          "Plan native EGS4 GPU memory before starting a live session");
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#ifdef KOKKOS_ENABLE_CUDA
    using gpu_execution_type = Kokkos::Cuda;
#else
    using gpu_execution_type = Kokkos::HIP;
#endif
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, gpu_execution_type>) {
      std::size_t free = 0, total = 0;
#ifdef KOKKOS_ENABLE_CUDA
      auto error = cudaMemGetInfo(&free, &total);
      if (error != cudaSuccess)
        throw std::runtime_error(std::string("CUDA memory query: ") +
                                 cudaGetErrorString(error));
#else
      auto error = hipMemGetInfo(&free, &total);
      if (error != hipSuccess)
        throw std::runtime_error(std::string("HIP memory query: ") +
                                 hipGetErrorString(error));
#endif
      using namespace c8_adapter;
      // Upper bound at capacity, including temporary next-generation storage.
      constexpr auto payload = 3 * sizeof(AirRecord) + sizeof(AirOutcome) +
                               sizeof(AirOutput) + sizeof(RareAirTransfer) +
                               sizeof(thinning_result_type) + 3 * sizeof(int) +
                               sizeof(HostOutputRecord);
      using Acc = ::corsika::kokkos::radio::RadioAccumulator<gpu_execution_type>;
      // Retained radio buffers can overlap their replacements when a batch grows.
      auto per_history =
          payload + (radio.enabled_
                         ? 2 * sizeof(::corsika::kokkos::LeptonTransportRecord) +
                               2 * sizeof(::corsika::kokkos::radio::RadioTrackKinematics)
                         : 0);
      // Retained workspaces can overlap their replacements during growth.
      return plan_gpu_memory(fraction, total, free, 2 * per_history,
                             512ULL * 1024 * 1024 + Acc::projectedDeviceBytes(radio));
    }
#endif
    throw std::invalid_argument("GPU memory fraction requires a GPU execution space");
  }
  template <class TExec>
  RunStatistics run_cascade(Tables const& tables, Configuration const& c,
                            std::vector<::corsika::kokkos::ParticleState> const& input,
                            OutputCallbacks const& output) {
    namespace na = c8_adapter;
    if (c.maximumWaves_ < 1 || !output.step_ || !output.radio_ || !output.observation_ ||
        !output.discarded_ ||
        ((c.nativePromptRhoDecay_ || c.nativePromptResonanceDecay_) &&
         !output.promptDecay_))
      throw std::invalid_argument(
          "Native EGS4 session requires bounded waves and all output consumers");
    auto env = na::make_air_environment(c.environment_, c.earthRadiusM_, c.sterncor_,
                                        na::AirConvention::c7_egs4_four_exponentials);
    ChannelThresholds thresholds{c.electronMassMeV_, c.electronuclearThresholdMeV_,
                                 c.photonuclearThresholdMeV_, c.muonPairThresholdMeV_};
    na::AirWavefront<TExec> queue(
        tables, thresholds, env, session_transport_settings(tables, c), c.queueCapacity_,
        c.resolveRareVertices_, rare_material(c), photonuclear_policy(c), c.thinning_,
        c.reuseQueueWorkspace_, c.splitRareKernels_);
    queue.beginShower(input, c.seed_, c.showerId_, c.firstChildId_);
    RunStatistics stats;
    stats.executionSpace_ = TExec::name();
    DeviceRadio<TExec> radio(c.radio_, output);
    DeviceProfile<TExec> profile(c.profile_, output);
    HostOutput<TExec> host_output;
    stats.injections_ = input.size();
    while (queue.activeCount()) {
      if (++stats.waves_ > std::uint64_t(c.maximumWaves_))
        throw std::runtime_error("Native EGS4 wave limit reached; shower incomplete");
      auto batch = queue.advance();
      stats.children_ += batch.created_;
      radio.accumulate(batch.outputs_, stats);
      profile.accumulate(batch.outputs_);
      auto records =
          host_output.collect(batch, profile.enabled(), radio.enabled(), stats);
      // Refuse the unfinished rare-channel integration before emitting this wave.
      for (std::size_t i = 0; i < records.extent(0); ++i)
        if (records(i).transfer_.count_)
          throw std::runtime_error(
              "Native EGS4 rare channel needs a same-model handler; no PROPOSAL "
              "substitution or completed-shower claim");
      for (std::size_t i = 0; i < records.extent(0); ++i) {
        auto const& r = records(i).output_;
        if (r.hasStep_ && !profile.enabled()) output.step_(r.step_);
        if (r.hasRadio_ && !radio.enabled()) output.radio_(r.radio_);
        if (r.hasObservation_) output.observation_(r.observation_);
        if (r.discard_ != na::AirDiscard::none) {
          output.discarded_(r.discardedEnergyGeV_ * records(i).weight_,
                            static_cast<unsigned>(r.discard_));
        }
      }
    }
    profile.finish(stats);
    radio.finish(stats);
    stats.queueWorkspaceAllocations_ = queue.workspaceAllocations();
    stats.queueWorkspaceReuses_ = queue.workspaceReuses();
    stats.queueProcessedRecords_ = queue.processedRecords();
    stats.queuePeakActive_ = queue.peakActive();
    stats.queueAdvanceWallMs_ = queue.advanceWallMilliseconds();
    stats.queueAppendCalls_ = queue.appendCalls();
    stats.queueAppendRecords_ = queue.appendRecords();
    stats.queueAppendCopiedActiveRecords_ = queue.appendCopiedActiveRecords();
    stats.queueAppendWallMs_ = queue.appendWallMilliseconds();
    stats.queueSplitRareWaves_ = queue.splitRareWaves();
    stats.queueSplitRareRecords_ = queue.splitRareRecords();
    return stats;
  }
  RunStatistics Session::run(Configuration const& c,
                             std::vector<::corsika::kokkos::ParticleState> const& input,
                             OutputCallbacks const& callbacks, bool host_reference) {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
      throw std::runtime_error("Native EGS4 session requires a live Kokkos runtime");
    if (impl_->stream_ || impl_->retained_)
      throw std::logic_error(
          "Finish/release the incremental EGS4 session before a standalone run");
    if (host_reference)
      return run_cascade<Kokkos::DefaultHostExecutionSpace>(impl_->tables_, c, input,
                                                            callbacks);
    return run_cascade<Kokkos::DefaultExecutionSpace>(impl_->tables_, c, input,
                                                      callbacks);
  }
  void Session::begin(Configuration const& c, OutputCallbacks const& output,
                      bool host_reference) {
    if (!Kokkos::is_initialized() || Kokkos::is_finalized())
      throw std::runtime_error("Native EGS4 runtime is not live");
    if (impl_->stream_ || impl_->retained_)
      throw std::logic_error(
          "Native EGS4 stream already started/retained; use resume with unchanged "
          "configuration");
    if (host_reference)
      impl_->stream_ = std::make_unique<NativeStream<Kokkos::DefaultHostExecutionSpace>>(
          impl_->tables_, c, output);
    else
      impl_->stream_ = std::make_unique<NativeStream<Kokkos::DefaultExecutionSpace>>(
          impl_->tables_, c, output);
    impl_->retainAcrossShowers_ = c.retainAcrossShowers_;
  }
  bool Session::hasRetainedShower() const { return bool(impl_->retained_); }
  void Session::resume(std::uint64_t seed, std::uint64_t shower,
                       OutputCallbacks const& output) {
    if (impl_->stream_ || !impl_->retained_ || impl_->retained_->failed_ ||
        !Kokkos::is_initialized() || Kokkos::is_finalized())
      throw std::logic_error("No completed native EGS4 state available for reuse");
    impl_->stream_ = std::move(impl_->retained_);
    try {
      impl_->stream_->restart(seed, shower, output);
    } catch (...) {
      impl_->stream_->failed_ = true;
      throw;
    }
  }
  void Session::append(std::vector<::corsika::kokkos::ParticleState> const& input) {
    if (!impl_->stream_ || impl_->stream_->failed_ || !Kokkos::is_initialized() ||
        Kokkos::is_finalized())
      throw std::logic_error("No usable native EGS4 stream");
    impl_->stream_->append(input); // rejected injection leaves the stream unchanged
  }
  std::size_t Session::activeParticles() const {
    if (!impl_->stream_ || impl_->stream_->failed_ || !Kokkos::is_initialized() ||
        Kokkos::is_finalized())
      throw std::logic_error("No usable native EGS4 stream");
    return impl_->stream_->active();
  }
  RunStatistics Session::progressStatistics() const {
    if (!impl_->stream_ || impl_->stream_->failed_)
      throw std::logic_error("No usable native EGS4 stream");
    return impl_->stream_->statistics();
  }
  WaveResult Session::advance(std::uint64_t first, std::uint64_t count) {
    if (!impl_->stream_ || impl_->stream_->failed_ || !Kokkos::is_initialized() ||
        Kokkos::is_finalized())
      throw std::logic_error("No usable native EGS4 stream");
    try {
      return impl_->stream_->advance(first, count);
    } catch (...) {
      impl_->stream_->failed_ = true;
      throw;
    } // partial output callbacks cannot be replayed
  }
  RunStatistics Session::finishEMQueue() {
    if (!impl_->stream_ || impl_->stream_->failed_ || !Kokkos::is_initialized() ||
        Kokkos::is_finalized() || impl_->stream_->active())
      throw std::logic_error("Native EGS4 queue is absent, failed or still active");
    try {
      impl_->stream_->finish();
    } catch (...) {
      impl_->stream_->failed_ = true;
      throw;
    }
    auto result = impl_->stream_->statistics();
    if (impl_->retainAcrossShowers_)
      impl_->retained_ = std::move(impl_->stream_);
    else
      impl_->stream_.reset();
    return result;
  }
} // namespace corsika::kokkos::egs4::application
