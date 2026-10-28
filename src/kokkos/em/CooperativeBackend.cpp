/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/em/execution/BackendInstance.hpp>
#include <corsika/kokkos/scheduling/CooperativeBackendMerge.hpp>
#include <corsika/kokkos/scheduling/IndependentEndpointDriver.hpp>
#include <corsika/kokkos/scheduling/IndependentSubshowerPump.hpp>
#include <corsika/kokkos/scheduling/CooperativeDriverAffinity.hpp>
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <corsika/kokkos/output/CooperativeProfileMerge.hpp>
#include <corsika/kokkos/radio/CooperativeRadioMerge.hpp>
#include <cuda_runtime_api.h>
#include <array>
#include <chrono>
#include <exception>
#include <optional>

namespace corsika::kokkos {

  namespace {

    using clock_type = std::chrono::steady_clock;

    // Air-shower host coordinator. Neither endpoint sees Stack/SecondaryView,
    // writers or scalar physics models. Only the CUDA endpoint's wavefront call
    // runs on a dedicated driver; OpenMP stays on its initialization thread.
    class CooperativeBackend final : public BackendInstance {
    public:
      explicit CooperativeBackend(RuntimeConfig config)
          : owner_(ownerConfig(config))
          , sliceObserver_(config.cooperativeSliceObserver_)
          , adaptive_(config.cooperativePolicy_ == "adaptive")
          , cpuPrimary_(config.executionBackend_ == "openmp-cuda") {
        if (cpuPrimary_ && adaptive_)
          throw std::invalid_argument(
              "openmp-cuda uses the CPU-primary queue policy, not experimental adaptive");
        config.cooperativeOwner_ = false;
        config.runtimeLease_ = owner_.shareCooperativeLifetime();
        config.executionBackend_ = "cuda";
        gpu_ = make_cuda_backend_instance(config);
        config.executionBackend_ = "openmp";
        cpu_ = make_openmp_backend_instance(config);
        info_ = gpu_->runtimeInfo();
        info_.backend_ = cpuPrimary_ ? "openmp-cuda" : "cuda-openmp";
        info_.openmp_ = true;
      }
      ~CooperativeBackend() override {
        subshowers_.reset();
        driver_.waitIdle();
      }
      void initialize(EnvironmentDataView const& environment,
                      TransportGeometryConfig const& geometry,
                      ProposalNativeTableSet const& table,
                      ProposalNativeAuxData const& auxiliary,
                      BackendConfig const& config) override {
        requireIdle();
        if (subshowers_) subshowers_->requireIdle();
        if (!config.deterministic_ ||
            (config.radio_.enabled_ && !config.radio_.deterministic_))
          throw std::invalid_argument(
              "Cooperative transport requires deterministic accumulators");
        if (config.profileProjection_.enabled_ &&
            !config.profileProjection_.accumulateOnDevice_)
          throw std::invalid_argument(
              "Cooperative profile requires endpoint integer accumulation");
        // Construct the small CPU workspace first, before any CUDA work is in flight.
        auto host = config;
        if (!adaptive_ && !cpuPrimary_) {
          host.minBatchSize_ = 256;
          host.residentBatchLimit_ = CooperativeHostPolicy::capacity(
              static_cast<std::size_t>(info_.hostThreads_));
          // Independent bounded host queues, not one arena per OpenMP thread.
          // This is the queue budget; tables, radio and result storage are additional.
          host.memoryFraction_ = std::min(config.memoryFraction_, 0.70 / 16.);
        }
        // Adaptive mode inherits the standalone OpenMP arena and queue budgets,
        // including any explicit user safety cap. No per-thread duplication.
        cpu_->initialize(environment, geometry, table, auxiliary, host);
        cpu_->prepareCooperativeWorkspace();
        if (adaptive_) hostProfileShards_ = cpu_->enableCooperativeHostProfileShards();
        gpu_->initialize(environment, geometry, table, auxiliary, config);
        // Allocation/preparation is serialized before either driver starts.
        // Otherwise a cold CUDA arena can delay its first kernels until after
        // the entire first OpenMP batch has already finished.
        gpu_->prepareCooperativeWorkspace();
        hostRadioTracks_.reserve(HostRadioBatchCapacity);
        // The auxiliary driver shares the authorized OpenMP CPU partition. Sleep
        // at existing CUDA result boundaries instead of contending with its worker
        // barriers. GPU-primary/adaptive and all single endpoints retain fences.
        if (cpuPrimary_) gpu_->setCooperativeBlockingWait(true);
        resetEvent();
      }
      void beginShower(ShowerConfig const& config) override {
        requireIdle();
        if (!hostRadioTracks_.empty())
          throw std::logic_error(
              "Cooperative beginShower with uncommitted host radio tracks");
        if (pendingPhotonCount() || pendingLeptonCount())
          throw std::logic_error("Cooperative beginShower with unconsumed particles");
        cpu_->beginShower(config);
        gpu_->beginShower(config);
        resetEvent();
      }
      bool canTransport(ParticleState const& p) const override {
        return gpu_->canTransport(p) && cpu_->canTransport(p);
      }
      bool hasProposalTable() const override {
        return gpu_->hasProposalTable() && cpu_->hasProposalTable();
      }
      std::size_t minimumBatchSize() const override { return gpu_->minimumBatchSize(); }
      std::size_t maximumResidentPhotonBatchSize() const override {
        return gpu_->maximumResidentPhotonBatchSize() +
               cpu_->maximumResidentPhotonBatchSize();
      }
      std::size_t maximumResidentLeptonBatchSize() const override {
        return gpu_->maximumResidentLeptonBatchSize() +
               cpu_->maximumResidentLeptonBatchSize();
      }
      std::size_t maximumResidentInputBatchSize() const override {
        return gpu_->maximumResidentInputBatchSize(); // never reduce the GPU arena
      }
      std::size_t pendingPhotonCount() const noexcept override {
        if (subshowers_) return subshowers_->pending(0);
        return eligible(*gpu_, true) + eligible(*cpu_, true);
      }
      std::size_t pendingLeptonCount() const noexcept override {
        if (subshowers_) return subshowers_->pending(1);
        return eligible(*gpu_, false) + eligible(*cpu_, false);
      }

      bool independentSubshowersEnabled() const noexcept override { return true; }
      // Preserve the primary EM front: completing every selected fallback here
      // forces tiny scalar-product batches back into the resident driver. Reuse
      // the existing bounded coordinator queue (4096 or final empty-front flush).
      // Original GPU-primary and every single-endpoint policy remain unchanged.
      bool batchIndependentSpecifiedFallbacks() const noexcept override {
        return adaptive_ || cpuPrimary_;
      }
      bool independentSubshowersReady() const noexcept override {
        return subshowers_ && subshowers_->ready();
      }
      std::size_t advanceIndependentSubshowers(
          std::vector<ParticleState> const& input,
          SubshowerCallbacks const& callbacks) override {
        CallScope scope(*this);
        if (!subshowers_) {
          subshowers_ = std::make_unique<IndependentSubshowerPump<BackendInstance>>(
              *gpu_, *cpu_, cooperative_,
              [device = info_.device_, affinity = driverAffinity_,
               restored = false]() mutable {
                if (!restored) {
                  affinity();
                  restored = true;
                }
                if (cudaSetDevice(device) != cudaSuccess)
                  throw std::runtime_error(
                      "Independent CUDA driver could not select its device");
              },
              sliceObserver_, static_cast<std::size_t>(info_.hostThreads_), adaptive_,
              64U << 20, cpuPrimary_);
        }
        return subshowers_->advance(input, callbacks);
      }

      PhotonCascadeResult runPhotonWavefront(std::vector<ParticleState> const& input,
                                             std::uint64_t first, std::size_t waves,
                                             std::size_t minimum) override {
        CallScope scope(*this);
        if (cpuPrimary_)
          throw std::logic_error(
              "CPU-primary execution requires the independent subshower API");
        if (subshowers_)
          throw std::logic_error(
              "Cannot mix synchronous and independent subshower calls");
        auto split = partition(input, true);
        if (!waves || waves > std::numeric_limits<std::uint64_t>::max() / 2 /
                                  std::max<std::size_t>(1, split.total()))
          throw std::overflow_error("Cooperative photon history reservation overflow");
        auto hostRange = 2 * split.cpuCount_ * waves;
        auto limit = cooperative_add(first, 2 * split.total() * waves);
        auto gpuFirst = cooperative_add(first, hostRange);
        cpu_->setCooperativePendingInputLimit(split.cpuPending_);
        auto results = runIndependent<PhotonCascadeResult>(
            split, true,
            [&] {
              return gpu_->runPhotonWavefront(split.gpu_, gpuFirst, waves, minimum);
            },
            [&] {
              return cpu_->runPhotonWavefront(split.cpu_, first,
                                              std::min(waves, HostEpochWaves), 1);
            });
        auto device = std::move(results.first);
        merge_cooperative(device, std::move(results.second));
        if (device.inputParticles_ != split.total() || gpuFirst > limit)
          throw std::logic_error("Cooperative photon commit lost/duplicated an input");
        ++cooperative_.photonCalls_;
        return device;
      }

      LeptonCascadeResult runLeptonWavefront(std::vector<ParticleState> const& input,
                                             std::uint64_t first, std::size_t waves,
                                             std::uint64_t limit,
                                             std::size_t minimum) override {
        CallScope scope(*this);
        if (cpuPrimary_)
          throw std::logic_error(
              "CPU-primary execution requires the independent subshower API");
        if (subshowers_)
          throw std::logic_error(
              "Cannot mix synchronous and independent subshower calls");
        auto split = partition(input, false);
        if (!waves || limit < first || split.total() > (limit - first) / 3)
          throw std::overflow_error("Cooperative lepton history reservation too small");
        auto available = limit - first;
        // Disjoint checked ID ranges; exhaustion returns remaining particles to
        // the coordinator. It never reallocates IDs from the other in-flight end.
        auto hostRange = split.cpuCount_
                             ? std::min<std::uint64_t>(192 * split.cpuCount_,
                                                       available - 3 * split.gpuCount_)
                             : 0;
        auto gpuFirst = cooperative_add(first, hostRange);
        cpu_->setCooperativePendingInputLimit(split.cpuPending_);
        auto results = runIndependent<LeptonCascadeResult>(
            split, false,
            [&] {
              return gpu_->runLeptonWavefront(split.gpu_, gpuFirst, waves, limit,
                                              minimum);
            },
            [&] {
              return cpu_->runLeptonWavefront(
                  split.cpu_, first, std::min(waves, HostEpochWaves), gpuFirst, 1);
            });
        auto device = std::move(results.first);
        merge_cooperative(device, std::move(results.second));
        if (device.inputParticles_ != split.total() ||
            device.secondaryHistoryIdsUsed_ > available)
          throw std::logic_error("Cooperative lepton commit lost/duplicated an input");
        ++cooperative_.leptonCalls_;
        return device;
      }

      BackendCapabilities capabilities() const override { return gpu_->capabilities(); }
      BackendStatistics const& statistics() const override {
        if (subshowers_) {
          subshowers_->requireHealthy();
          if (subshowers_->inFlight())
            throw std::logic_error("Cannot snapshot live endpoint statistics");
        }
        stats_ = merge_cooperative_statistics(gpu_->statistics(), cpu_->statistics());
        stats_.acceleratorBackend_ = info_.backend_;
        stats_.acceleratorOpenmp_ = true;
        stats_.cooperative_ = cooperative_;
        if (cpuPrimary_) {
          requireIdle();
          auto const wait = gpu_->cooperativeWaitStatistics();
          stats_.cooperative_.auxiliaryBlockingWaitEnabled_ = wait.blockingEnabled_;
          stats_.cooperative_.auxiliaryBlockingWaitCalls_ = wait.blockingCalls_;
          stats_.cooperative_.auxiliaryBlockingWaitMs_ =
              1000. * wait.blockingHostSeconds_;
        }
        stats_.cooperative_.openmpWorkspaceBytes_ =
            cpu_->statistics().physicalWorkspaceBytes_;
        return stats_;
      }
      bool gpuProfileEnabled() const noexcept override {
        return gpu_->gpuProfileEnabled();
      }
      ProfileResult downloadProfile() override {
        requireIdle();
        if (subshowers_) subshowers_->requireIdle();
        if (!profile_) {
          auto a = gpu_->downloadFixedProfile(identity_, CooperativeEndpoint::Cuda);
          auto b = cpu_->downloadFixedProfile(identity_, CooperativeEndpoint::OpenMP);
          CooperativeProfileMerge merged(identity_, a.config_);
          merged.commit(a);
          merged.commit(b);
          profile_ = decode_fixed_profile(merged.take());
        }
        return *profile_;
      }
      bool gpuRadioEnabled() const noexcept override { return gpu_->gpuRadioEnabled(); }
      void enqueueRadioTracks(boost::span<LeptonTransportRecord const> tracks) override {
        auto const* records = tracks.data();
        auto count = tracks.size();
        if (failed_ || radio_)
          throw std::logic_error("Cooperative radio unavailable after failure/output");
        if (count && !records)
          throw std::invalid_argument("null cooperative radio tracks");
        while (count) {
          auto n = std::min(count, HostRadioBatchCapacity - hostRadioTracks_.size());
          hostRadioTracks_.insert(hostRadioTracks_.end(), records, records + n);
          records += n;
          count -= n;
          if (hostRadioTracks_.size() == HostRadioBatchCapacity) flushHostRadioTracks();
        }
      }
      radio::RadioWaveforms downloadRadioWaveforms() override {
        requireIdle();
        if (subshowers_) subshowers_->requireIdle();
        if (!radio_) {
          flushHostRadioTracks();
          auto a = gpu_->downloadFixedRadio(identity_, CooperativeEndpoint::Cuda);
          auto b = cpu_->downloadFixedRadio(identity_, CooperativeEndpoint::OpenMP);
          radio::CooperativeRadioMerge merged(identity_, a.config_);
          merged.commit(a);
          merged.commit(b);
          radio_ = radio::decode_fixed_radio(merged.take());
        }
        return *radio_;
      }
      std::optional<FirstInteractionRecord> downloadFirstInteractionSnapshot() override {
        if (subshowers_) subshowers_->requireIdle();
        auto a = gpu_->downloadFirstInteractionSnapshot(),
             b = cpu_->downloadFirstInteractionSnapshot();
        if (a && b)
          throw std::logic_error("Cooperative first interaction has two owners");
        return a ? a : b;
      }
      RuntimeInfo const& runtimeInfo() const noexcept override { return info_; }
      // Oracle/stage APIs explicitly use one endpoint, never pretend to test both.
      InteractionBatchResult selectInteractionsForValidation(
          std::vector<ParticleState> const& p) override {
        return gpu_->selectInteractionsForValidation(p);
      }
      PhotonTransportBatchResult transportPhotonsForValidation(
          std::vector<InteractionRecord> const& p) override {
        return gpu_->transportPhotonsForValidation(p);
      }
      LeptonTransportBatchResult transportLeptonsForValidation(
          std::vector<InteractionRecord> const& p) override {
        return gpu_->transportLeptonsForValidation(p);
      }
      LeptonVertexSelectionBatchResult selectLeptonVerticesForValidation(
          std::vector<InteractionRecord> const& p) override {
        return gpu_->selectLeptonVerticesForValidation(p);
      }
      PhotonFinalStateBatchResult generatePhotonFinalStatesForValidation(
          std::vector<InteractionRecord> const& p, std::uint64_t h) override {
        return gpu_->generatePhotonFinalStatesForValidation(p, h);
      }
      LeptonFinalStateBatchResult generateLeptonFinalStatesForValidation(
          std::vector<InteractionRecord> const& p, std::uint64_t h) override {
        return gpu_->generateLeptonFinalStatesForValidation(p, h);
      }
      radio::RadioWaveforms projectRadioForValidation(
          std::vector<LeptonTransportRecord> const&) override {
        throw std::logic_error("Use endpoint radio oracle, not cooperative accumulator");
      }
      void setCooperativeProgress(std::function<bool()>) override {
        throw std::logic_error("Nested cooperative coordinator is forbidden");
      }
      void setCooperativeBlockingWait(bool) override {
        throw std::logic_error("Cooperative wait mode is selected by the coordinator");
      }
      CooperativeCudaWaitStatistics cooperativeWaitStatistics() const override {
        requireIdle();
        if (subshowers_) subshowers_->requireIdle();
        return gpu_->cooperativeWaitStatistics();
      }
      void prepareCooperativeWorkspace() override {
        throw std::logic_error("Cooperative workspace is already prepared");
      }
      void setCooperativePendingInputLimit(std::size_t) override {
        throw std::logic_error("Nested cooperative coordinator is forbidden");
      }
      std::vector<ParticleState> takeCooperativePending(bool, std::size_t) override {
        throw std::logic_error("Nested cooperative migration is forbidden");
      }
      FixedProfileSnapshot downloadFixedProfile(std::string const&,
                                                CooperativeEndpoint) override {
        throw std::logic_error("Cooperative output already contains both endpoints");
      }
      radio::FixedRadioSnapshot downloadFixedRadio(std::string const&,
                                                   CooperativeEndpoint) override {
        throw std::logic_error("Cooperative output already contains both endpoints");
      }

    private:
      static constexpr std::size_t HostRadioBatchCapacity = 4096;
      void flushHostRadioTracks() {
        if (hostRadioTracks_.empty()) return;
        // Scalar fallback callbacks may run while a CUDA subshower is in flight.
        // Wait once per bounded batch, never per CPU track, before touching that
        // endpoint's accumulator. Its transport result stays owned by the pump.
        driver_.waitIdle();
        if (subshowers_) subshowers_->waitCudaDriver();
        gpu_->enqueueRadioTracks({hostRadioTracks_.data(), hostRadioTracks_.size()});
        hostRadioTracks_.clear();
      }
      // A scheduling checkpoint, not a particle cut or a GPU-wait time slice.
      // Bound the longest host epoch so a growing CPU cascade can be rebalanced.
      static constexpr std::size_t HostEpochWaves = 8;
      struct SliceObservation {
        std::function<void(bool)> const& observer_;
        bool active_;
        SliceObservation(std::function<void(bool)> const& f, bool pending)
            : observer_(f)
            , active_(pending && bool(f)) {
          if (active_) observer_(true);
        }
        ~SliceObservation() {
          if (active_) observer_(false);
        }
      };
      static RuntimeConfig ownerConfig(RuntimeConfig c) {
        if (c.cooperativeOwner_ || c.runtimeLease_)
          throw std::invalid_argument("Cooperative backend must own its process runtime");
        c.executionBackend_ = "cuda";
        c.cooperativeOwner_ = true;
        return c;
      }
      static std::size_t eligible(BackendInstance const& b, bool photons) noexcept {
        return std::min(photons ? b.pendingPhotonCount() : b.pendingLeptonCount(),
                        photons ? b.maximumResidentPhotonBatchSize()
                                : b.maximumResidentLeptonBatchSize());
      }
      struct Split {
        std::vector<ParticleState> gpu_, cpu_;
        std::size_t gpuCount_{}, cpuCount_{}, cpuPending_{};
        std::size_t total() const { return gpuCount_ + cpuCount_; }
      };
      // Bounded independent calls. Join before transferring result ownership or
      // unwinding inputs, Views and runtime leases. No shared Stack/writer access.
      template <class TResult, class TDeviceCall, class THostCall>
      std::pair<TResult, TResult> runIndependent(Split const& split, bool photons,
                                                 TDeviceCall deviceCall,
                                                 THostCall hostCall) {
        struct Timed {
          TResult result_;
          clock_type::time_point begin_, end_;
        };
        auto epoch = clock_type::now();
        std::future<Timed> device;
        if (split.gpuCount_) {
          device = driver_.submit([&, deviceCall] {
            if (cudaSetDevice(info_.device_) != cudaSuccess)
              throw std::runtime_error(
                  "Independent CUDA driver could not select its device");
            auto begin = clock_type::now();
            auto result = deviceCall();
            return Timed{std::move(result), begin, clock_type::now()};
          });
        }
        TResult host;
        host.completed_ = true;
        auto hostBegin = clock_type::now(), hostEnd = hostBegin;
        try {
          if (split.cpuCount_) {
            SliceObservation observation(sliceObserver_, true);
            hostBegin = clock_type::now();
            host = hostCall(); // independent of CUDA wait callbacks
            hostEnd = clock_type::now();
          }
        } catch (...) {
          if (device.valid()) device.wait();
          driver_.waitIdle();
          throw;
        }
        Timed gpu;
        gpu.result_.completed_ = true;
        gpu.begin_ = gpu.end_ = epoch;
        try {
          if (device.valid()) gpu = device.get();
        } catch (...) {
          driver_.waitIdle();
          throw;
        }
        driver_.waitIdle();
        auto ms = [](auto a, auto b) {
          return std::chrono::duration<double, std::milli>(b - a).count();
        };
        double cpuMs = ms(hostBegin, hostEnd), gpuMs = ms(gpu.begin_, gpu.end_);
        auto& rates = throughput_[photons ? 0 : 1];
        if (split.cpuCount_) {
          ++cooperative_.openmpSlices_;
          cooperative_.openmpWallMs_ += cpuMs;
          cooperative_.maximumSliceMs_ = std::max(cooperative_.maximumSliceMs_, cpuMs);
          double rate = split.cpuCount_ / std::max(cpuMs, 1.e-6);
          rates[1] = rates[1] == 0 ? rate : .8 * rates[1] + .2 * rate;
        }
        if (split.gpuCount_) {
          cooperative_.cudaDriverWallMs_ += gpuMs;
          double rate = split.gpuCount_ / std::max(gpuMs, 1.e-6);
          rates[0] = rates[0] == 0 ? rate : .8 * rates[0] + .2 * rate;
        }
        if (split.gpuCount_ && split.cpuCount_) {
          ++cooperative_.independentJointCalls_;
          cooperative_.endpointWindowOverlapMs_ += std::max(
              0., ms(std::max(gpu.begin_, hostBegin), std::min(gpu.end_, hostEnd)));
          cooperative_.cudaFinishedBeforeHostMs_ += std::max(0., ms(gpu.end_, hostEnd));
          cooperative_.hostFinishedBeforeCudaMs_ += std::max(0., ms(hostEnd, gpu.end_));
        }
        cooperative_.jointWallMs_ += ms(epoch, clock_type::now());
        return {std::move(gpu.result_), std::move(host)};
      }
      Split partition(std::vector<ParticleState> const& input, bool photons) {
        auto gp = eligible(*gpu_, photons), cp = eligible(*cpu_, photons);
        auto gc = photons ? gpu_->maximumResidentPhotonBatchSize()
                          : gpu_->maximumResidentLeptonBatchSize();
        auto cc = photons ? cpu_->maximumResidentPhotonBatchSize()
                          : cpu_->maximumResidentLeptonBatchSize();
        if (input.size() > gc - gp + cc - cp)
          throw std::length_error("Cooperative input exceeds endpoint capacity");
        auto preserved = std::min(minimumBatchSize(), gc);
        auto const& rates = throughput_[photons ? 0 : 1];
        // Match predicted completion times including existing resident backlog.
        // The host is an execution peer, not limited to GPU wait opportunities.
        double cpuTarget = 256.;
        if (rates[0] > 0 && rates[1] > 0)
          cpuTarget = (input.size() + gp + cp) * rates[1] / (rates[0] + rates[1]);
        auto target = static_cast<std::size_t>(std::clamp(cpuTarget, 0., double(cc)));
        cooperative_.openmpBatchTarget_ = target;
        auto desired = std::min(target > cp ? target - cp : 0, cc - cp);
        // Do not bootstrap all small primary EM batches on the much slower
        // host before any throughput estimate exists. Preserve useful CUDA
        // fronts; peer execution does not imply equal particle counts.
        auto share =
            input.size() + gp > preserved
                ? std::min({desired, input.size(), input.size() + gp - preserved})
                : std::size_t{0};
        share = std::max(share, input.size() > gc - gp ? input.size() - (gc - gp) : 0);
        Split s;
        s.gpu_.assign(input.begin(), input.end() - share);
        s.cpu_.assign(input.end() - share, input.end());
        // Idle CPU may borrow a bounded stable prefix only at a safe boundary.
        // Preserve >= min-batch on CUDA and never touch in-flight work.
        auto actual = photons ? gpu_->pendingPhotonCount() : gpu_->pendingLeptonCount();
        if (cp == 0 && share == 0 && actual == gp && gp > preserved) {
          auto count = std::min(desired, gp - preserved);
          s.cpu_ = gpu_->takeCooperativePending(photons, count);
          gp -= count;
          cooperative_.migrationBytes_ =
              cooperative_add(cooperative_.migrationBytes_,
                              static_cast<std::uint64_t>(count * sizeof(ParticleState)));
        }
        // Symmetric safe-boundary migration: an idle CUDA end may take CPU
        // cross-species backlog. Do not refill a truncated eligible prefix in
        // this call: that would consume more histories than the caller reserved.
        auto cpuActual =
            photons ? cpu_->pendingPhotonCount() : cpu_->pendingLeptonCount();
        if (gp + s.gpu_.size() == 0 && cp && cpuActual == cp) {
          double fraction =
              rates[0] > 0 && rates[1] > 0 ? rates[0] / (rates[0] + rates[1]) : .75;
          auto count = std::min({gc, cp, std::max<std::size_t>(1, cp * fraction)});
          auto moved = cpu_->takeCooperativePending(photons, count);
          s.gpu_.insert(s.gpu_.end(), std::make_move_iterator(moved.begin()),
                        std::make_move_iterator(moved.end()));
          cp -= count;
          cooperative_.migrationBytes_ =
              cooperative_add(cooperative_.migrationBytes_,
                              static_cast<std::uint64_t>(count * sizeof(ParticleState)));
        }
        s.cpuPending_ = cp;
        s.cpuCount_ = cp + s.cpu_.size();
        s.gpuCount_ = gp + s.gpu_.size();
        cooperative_.cudaInputParticles_ += s.gpuCount_;
        cooperative_.openmpInputParticles_ += s.cpuCount_;
        return s;
      }
      void requireIdle() const {
        if (failed_)
          throw std::logic_error(
              "Cooperative shower failed; outputs cannot be completed");
        if (active_) throw std::logic_error("Cooperative call is already active");
      }
      struct CallScope {
        CooperativeBackend& b_;
        int exceptions_{std::uncaught_exceptions()};
        explicit CallScope(CooperativeBackend& backend)
            : b_(backend) {
          b_.requireIdle();
          if (b_.profile_ || b_.radio_)
            throw std::logic_error("Cooperative event outputs are sealed");
          b_.active_ = true;
        }
        ~CallScope() {
          b_.cpu_->setCooperativePendingInputLimit(
              std::numeric_limits<std::size_t>::max());
          b_.active_ = false;
          if (std::uncaught_exceptions() > exceptions_) b_.failed_ = true;
        }
      };
      void resetEvent() {
        if (subshowers_) subshowers_->requireIdle();
        subshowers_.reset();
        cooperative_ = {};
        cooperative_.enabled_ = true;
        cooperative_.adaptivePolicy_ = adaptive_;
        cooperative_.hostProfileShards_ = hostProfileShards_.first;
        cooperative_.hostProfileShardBytes_ = hostProfileShards_.second;
        cooperative_.independentDrivers_ = true;
        profile_.reset();
        radio_.reset();
        throughput_ = {};
        identity_ =
            "cooperative-shower-" + std::to_string(gpu_->statistics().showerOrdinal_);
      }
      // Runtime outlives all Views, accumulators, stream tickets and borrowed leases.
      std::function<void()> driverAffinity_ = capture_cooperative_driver_affinity();
      BackendRuntime owner_;
      std::unique_ptr<BackendInstance> gpu_, cpu_;
      RuntimeInfo info_;
      std::function<void(bool)> sliceObserver_;
      mutable BackendStatistics stats_;
      CooperativeStatistics cooperative_;
      std::optional<ProfileResult> profile_;
      std::optional<radio::RadioWaveforms> radio_;
      std::string identity_;
      bool active_{}, failed_{}, adaptive_{}, cpuPrimary_{};
      std::pair<std::size_t, std::size_t> hostProfileShards_{};
      std::array<std::array<double, 2>, 2> throughput_{};
      std::vector<LeptonTransportRecord> hostRadioTracks_;
      std::unique_ptr<IndependentSubshowerPump<BackendInstance>> subshowers_;
      IndependentEndpointDriver driver_; // destroyed/drained before endpoints
    };
  } // namespace
  std::unique_ptr<BackendInstance> make_cooperative_backend_instance(
      RuntimeConfig const& c) {
    return std::make_unique<CooperativeBackend>(c);
  }

} // namespace corsika::kokkos
