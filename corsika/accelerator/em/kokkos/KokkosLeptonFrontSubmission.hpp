/*
 * (c) Copyright 2026 CORSIKA Project; distributed under the BSD-3-Clause license.
 */
#pragma once
#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <corsika/accelerator/em/kokkos/CudaWavefrontControl.hpp>
#endif

namespace corsika::accelerator::em::kokkos_detail {
template<class ExecutionSpace> class LeptonControlDownload {
 public:
  using Memory = typename ExecutionSpace::memory_space;
  void submit(Kokkos::View<ResidentLeptonFrontControl, Memory> const& source,
              ExecutionSpace const& execution) {
    Kokkos::deep_copy(execution, host_, source);
    execution.fence("download OpenMP lepton front control");
  }
  bool poll() const noexcept { return true; }
  ResidentLeptonFrontControl take() const { return host_(); }
 private:
  Kokkos::View<ResidentLeptonFrontControl, Kokkos::HostSpace> host_{
      "c8_cooperative_lepton_control"};
};
#if defined(KOKKOS_ENABLE_CUDA)
template<> class LeptonControlDownload<Kokkos::Cuda>
    : public CudaWavefrontControl<ResidentLeptonFrontControl> {};
#endif

/** Single-coordinator resumable lepton front, using the production kernels.
 *
 * Ready -> WaitingForPhysics -> AccumulationReady -> WaitingForAccumulation
 *       -> ResultsReady -> Committed -> Ready (resident continuation).
 *
 * Only prepare() allocates. Both submit operations return without waiting on
 * CUDA; poll() uses this stream's completion event. Profile and CoREAS/ZHS
 * consume resident tracks in the SAME execution instance, not host copies.
 * All table views and optional accumulators must outlive the frame.
 * consume() lends output views once; consumers finish their copies before
 * returning. Cross-species queues and CPU fallback belong to the coordinator.
 * A bounded-capacity/history checkpoint retains the unprocessed next front;
 * takeRemaining() must claim it before the frame can accept new input.
 */
template<class ExecutionSpace> class KokkosLeptonFrontSubmission {
 public:
  enum class Phase { Empty, Ready, WaitingForPhysics, AccumulationReady,
                     WaitingForAccumulation, ResultsReady, Consuming,
                     Committed, Failed };
  using Workspace = KokkosResidentLeptonWorkspace<ExecutionSpace>;
  using Radio = radio::kokkos_detail::KokkosRadioAccumulator<ExecutionSpace>;

  KokkosLeptonFrontSubmission(
      std::size_t maximum_sources,
      KokkosPhysicsContextView<ExecutionSpace> physics_context,
      double electron_mass_GeV, bool air_moliere_fast_path,
      ExecutionSpace execution = {}, std::size_t chunk = 0,
      gpu::em::detail::DeviceProfileProjection projection = {},
      KokkosProfileAccumulator<ExecutionSpace>* profile = nullptr,
      Radio* radio = nullptr)
      : execution_(execution), physics_(physics_context),
        maximum_sources_(maximum_sources), chunk_(chunk),
        mass_(electron_mass_GeV), air_(air_moliere_fast_path),
        projection_(projection), profile_(profile), radio_(radio) {
    if (!maximum_sources || maximum_sources > std::numeric_limits<std::size_t>::max()/6 ||
        !physics_context.data() || !std::isfinite(mass_) || mass_ <= 0.)
      throw std::invalid_argument("invalid cooperative lepton capacity/context/mass");
  }
  KokkosLeptonFrontSubmission(KokkosLeptonFrontSubmission const&) = delete;
  KokkosLeptonFrontSubmission& operator=(KokkosLeptonFrontSubmission const&) = delete;
  ~KokkosLeptonFrontSubmission() {
    if (work_pending_) {
      try { execution_.fence("unwind cooperative lepton front"); } catch (...) {}
    }
  }

  void prepare(std::vector<gpu::em::EmParticleState> const& particles,
               std::uint64_t first_history, std::uint64_t history_limit,
               bool project_steps = false, bool capture_first = false) {
    if (phase_ != Phase::Empty &&
        (phase_ != Phase::Committed || (remainingCount() != 0 && !remaining_claimed_)))
      throw std::logic_error("cooperative lepton workspace/input is still owned");
    if (particles.size() > maximum_sources_ || first_history == 0 ||
        history_limit <= first_history ||
        particles.size() > (history_limit-first_history)/3)
      throw std::invalid_argument("cooperative lepton input/history capacity exceeded");
    for (auto const& p : particles)
      if (p.pid != 11 && p.pid != -11 && p.pid != 13 && p.pid != -13)
        throw std::invalid_argument("cooperative lepton input requires charged leptons");
    count_ = particles.size();
    capacity_ = std::max<std::size_t>(1, count_*3);
    photon_capacity_ = capacity_*2;
    source_capacity_ = std::min(capacity_, maximum_sources_);
    history_ = first_history; initial_history_ = first_history; history_limit_ = history_limit;
    project_steps_ = project_steps; capture_first_ = capture_first;
    remaining_claimed_ = false;
    result_ = {};
    try {
      work_pending_ = true;
      workspace_.ensureQueueCapacity(capacity_, execution_);
      workspace_.ensureSourceCapacity(source_capacity_, capacity_, execution_);
      workspace_.ensureOutputCapacity(source_capacity_, source_capacity_,
          source_capacity_*2, source_capacity_, source_capacity_, source_capacity_,
          project_steps_, capacity_, photon_capacity_, execution_);
      workspace_.bucketing.ensureCountRange(source_capacity_, execution_);
      if (radio_) radio_->reserveTrackCapacity(source_capacity_, execution_);
      workspace_.queue.upload(particles, execution_);
      Kokkos::deep_copy(execution_, workspace_.call_statistics, std::uint64_t{0});
      Kokkos::deep_copy(execution_, workspace_.first_interaction_candidates, std::uint32_t{0});
      Kokkos::deep_copy(execution_, workspace_.front_control, ResidentLeptonFrontControl{});
      bucket();
      execution_.fence("prepare cooperative lepton front");
      workspace_.queue.markExecutionSynchronized();
      work_pending_ = false; phase_ = Phase::Ready;
    } catch (...) { phase_ = Phase::Failed; throw; }
  }

  void submit() {
    if (phase_ != Phase::Ready)
      throw std::logic_error("cooperative lepton front is not ready");
    try {
      work_pending_ = true;
      if (count_)
        enqueueResidentLeptonFront(physics_, mass_, air_, bucket_.particles.rawDeviceView(),
            workspace_, count_, capacity_, photon_capacity_, history_, history_limit_,
            project_steps_, capture_first_, execution_, chunk_);
      control_.submit(workspace_.front_control, execution_);
      phase_ = Phase::WaitingForPhysics;
    } catch (...) { phase_ = Phase::Failed; throw; }
  }

  // true means the current submitted phase is finished, NOT necessarily that
  // the whole front is ready to commit. The caller examines phase().
  bool poll() {
    if (phase_ == Phase::AccumulationReady || phase_ == Phase::ResultsReady) return true;
    if (phase_ != Phase::WaitingForPhysics && phase_ != Phase::WaitingForAccumulation)
      throw std::logic_error("cooperative lepton front has no pending phase");
    try {
      if (!control_.poll()) return false;
      auto completed = control_.take();
      workspace_.queue.markExecutionSynchronized();
      work_pending_ = false;
      if (phase_ == Phase::WaitingForPhysics) {
        result_ = completed;
        validateControl();
        phase_ = Phase::AccumulationReady;
      } else {
        phase_ = Phase::ResultsReady;
      }
      return true;
    } catch (...) { phase_ = Phase::Failed; throw; }
  }

  void submitAccumulation() {
    if (phase_ != Phase::AccumulationReady)
      throw std::logic_error("cooperative lepton accumulation is not ready/already submitted");
    try {
      work_pending_ = true;
      if (profile_) profile_->requireOpenForAccumulation();
      if (count_)
        enqueueResidentLeptonAccumulation(workspace_, count_, result_, projection_,
                                         profile_, radio_, execution_, chunk_);
      // Completion covers profile statistics AND all radio projections. This
      // tiny second control copy reuses the preallocated pinned slot/event.
      control_.submit(workspace_.front_control, execution_);
      phase_ = Phase::WaitingForAccumulation;
    } catch (...) { phase_ = Phase::Failed; throw; }
  }

  template<class Consumer> void consume(Consumer&& consumer) {
    if (phase_ != Phase::ResultsReady)
      throw std::logic_error("cooperative lepton outputs not ready/already committed");
    phase_ = Phase::Consuming;
    try {
      consumer(result_, static_cast<Workspace const&>(workspace_), execution_);
      phase_ = Phase::Committed;
    } catch (...) { work_pending_ = true; phase_ = Phase::Failed; throw; }
  }

  bool continueResident() {
    if (phase_ != Phase::Committed || remaining_claimed_)
      throw std::logic_error("cooperative lepton continuation requires owned committed output");
    auto next = remainingCount();
    auto children = result_.totals.values[LeptonChildOffset];
    if (!next || next > source_capacity_ ||
        next > (history_limit_ - history_ - children)/3) return false;
    try {
      history_ += children;
      workspace_.queue.commitNextAlreadySynchronized(next);
      count_ = next;
      work_pending_ = true;
      Kokkos::deep_copy(execution_, workspace_.front_control, ResidentLeptonFrontControl{});
      bucket();
      result_ = {};
      phase_ = Phase::Ready;
      return true;
    } catch (...) { phase_ = Phase::Failed; throw; }
  }

  std::vector<gpu::em::EmParticleState> takeRemaining() {
    if (phase_ != Phase::Committed || remaining_claimed_)
      throw std::logic_error("cooperative lepton remaining front not ready/already claimed");
    try {
      workspace_.queue.commitNextAlreadySynchronized(remainingCount());
      auto particles = workspace_.queue.download(execution_);
      workspace_.queue.clear();
      remaining_claimed_ = true;
      return particles;
    } catch (...) { work_pending_ = true; phase_ = Phase::Failed; throw; }
  }
  Phase phase() const noexcept { return phase_; }
  std::size_t inputSize() const noexcept { return count_; }
  std::size_t remainingCount() const noexcept {
    return static_cast<std::size_t>(result_.totals.values[LeptonNextOffset]);
  }
  std::size_t deviceBytes() const { return workspace_.deviceBytes(); }
  std::uint64_t secondaryHistoryIdsUsed() const {
    if(phase_ != Phase::Committed)
      throw std::logic_error("cooperative history usage requires committed output");
    return history_ - initial_history_ + result_.totals.values[LeptonChildOffset];
  }

 private:
  void bucket() {
    bucket_ = bucketKokkosWavefront(workspace_.queue.current(), count_,
                                  workspace_.bucketing, execution_, chunk_);
  }
  void validateControl() const {
    auto const& t = result_.totals.values;
    auto remaining = static_cast<std::uint64_t>(count_);
    for (auto i : {LeptonSelectionFallbackOffset, LeptonTransportFallbackOffset,
                   LeptonVertexFallbackOffset, LeptonFinalStateFallbackOffset}) {
      if (t[i] > remaining) throw std::runtime_error("cooperative lepton fallback accounting overflow");
      remaining -= t[i];
    }
    if (result_.materialization_error || result_.interaction_count > count_ ||
        t[LeptonVertexFallbackOffset] > result_.interaction_count ||
        result_.vertex_interaction_count >
            result_.interaction_count-t[LeptonVertexFallbackOffset] ||
        t[LeptonFinalStateFallbackOffset] > result_.vertex_interaction_count ||
        t[LeptonStepOffset] > count_ ||
        t[LeptonRecordOffset] > result_.interaction_count ||
        t[LeptonObservationOffset] > count_ || t[LeptonDecayOffset] > count_ ||
        t[LeptonNextOffset] > capacity_ || t[LeptonNextOffset] > 3*count_ ||
        t[LeptonPhotonOffset] > photon_capacity_ || t[LeptonPhotonOffset] > 2*count_ ||
        t[LeptonChildOffset] > 3*count_ ||
        t[LeptonChildOffset] > history_limit_-history_)
      throw std::runtime_error("invalid cooperative lepton materialization control");
  }
  ExecutionSpace execution_;
  KokkosPhysicsContextView<ExecutionSpace> physics_;
  Workspace workspace_;
  KokkosWavefrontBucketBatch<ExecutionSpace> bucket_;
  LeptonControlDownload<ExecutionSpace> control_;
  std::size_t maximum_sources_, chunk_, count_{}, capacity_{}, photon_capacity_{}, source_capacity_{};
  std::uint64_t history_{}, initial_history_{}, history_limit_{};
  double mass_;
  bool air_, project_steps_{}, capture_first_{}, work_pending_{}, remaining_claimed_{};
  gpu::em::detail::DeviceProfileProjection projection_;
  KokkosProfileAccumulator<ExecutionSpace>* profile_;
  Radio* radio_;
  ResidentLeptonFrontControl result_{};
  Phase phase_{Phase::Empty};
};
} // namespace corsika::accelerator::em::kokkos_detail
