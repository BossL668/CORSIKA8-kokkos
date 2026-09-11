/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */
#pragma once

#include <corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <corsika/accelerator/em/kokkos/CudaWavefrontControl.hpp>
#endif

namespace corsika::accelerator::em::kokkos_detail {

  // The CPU instance finishes its submitted phase synchronously. It uses no
  // CUDA allocations or initialization in an OpenMP-only build.
  template <class ExecutionSpace>
  class PhotonControlDownload {
   public:
    using Memory = typename ExecutionSpace::memory_space;
    void submit(Kokkos::View<ResidentPhotonFrontControl, Memory> const& source,
                ExecutionSpace const& execution) {
      Kokkos::deep_copy(execution, host_, source);
      execution.fence("download OpenMP photon front control");
    }
    bool poll() const noexcept { return true; }
    ResidentPhotonFrontControl take() const { return host_(); }
   private:
    Kokkos::View<ResidentPhotonFrontControl, Kokkos::HostSpace> host_{
        "c8_cooperative_photon_control"};
  };

#if defined(KOKKOS_ENABLE_CUDA)
  template <>
  class PhotonControlDownload<Kokkos::Cuda>
      : public CudaWavefrontControl<ResidentPhotonFrontControl> {};
#endif

  /** Resumable selection/transport/materialization phase, not a whole cascade.
   *
   * prepare() is an explicit allocation/upload boundary. submit() enqueues
   * the SAME production photon kernels and a control copy, then returns on
   * CUDA. poll() never fences. Only the coordinator may call these methods.
   * No CPU stack, writer, profile or radio accumulator is touched here.
   *
   * The private workspace stays resident and cannot be replaced while in
   * flight. consume() lends its read-only result views to exactly one host
   * callback. continueResident() then swaps resident photon buffers without
   * uploading/downloading that queue; charged children/fallback/profile
   * handling is still the consumer's responsibility, not implemented here.
   * The callback must finish reading/downloading before returning;
   * views must not escape it. The session owning the tables referenced by
   * physics_context must outlive this object. This is intentionally not yet
   * the public backend runPhotonWavefront API.
   */
  template <class ExecutionSpace>
  class KokkosPhotonFrontSubmission {
   public:
    enum class Phase { Empty, Ready, WaitingForControl, ResultsReady, Consuming, Committed, Failed };
    using Workspace = KokkosResidentPhotonWorkspace<ExecutionSpace>;

    KokkosPhotonFrontSubmission(
        std::size_t maximum_particles,
        KokkosPhysicsContextView<ExecutionSpace> physics_context,
        ExecutionSpace execution = {}, std::size_t openmp_chunk_size = 0)
        : execution_(execution), physics_context_(physics_context),
          maximum_particles_(maximum_particles), chunk_(openmp_chunk_size) {
      if (!maximum_particles ||
          maximum_particles > std::numeric_limits<std::size_t>::max() / 2 ||
          physics_context.data() == nullptr)
        throw std::invalid_argument("invalid cooperative photon front capacity/context");
    }
    KokkosPhotonFrontSubmission(KokkosPhotonFrontSubmission const&) = delete;
    KokkosPhotonFrontSubmission& operator=(KokkosPhotonFrontSubmission const&) = delete;
    ~KokkosPhotonFrontSubmission() {
      // Also covers a launch failing before the control event was recorded.
      // Only this execution instance is drained, never the other endpoint.
      if (work_pending_) {
        try { execution_.fence("unwind cooperative photon front"); }
        catch (...) {} // Do not mask an already-reported device exception.
      }
    }

    void prepare(std::vector<gpu::em::EmParticleState> const& particles,
                 std::uint64_t first_secondary_history_id,
                 bool project_steps = false, bool capture_first = false) {
      if (phase_ != Phase::Empty && phase_ != Phase::Committed)
        throw std::logic_error("cooperative photon front is still owned/in flight");
      if (particles.size() > maximum_particles_ || !first_secondary_history_id ||
          particles.size() >
              (std::numeric_limits<std::uint64_t>::max() - first_secondary_history_id) / 2)
        throw std::invalid_argument("cooperative photon input/history capacity exceeded");
      for (auto const& particle : particles)
        if (particle.pid != static_cast<std::int32_t>(gpu::em::EmPid::Photon))
          throw std::invalid_argument("cooperative photon front requires photons");
      count_ = particles.size();
      logical_capacity_ = count_;
      history_ = first_secondary_history_id;
      project_steps_ = project_steps;
      capture_first_ = capture_first;
      try {
        work_pending_ = true;
        workspace_.ensureCapacity(std::max<std::size_t>(1, count_), execution_);
        workspace_.ensureCandidateCapacity(std::max<std::size_t>(1, count_), execution_);
        workspace_.bucketing.ensureCountRange(std::max<std::size_t>(1,count_), execution_);
        workspace_.queue.upload(particles, execution_);
        Kokkos::deep_copy(execution_, workspace_.first_interaction_candidates, std::uint32_t{0});
        Kokkos::deep_copy(execution_, workspace_.front_control, ResidentPhotonFrontControl{});
        bucket_ = bucketKokkosWavefront(workspace_.queue.current(), count_,
                                       workspace_.bucketing, execution_, chunk_);
        // Explicit staging boundary, before either endpoint is submitted.
        execution_.fence("prepare cooperative photon front");
        workspace_.queue.markExecutionSynchronized();
        work_pending_ = false;
        phase_ = Phase::Ready;
      } catch (...) { phase_ = Phase::Failed; throw; }
    }

    void submit() {
      if (phase_ != Phase::Ready)
        throw std::logic_error("cooperative photon front is not ready for submission");
      try {
        work_pending_ = true;
        if (count_ != 0)
          enqueueResidentPhotonFront(
              physics_context_, bucket_.particles.rawDeviceView(),
              workspace_.queue.next().rawDeviceView(), workspace_, count_, logical_capacity_,
              history_, project_steps_, capture_first_, execution_, chunk_);
        control_.submit(workspace_.front_control, execution_);
        phase_ = Phase::WaitingForControl;
      } catch (...) { phase_ = Phase::Failed; throw; }
    }

    bool poll() {
      if (phase_ == Phase::ResultsReady) return true;
      if (phase_ != Phase::WaitingForControl)
        throw std::logic_error("cooperative photon front has no pending control");
      try {
        if (!control_.poll()) return false;
        result_control_ = control_.take();
        workspace_.queue.markExecutionSynchronized();
        work_pending_ = false;
        auto const& t = result_control_.totals.values;
        if (result_control_.materialization_error ||
            result_control_.interaction_count > count_ ||
            t[PhotonNextOffset] > count_ || t[PhotonChargedOffset] > 2 * count_ ||
            t[PhotonChildOffset] > 2 * count_ ||
            t[PhotonStepOffset] > count_ || t[PhotonObservationOffset] > count_ ||
            t[PhotonRecordOffset] > result_control_.interaction_count ||
            t[PhotonSelectionFallbackOffset] > count_ ||
            t[PhotonTransportFallbackOffset] > count_ - t[PhotonSelectionFallbackOffset] ||
            t[PhotonFinalStateFallbackOffset] > count_ - t[PhotonSelectionFallbackOffset] -
                                                   t[PhotonTransportFallbackOffset])
          throw std::runtime_error("invalid cooperative photon materialization control");
        phase_ = Phase::ResultsReady;
        return true;
      } catch (...) { phase_ = Phase::Failed; throw; }
    }

    template <class Consumer>
    void consume(Consumer&& consumer) {
      if (phase_ != Phase::ResultsReady)
        throw std::logic_error("cooperative photon result is not ready/already committed");
      // Transition before the callback, also rejecting recursive consume().
      phase_ = Phase::Consuming;
      try {
        consumer(result_control_, static_cast<Workspace const&>(workspace_), execution_);
        phase_ = Phase::Committed;
      } catch (...) {
        work_pending_ = true; // a consumer may have queued an output transfer
        phase_ = Phase::Failed;
        throw;
      }
    }

    Phase phase() const noexcept { return phase_; }

    // Resume the same photon history on its current execution space. No
    // particle data crosses the device boundary at this transition.
    bool continueResident() {
      if (phase_ != Phase::Committed)
        throw std::logic_error("photon continuation requires a committed front");
      auto const children = result_control_.totals.values[PhotonChildOffset];
      auto const next = result_control_.totals.values[PhotonNextOffset];
      if (!next) return false;
      try {
        if (children > std::numeric_limits<std::uint64_t>::max() - history_ ||
            next > (std::numeric_limits<std::uint64_t>::max() - history_ - children) / 2)
          throw std::overflow_error("cooperative resident photon history exhausted");
        history_ += children;
        workspace_.queue.commitNextAlreadySynchronized(next);
        count_ = static_cast<std::size_t>(next);
        work_pending_ = true;
        Kokkos::deep_copy(execution_, workspace_.error, std::uint32_t{0});
        bucket_ = bucketKokkosWavefront(workspace_.queue.current(), count_,
                                       workspace_.bucketing, execution_, chunk_);
        phase_ = Phase::Ready;
        return true;
      } catch (...) { phase_ = Phase::Failed; throw; }
    }

    std::size_t deviceBytes() const { return workspace_.deviceBytes(); }
    std::size_t inputSize() const noexcept { return count_; }

   private:
    ExecutionSpace execution_;
    KokkosPhysicsContextView<ExecutionSpace> physics_context_;
    Workspace workspace_;
    KokkosWavefrontBucketBatch<ExecutionSpace> bucket_;
    PhotonControlDownload<ExecutionSpace> control_;
    std::size_t maximum_particles_, chunk_, count_{}, logical_capacity_{};
    std::uint64_t history_{};
    bool project_steps_{}, capture_first_{}, work_pending_{};
    ResidentPhotonFrontControl result_control_{};
    Phase phase_{Phase::Empty};
  };
} // namespace corsika::accelerator::em::kokkos_detail
