/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/PhotonFrontData.hpp>
#include <corsika/kokkos/em/algorithms/InteractionSelection.hpp>
#include <corsika/kokkos/em/algorithms/PhotonFinalStateStep.hpp>
#include <corsika/kokkos/em/PhotonTransportRecord.hpp>
#include <corsika/kokkos/output/ObservationRecord.hpp>
#include <corsika/kokkos/output/ProjectedStepRecord.hpp>
#include <corsika/kokkos/scheduling/WavefrontBucketing.hpp>
#include <corsika/kokkos/scheduling/WavefrontQueue.hpp>
#include <corsika/kokkos/runtime/HostStagingSpace.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace corsika::kokkos {

  /** Retained production photon buffers, control aliases and host staging.
   * One execution instance owns pending operations. Finish earlier work
   * before replacing ordinary buffers or destroying this workspace.
   * Candidate growth retains its production fence; views expose no C8 objects. */
  template <class TExecutionSpace>
  class PhotonFrontWorkspace {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using host_memory_space_type = host_staging_space_type<TExecutionSpace>;
    using unmanaged_type = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using interaction_count_view_type =
        Kokkos::View<std::uint64_t, memory_space_type, unmanaged_type>;
    using scan_totals_view_type =
        Kokkos::View<ResidentPhotonScanValue, memory_space_type, unmanaged_type>;
    using error_view_type =
        Kokkos::View<std::uint32_t*, memory_space_type, unmanaged_type>;

    using host_projected_step_view_type =
        Kokkos::View<ProjectedStepRecord*, host_memory_space_type>;
    using host_step_view_type =
        Kokkos::View<PhotonTransportRecord*, host_memory_space_type>;
    using host_record_view_type =
        Kokkos::View<PhotonFinalStateRecord*, host_memory_space_type>;
    using host_particle_view_type = Kokkos::View<ParticleState*, host_memory_space_type>;
    using host_fallback_view_type =
        Kokkos::View<FallbackRequest*, host_memory_space_type>;
    using host_observation_view_type =
        Kokkos::View<ObservationRecord*, host_memory_space_type>;

    void ensureCapacity(std::size_t const requested, TExecutionSpace const& execution);

    MemoryProjection projectedCapacity(std::size_t const requested) const;

    /**
     * Grow the register-heavy final-state arena to the active-front upper
     * bound.  Actual interaction work is still guarded by the compacted count
     * retained on the device.
     */
    void ensureCandidateCapacity(std::size_t const requested,
                                 TExecutionSpace const& execution);

    MemoryProjection projectedCandidateCapacity(std::size_t const requested) const;

    MemoryProjection projectedBucketingCapacity(std::size_t const requested,
                                                TExecutionSpace const& execution) const;

    void ensureHostOutputCapacity(std::size_t const projectedCount,
                                  std::size_t const stepCount,
                                  std::size_t const recordCount,
                                  std::size_t const chargedCount,
                                  std::size_t const fallbackCount,
                                  std::size_t const observationCount);

    std::size_t capacity() const noexcept;

    std::size_t deviceBytes() const noexcept;

    WavefrontQueue<TExecutionSpace> queue_{};
    WavefrontBucketingWorkspace<TExecutionSpace> bucketing_{};
    Kokkos::View<InteractionSelectionOutcome*, memory_space_type> selections_{};
    Kokkos::View<PhotonTransportRecord*, memory_space_type> transports_{};
    Kokkos::View<FallbackRequest*, memory_space_type> transportFallbacks_{};
    Kokkos::View<PhotonFinalStateClassification*, memory_space_type> finalStates_{};
    Kokkos::View<ResidentPhotonSourceCounts*, memory_space_type> counts_{};
    Kokkos::View<std::uint32_t*, memory_space_type> interactionFlags_{};
    Kokkos::View<std::size_t*, memory_space_type> interactionSources_{};
    Kokkos::View<std::uint64_t* [PhotonOffsetCount], memory_space_type> offsets_ {};
    Kokkos::View<PhotonTransportRecord*, memory_space_type> steps_{};
    Kokkos::View<ProjectedStepRecord*, memory_space_type> projectedSteps_{};
    Kokkos::View<PhotonFinalStateRecord*, memory_space_type> records_{};
    Kokkos::View<ParticleState*, memory_space_type> charged_{};
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks_{};
    Kokkos::View<ObservationRecord*, memory_space_type> observations_{};
    Kokkos::View<FirstInteractionRecord*, memory_space_type> firstSnapshots_{};
    Kokkos::View<std::uint32_t*, memory_space_type> firstInteractionCandidates_{};
    Kokkos::View<std::uint64_t*, memory_space_type> callStatistics_{};
    Kokkos::View<ResidentPhotonFrontControl, memory_space_type> frontControl_{};
    error_view_type error_{};
    interaction_count_view_type interactionCount_{};
    scan_totals_view_type scanTotals_{};
    Kokkos::View<ResidentPhotonFrontControl, host_staging_space_type<TExecutionSpace>>
        hostFrontControl_{};
    host_projected_step_view_type hostProjectedSteps_{};
    host_step_view_type hostSteps_{};
    host_record_view_type hostRecords_{};
    host_particle_view_type hostCharged_{};
    host_fallback_view_type hostFallbacks_{};
    host_observation_view_type hostObservations_{};
    Kokkos::View<std::uint64_t*, host_memory_space_type> hostCallStatistics_{};
    Kokkos::View<std::uint32_t*, host_memory_space_type>
        hostFirstInteractionCandidates_{};
    Kokkos::View<FirstInteractionRecord*, host_memory_space_type> hostFirstSnapshots_{};

  private:
    template <class TView>
    static void growHostView(TView& view, std::size_t const requested,
                             char const* const label);

    std::size_t capacity_{};
    std::size_t candidateCapacity_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/PhotonFrontWorkspace.inl>
