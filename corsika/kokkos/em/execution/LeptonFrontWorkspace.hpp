/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/LeptonFrontData.hpp>
#include <corsika/kokkos/em/algorithms/InteractionSelection.hpp>
#include <corsika/kokkos/em/algorithms/LeptonVertexSelection.hpp>
#include <corsika/kokkos/em/algorithms/LeptonFinalStateStep.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
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

  /** Production lepton buffers with separate logical and allocated capacities.
   * Queue, source and output growth retain their original checkpoints/fences.
   * Finish pending execution before replacing buffers or destroying the owner. */
  template <class TExecutionSpace>
  class LeptonFrontWorkspace {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using host_memory_space_type = host_staging_space_type<TExecutionSpace>;
    using unmanaged_type = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using interaction_count_view_type =
        Kokkos::View<std::uint64_t, memory_space_type, unmanaged_type>;
    using scan_totals_view_type =
        Kokkos::View<ResidentLeptonScanValue, memory_space_type, unmanaged_type>;
    using error_view_type =
        Kokkos::View<std::uint32_t*, memory_space_type, unmanaged_type>;

    using host_projected_step_view_type =
        Kokkos::View<ProjectedStepRecord*, host_memory_space_type>;
    using host_step_view_type =
        Kokkos::View<LeptonTransportRecord*, host_memory_space_type>;
    using host_record_view_type =
        Kokkos::View<LeptonFinalStateRecord*, host_memory_space_type>;
    using host_particle_view_type = Kokkos::View<ParticleState*, host_memory_space_type>;
    using host_fallback_view_type =
        Kokkos::View<FallbackRequest*, host_memory_space_type>;
    using host_observation_view_type =
        Kokkos::View<ObservationRecord*, host_memory_space_type>;

    /**
     * Reserve the lossless next-front limit without applying that limit to
     * every source and output array.  A lepton source may produce three
     * charged children, but the current source front normally remains much
     * smaller than this logical checkpoint capacity. The queue may reserve a
     * larger geometric allocation, but all front/history checks below retain
     * `requested` as their physical checkpoint.
     */
    void ensureQueueCapacity(std::size_t const requested,
                             TExecutionSpace const& execution);

    MemoryProjection projectedQueueCapacity(std::size_t const requested) const;

    /** Grow source-indexed storage only to the largest front actually run. */
    void ensureSourceCapacity(std::size_t const requested, std::size_t const hardLimit,
                              TExecutionSpace const& execution);

    MemoryProjection projectedSourceCapacity(std::size_t const requested,
                                             std::size_t const hardLimit) const;

    /**
     * Grow materialized output arrays from the source-ordered scan totals.
     * This is called only after the caller has passed the existing workspace
     * and history checkpoints, so allocation cannot change checkpoint state.
     */
    void ensureOutputCapacity(std::size_t const stepCount, std::size_t const recordCount,
                              std::size_t const photonCount,
                              std::size_t const fallbackCount,
                              std::size_t const observationCount,
                              std::size_t const decayCount, bool const projectSteps,
                              std::size_t const sourceHardLimit,
                              std::size_t const photonHardLimit,
                              TExecutionSpace const& execution);

    MemoryProjection projectedOutputCapacity(
        std::size_t const stepCount, std::size_t const recordCount,
        std::size_t const photonCount, std::size_t const fallbackCount,
        std::size_t const observationCount, std::size_t const decayCount,
        bool const projectSteps, std::size_t const sourceHardLimit,
        std::size_t const photonHardLimit) const;

    MemoryProjection projectedBucketingCapacity(std::size_t const requested,
                                                TExecutionSpace const& execution) const;

    void ensureHostOutputCapacity(
        std::size_t const projectedCount, std::size_t const stepCount,
        std::size_t const recordCount, std::size_t const photonCount,
        std::size_t const fallbackCount, std::size_t const observationCount,
        std::size_t const decayCount, std::size_t const sourceHardLimit,
        std::size_t const photonHardLimit);

    std::size_t capacity() const noexcept;

    std::size_t deviceBytes() const noexcept;

    WavefrontQueue<TExecutionSpace> queue_{};
    WavefrontBucketingWorkspace<TExecutionSpace> bucketing_{};
    Kokkos::View<InteractionSelectionOutcome*, memory_space_type> selections_{};
    Kokkos::View<LeptonTransportRecord*, memory_space_type> transports_{};
    Kokkos::View<FallbackRequest*, memory_space_type> transportFallbacks_{};
    Kokkos::View<LeptonVertexSelectionOutcome*, memory_space_type> vertices_{};
    Kokkos::View<LeptonFinalStateClassification*, memory_space_type> finalStates_{};
    Kokkos::View<ResidentLeptonSourceCounts*, memory_space_type> counts_{};
    Kokkos::View<std::uint32_t*, memory_space_type> interactionFlags_{};
    Kokkos::View<std::uint32_t*, memory_space_type> transportStates_{};
    Kokkos::View<std::size_t*, memory_space_type> interactionSources_{};
    Kokkos::View<std::uint64_t* [LeptonOffsetCount], memory_space_type> offsets_ {};
    Kokkos::View<LeptonTransportRecord*, memory_space_type> steps_{};
    Kokkos::View<ProjectedStepRecord*, memory_space_type> projectedSteps_{};
    Kokkos::View<LeptonFinalStateRecord*, memory_space_type> records_{};
    Kokkos::View<ParticleState*, memory_space_type> photons_{};
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks_{};
    Kokkos::View<ObservationRecord*, memory_space_type> observations_{};
    Kokkos::View<ParticleState*, memory_space_type> decays_{};
    Kokkos::View<FirstInteractionRecord*, memory_space_type> firstSnapshots_{};
    Kokkos::View<std::uint32_t*, memory_space_type> firstInteractionCandidates_{};
    Kokkos::View<std::uint64_t*, memory_space_type> callStatistics_{};
    Kokkos::View<ResidentLeptonFrontControl, memory_space_type> frontControl_{};
    error_view_type error_{};
    interaction_count_view_type interactionCount_{};
    interaction_count_view_type vertexInteractionCount_{};
    scan_totals_view_type scanTotals_{};
    Kokkos::View<ResidentLeptonFrontControl, host_staging_space_type<TExecutionSpace>>
        hostFrontControl_{};
    host_projected_step_view_type hostProjectedSteps_{};
    host_step_view_type hostSteps_{};
    host_record_view_type hostRecords_{};
    host_particle_view_type hostPhotons_{};
    host_fallback_view_type hostFallbacks_{};
    host_observation_view_type hostObservations_{};
    host_particle_view_type hostDecays_{};
    Kokkos::View<std::uint64_t*, host_memory_space_type> hostCallStatistics_{};
    Kokkos::View<std::uint32_t*, host_memory_space_type>
        hostFirstInteractionCandidates_{};
    Kokkos::View<FirstInteractionRecord*, host_memory_space_type> hostFirstSnapshots_{};

  private:
    template <class TView>
    static std::size_t viewBytes(TView const& view) noexcept;

    template <class TElement, class TView>
    static void replaceViewProjection(MemoryProjectionBuilder& projection,
                                      TView const& view, std::size_t const extent);

    template <class TElement, class TView>
    static void appendGrowViewProjection(MemoryProjectionBuilder& projection,
                                         TView const& view, std::size_t const current,
                                         std::size_t const requested,
                                         std::size_t const hardLimit);

    void appendDiagnosticProjection(MemoryProjectionBuilder& projection) const;

    static std::size_t grownCapacity(std::size_t const current,
                                     std::size_t const requested,
                                     std::size_t const hardLimit);

    template <class TView>
    static void growView(TView& view, std::size_t& current, std::size_t const requested,
                         std::size_t const hardLimit, TExecutionSpace const& execution,
                         char const* const label);

    template <class TView>
    static void growHostView(TView& view, std::size_t const requested,
                             std::size_t const hardLimit, char const* const label);

    void ensureDiagnosticViews(TExecutionSpace const& execution);

    std::size_t sourceCapacity_{};
    std::size_t stepCapacity_{};
    std::size_t projectedStepCapacity_{};
    std::size_t recordCapacity_{};
    std::size_t photonCapacity_{};
    std::size_t fallbackCapacity_{};
    std::size_t observationCapacity_{};
    std::size_t decayCapacity_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/LeptonFrontWorkspace.inl>
