/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  void PhotonFrontWorkspace<TExecutionSpace>::ensureCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
    if (requested == 0)
      throw std::invalid_argument(
          "resident Kokkos photon workspace capacity must be positive");
    if (callStatistics_.extent(0) == 0)
      callStatistics_ = decltype(callStatistics_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_call_statistics"),
          PhotonCallStatisticCount);
    if (hostCallStatistics_.extent(0) == 0)
      hostCallStatistics_ = decltype(hostCallStatistics_)(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_host_call_statistics"),
          PhotonCallStatisticCount);
    if (frontControl_.data() == nullptr) {
      frontControl_ = decltype(frontControl_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_front_control"));
      hostFrontControl_ =
          decltype(hostFrontControl_)("c8_kokkos_resident_photon_host_front_control");
      auto* const controlBytes = reinterpret_cast<unsigned char*>(frontControl_.data());
      scanTotals_ = scan_totals_view_type(reinterpret_cast<ResidentPhotonScanValue*>(
          controlBytes + offsetof(ResidentPhotonFrontControl, totals_)));
      interactionCount_ = interaction_count_view_type(reinterpret_cast<std::uint64_t*>(
          controlBytes + offsetof(ResidentPhotonFrontControl, interactionCount_)));
      error_ = error_view_type(
          reinterpret_cast<std::uint32_t*>(
              controlBytes + offsetof(ResidentPhotonFrontControl, materializationError_)),
          1);
      hostFirstInteractionCandidates_ = decltype(hostFirstInteractionCandidates_)(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_host_first_candidates"),
          1);
      hostFirstSnapshots_ = decltype(hostFirstSnapshots_)(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_host_first_snapshot"),
          1);
    }
    if (requested <= capacity_) return;
    auto const grown = capacity_ > std::numeric_limits<std::size_t>::max() / 2
                           ? requested
                           : std::max(requested, capacity_ * 2);
    if (grown > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error(
          "resident Kokkos photon charged workspace exceeds addressable capacity");
    auto const chargedCapacity = grown * 2;
    queue_.ensureCapacity(grown, execution);
    selections_ =
        decltype(selections_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_resident_photon_selections"),
                              grown);
    transports_ =
        decltype(transports_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_resident_photon_transports"),
                              grown);
    transportFallbacks_ = decltype(transportFallbacks_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_photon_transport_fallbacks"),
        grown);
    counts_ = decltype(counts_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                   "c8_kokkos_resident_photon_counts"),
                                grown);
    interactionFlags_ = decltype(interactionFlags_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_photon_interaction_flags"),
        grown);
    interactionSources_ = decltype(interactionSources_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_photon_interaction_sources"),
        grown);
    offsets_ =
        decltype(offsets_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                              "c8_kokkos_resident_photon_offsets"),
                           grown);
    steps_ = decltype(steps_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_resident_photon_steps"),
                              grown);
    projectedSteps_ = decltype(projectedSteps_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_photon_projected_steps"),
        grown);
    records_ =
        decltype(records_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                              "c8_kokkos_resident_photon_records"),
                           grown);
    charged_ =
        decltype(charged_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                              "c8_kokkos_resident_photon_charged"),
                           chargedCapacity);
    fallbacks_ =
        decltype(fallbacks_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                "c8_kokkos_resident_photon_fallbacks"),
                             grown);
    observations_ = decltype(observations_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_photon_observations"),
        grown);
    // A shower has exactly one generation-zero projectile.  Match the
    // native-CUDA capture contract and retain only its first interaction,
    // instead of allocating and downloading one large snapshot per active
    // photon on every wavefront.
    if (firstSnapshots_.extent(0) == 0) {
      firstSnapshots_ = decltype(firstSnapshots_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_first_snapshot"),
          1);
      firstInteractionCandidates_ = decltype(firstInteractionCandidates_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_photon_first_candidates"),
          1);
    }
    // Commit the advertised capacity only after every allocation succeeds.
    // If an allocation throws, a retry must not observe a partially grown
    // workspace and return early with stale/null Views.
    capacity_ = grown;
  }

  template <class TExecutionSpace>
  MemoryProjection PhotonFrontWorkspace<TExecutionSpace>::projectedCapacity(
      std::size_t const requested) const {
    if (requested == 0)
      throw std::invalid_argument(
          "resident Kokkos photon workspace capacity must be positive");
    auto const current = deviceBytes();
    auto const viewBytes = [](auto const& view) {
      return view.span() * sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    auto const bytes = [](std::size_t const count, std::size_t const element) {
      return checked_memory_multiply(count, element);
    };
    MemoryProjectionBuilder projection(current);
    if (callStatistics_.extent(0) == 0)
      projection.replace(0, bytes(PhotonCallStatisticCount, sizeof(std::uint64_t)));
    if (frontControl_.data() == nullptr)
      projection.replace(0, sizeof(ResidentPhotonFrontControl));
    if (requested <= capacity_) return projection.result();

    auto const grown = capacity_ > std::numeric_limits<std::size_t>::max() / 2
                           ? requested
                           : std::max(requested, capacity_ * 2);
    if (grown > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error(
          "resident Kokkos photon charged workspace exceeds addressable capacity");
    projection.merge(queue_.deviceBytes(), queue_.projectedCapacity(grown));
    projection.replace(viewBytes(selections_),
                       bytes(grown, sizeof(InteractionSelectionOutcome)));
    projection.replace(viewBytes(transports_),
                       bytes(grown, sizeof(PhotonTransportRecord)));
    projection.replace(viewBytes(transportFallbacks_),
                       bytes(grown, sizeof(FallbackRequest)));
    projection.replace(viewBytes(counts_),
                       bytes(grown, sizeof(ResidentPhotonSourceCounts)));
    projection.replace(viewBytes(interactionFlags_), bytes(grown, sizeof(std::uint32_t)));
    projection.replace(viewBytes(interactionSources_), bytes(grown, sizeof(std::size_t)));
    projection.replace(viewBytes(offsets_),
                       bytes(bytes(grown, PhotonOffsetCount), sizeof(std::uint64_t)));
    projection.replace(viewBytes(steps_), bytes(grown, sizeof(PhotonTransportRecord)));
    projection.replace(viewBytes(projectedSteps_),
                       bytes(grown, sizeof(ProjectedStepRecord)));
    projection.replace(viewBytes(records_), bytes(grown, sizeof(PhotonFinalStateRecord)));
    projection.replace(viewBytes(charged_),
                       bytes(bytes(grown, 2), sizeof(ParticleState)));
    projection.replace(viewBytes(fallbacks_), bytes(grown, sizeof(FallbackRequest)));
    projection.replace(viewBytes(observations_), bytes(grown, sizeof(ObservationRecord)));
    if (firstSnapshots_.extent(0) == 0) {
      projection.replace(0, sizeof(FirstInteractionRecord));
      projection.replace(0, sizeof(std::uint32_t));
    }
    return projection.result();
  }

  template <class TExecutionSpace>
  void PhotonFrontWorkspace<TExecutionSpace>::ensureCandidateCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
    if (requested <= candidateCapacity_) return;
    auto const grown = candidateCapacity_ > std::numeric_limits<std::size_t>::max() / 2
                           ? requested
                           : std::max(requested, candidateCapacity_ * 2);
    // Statistics/profile kernels from the preceding front may still read
    // the old arena when no host output was downloaded.  Growth is rare;
    // synchronize only before replacing the View.
    execution.fence("grow resident Kokkos photon candidate workspace");
    finalStates_ = decltype(finalStates_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_photon_final_states"),
        grown);
    candidateCapacity_ = grown;
  }

  template <class TExecutionSpace>
  MemoryProjection PhotonFrontWorkspace<TExecutionSpace>::projectedCandidateCapacity(
      std::size_t const requested) const {
    auto const current = deviceBytes();
    if (requested <= candidateCapacity_) return {current, current};
    auto const grown = candidateCapacity_ > std::numeric_limits<std::size_t>::max() / 2
                           ? requested
                           : std::max(requested, candidateCapacity_ * 2);
    auto const oldBytes =
        finalStates_.span() * sizeof(typename decltype(finalStates_)::value_type);
    auto const newBytes =
        checked_memory_multiply(grown, sizeof(PhotonFinalStateClassification));
    MemoryProjectionBuilder projection(current);
    projection.replace(oldBytes, newBytes);
    return projection.result();
  }

  template <class TExecutionSpace>
  MemoryProjection PhotonFrontWorkspace<TExecutionSpace>::projectedBucketingCapacity(
      std::size_t const requested, TExecutionSpace const& execution) const {
    MemoryProjectionBuilder projection(deviceBytes());
    projection.merge(bucketing_.deviceBytes(),
                     bucketing_.projectedCapacity(requested, execution));
    return projection.result();
  }

  template <class TExecutionSpace>
  void PhotonFrontWorkspace<TExecutionSpace>::ensureHostOutputCapacity(
      std::size_t const projectedCount, std::size_t const stepCount,
      std::size_t const recordCount, std::size_t const chargedCount,
      std::size_t const fallbackCount, std::size_t const observationCount) {
    growHostView(hostProjectedSteps_, projectedCount,
                 "c8_kokkos_resident_photon_host_projected_steps");
    growHostView(hostSteps_, stepCount, "c8_kokkos_resident_photon_host_steps");
    growHostView(hostRecords_, recordCount, "c8_kokkos_resident_photon_host_records");
    growHostView(hostCharged_, chargedCount, "c8_kokkos_resident_photon_host_charged");
    growHostView(hostFallbacks_, fallbackCount,
                 "c8_kokkos_resident_photon_host_fallbacks");
    growHostView(hostObservations_, observationCount,
                 "c8_kokkos_resident_photon_host_observations");
  }

  template <class TExecutionSpace>
  std::size_t PhotonFrontWorkspace<TExecutionSpace>::capacity() const noexcept {
    return capacity_;
  }

  template <class TExecutionSpace>
  std::size_t PhotonFrontWorkspace<TExecutionSpace>::deviceBytes() const noexcept {
    auto const viewBytes = [](auto const& view) {
      // A default-constructed rank-zero Kokkos::View can report span()==1
      // even though data()==nullptr and no allocation exists.
      return (view.data() == nullptr ? 0 : view.span()) *
             sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    // error, interaction_count and scan_totals are unmanaged aliases into
    // front_control and therefore must not be counted a second time.
    return queue_.deviceBytes() + bucketing_.deviceBytes() + viewBytes(selections_) +
           viewBytes(transports_) + viewBytes(transportFallbacks_) +
           viewBytes(finalStates_) + viewBytes(counts_) + viewBytes(interactionFlags_) +
           viewBytes(interactionSources_) + viewBytes(offsets_) + viewBytes(steps_) +
           viewBytes(projectedSteps_) + viewBytes(records_) + viewBytes(charged_) +
           viewBytes(fallbacks_) + viewBytes(observations_) + viewBytes(firstSnapshots_) +
           viewBytes(firstInteractionCandidates_) + viewBytes(callStatistics_) +
           viewBytes(frontControl_);
  }

  template <class TExecutionSpace>
  template <class TView>
  void PhotonFrontWorkspace<TExecutionSpace>::growHostView(TView& view,
                                                           std::size_t const requested,
                                                           char const* const label) {
    if (requested <= view.extent(0)) return;
    auto const current = view.extent(0);
    auto const grown = current > std::numeric_limits<std::size_t>::max() / 2
                           ? requested
                           : std::max(requested, current * 2);
    TView replacement(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                      grown);
    view = std::move(replacement);
  }

} // namespace corsika::kokkos
