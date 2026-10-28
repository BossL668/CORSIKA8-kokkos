/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  void LeptonFrontWorkspace<TExecutionSpace>::ensureQueueCapacity(
      std::size_t const requested, TExecutionSpace const& execution) {
    if (requested == 0)
      throw std::invalid_argument(
          "resident Kokkos lepton queue capacity must be positive");
    queue_.ensureCapacity(requested, execution);
    ensureDiagnosticViews(execution);
  }

  template <class TExecutionSpace>
  MemoryProjection LeptonFrontWorkspace<TExecutionSpace>::projectedQueueCapacity(
      std::size_t const requested) const {
    if (requested == 0)
      throw std::invalid_argument(
          "resident Kokkos lepton queue capacity must be positive");
    MemoryProjectionBuilder projection(deviceBytes());
    projection.merge(queue_.deviceBytes(), queue_.projectedCapacity(requested));
    appendDiagnosticProjection(projection);
    return projection.result();
  }

  template <class TExecutionSpace>
  void LeptonFrontWorkspace<TExecutionSpace>::ensureSourceCapacity(
      std::size_t const requested, std::size_t const hardLimit,
      TExecutionSpace const& execution) {
    if (requested == 0 || requested > hardLimit)
      throw std::length_error(
          "resident Kokkos lepton source capacity exceeds its logical limit");
    if (requested <= sourceCapacity_) return;

    // The preceding wavefront may still have statistics or output copies
    // reading these Views on the same execution-space instance.  Growth is
    // rare, so fence only on the allocation path before replacing them.
    execution.fence("grow resident Kokkos lepton source workspace");
    auto const grown = grownCapacity(sourceCapacity_, requested, hardLimit);
    selections_ =
        decltype(selections_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_resident_lepton_selections"),
                              grown);
    transports_ =
        decltype(transports_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                 "c8_kokkos_resident_lepton_transports"),
                              grown);
    transportFallbacks_ = decltype(transportFallbacks_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_lepton_transport_fallbacks"),
        grown);
    vertices_ =
        decltype(vertices_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                               "c8_kokkos_resident_lepton_vertices"),
                            grown);
    finalStates_ = decltype(finalStates_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_lepton_final_states"),
        grown);
    counts_ = decltype(counts_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                                   "c8_kokkos_resident_lepton_counts"),
                                grown);
    interactionFlags_ = decltype(interactionFlags_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_lepton_interaction_flags"),
        grown);
    transportStates_ = decltype(transportStates_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_lepton_transport_states"),
        grown);
    interactionSources_ = decltype(interactionSources_)(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                           "c8_kokkos_resident_lepton_interaction_sources"),
        grown);
    offsets_ =
        decltype(offsets_)(Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                              "c8_kokkos_resident_lepton_offsets"),
                           grown);
    sourceCapacity_ = grown;
    ensureDiagnosticViews(execution);
  }

  template <class TExecutionSpace>
  MemoryProjection LeptonFrontWorkspace<TExecutionSpace>::projectedSourceCapacity(
      std::size_t const requested, std::size_t const hardLimit) const {
    if (requested == 0 || requested > hardLimit)
      throw std::length_error(
          "resident Kokkos lepton source capacity exceeds its logical limit");
    auto const current = deviceBytes();
    MemoryProjectionBuilder projection(current);
    if (requested > sourceCapacity_) {
      auto const grown = grownCapacity(sourceCapacity_, requested, hardLimit);
      replaceViewProjection<InteractionSelectionOutcome>(projection, selections_, grown);
      replaceViewProjection<LeptonTransportRecord>(projection, transports_, grown);
      replaceViewProjection<FallbackRequest>(projection, transportFallbacks_, grown);
      replaceViewProjection<LeptonVertexSelectionOutcome>(projection, vertices_, grown);
      replaceViewProjection<LeptonFinalStateClassification>(projection, finalStates_,
                                                            grown);
      replaceViewProjection<ResidentLeptonSourceCounts>(projection, counts_, grown);
      replaceViewProjection<std::uint32_t>(projection, interactionFlags_, grown);
      replaceViewProjection<std::uint32_t>(projection, transportStates_, grown);
      replaceViewProjection<std::size_t>(projection, interactionSources_, grown);
      projection.replace(
          viewBytes(offsets_),
          checked_memory_multiply(checked_memory_multiply(grown, LeptonOffsetCount),
                                  sizeof(std::uint64_t)));
    }
    appendDiagnosticProjection(projection);
    return projection.result();
  }

  template <class TExecutionSpace>
  void LeptonFrontWorkspace<TExecutionSpace>::ensureOutputCapacity(
      std::size_t const stepCount, std::size_t const recordCount,
      std::size_t const photonCount, std::size_t const fallbackCount,
      std::size_t const observationCount, std::size_t const decayCount,
      bool const projectSteps, std::size_t const sourceHardLimit,
      std::size_t const photonHardLimit, TExecutionSpace const& execution) {
    auto const needsGrowth =
        stepCount > stepCapacity_ || recordCount > recordCapacity_ ||
        photonCount > photonCapacity_ || fallbackCount > fallbackCapacity_ ||
        observationCount > observationCapacity_ || decayCount > decayCapacity_ ||
        (projectSteps && stepCount > projectedStepCapacity_);
    if (!needsGrowth) return;

    // A preceding front can leave profile/radio work queued when no host
    // output is required.  Synchronize only on the rare growth path before
    // replacing any View still referenced by that work.
    execution.fence("grow resident Kokkos lepton output workspace");
    growView(steps_, stepCapacity_, stepCount, sourceHardLimit, execution,
             "c8_kokkos_resident_lepton_steps");
    if (projectSteps)
      growView(projectedSteps_, projectedStepCapacity_, stepCount, sourceHardLimit,
               execution, "c8_kokkos_resident_lepton_projected_steps");
    growView(records_, recordCapacity_, recordCount, sourceHardLimit, execution,
             "c8_kokkos_resident_lepton_records");
    growView(photons_, photonCapacity_, photonCount, photonHardLimit, execution,
             "c8_kokkos_resident_lepton_photons");
    growView(fallbacks_, fallbackCapacity_, fallbackCount, sourceHardLimit, execution,
             "c8_kokkos_resident_lepton_fallbacks");
    growView(observations_, observationCapacity_, observationCount, sourceHardLimit,
             execution, "c8_kokkos_resident_lepton_observations");
    growView(decays_, decayCapacity_, decayCount, sourceHardLimit, execution,
             "c8_kokkos_resident_lepton_decays");
  }

  template <class TExecutionSpace>
  MemoryProjection LeptonFrontWorkspace<TExecutionSpace>::projectedOutputCapacity(
      std::size_t const stepCount, std::size_t const recordCount,
      std::size_t const photonCount, std::size_t const fallbackCount,
      std::size_t const observationCount, std::size_t const decayCount,
      bool const projectSteps, std::size_t const sourceHardLimit,
      std::size_t const photonHardLimit) const {
    MemoryProjectionBuilder projection(deviceBytes());
    appendGrowViewProjection<LeptonTransportRecord>(projection, steps_, stepCapacity_,
                                                    stepCount, sourceHardLimit);
    if (projectSteps)
      appendGrowViewProjection<ProjectedStepRecord>(projection, projectedSteps_,
                                                    projectedStepCapacity_, stepCount,
                                                    sourceHardLimit);
    appendGrowViewProjection<LeptonFinalStateRecord>(
        projection, records_, recordCapacity_, recordCount, sourceHardLimit);
    appendGrowViewProjection<ParticleState>(projection, photons_, photonCapacity_,
                                            photonCount, photonHardLimit);
    appendGrowViewProjection<FallbackRequest>(projection, fallbacks_, fallbackCapacity_,
                                              fallbackCount, sourceHardLimit);
    appendGrowViewProjection<ObservationRecord>(projection, observations_,
                                                observationCapacity_, observationCount,
                                                sourceHardLimit);
    appendGrowViewProjection<ParticleState>(projection, decays_, decayCapacity_,
                                            decayCount, sourceHardLimit);
    return projection.result();
  }

  template <class TExecutionSpace>
  MemoryProjection LeptonFrontWorkspace<TExecutionSpace>::projectedBucketingCapacity(
      std::size_t const requested, TExecutionSpace const& execution) const {
    MemoryProjectionBuilder projection(deviceBytes());
    projection.merge(bucketing_.deviceBytes(),
                     bucketing_.projectedCapacity(requested, execution));
    return projection.result();
  }

  template <class TExecutionSpace>
  void LeptonFrontWorkspace<TExecutionSpace>::ensureHostOutputCapacity(
      std::size_t const projectedCount, std::size_t const stepCount,
      std::size_t const recordCount, std::size_t const photonCount,
      std::size_t const fallbackCount, std::size_t const observationCount,
      std::size_t const decayCount, std::size_t const sourceHardLimit,
      std::size_t const photonHardLimit) {
    growHostView(hostProjectedSteps_, projectedCount, sourceHardLimit,
                 "c8_kokkos_resident_lepton_host_projected_steps");
    growHostView(hostSteps_, stepCount, sourceHardLimit,
                 "c8_kokkos_resident_lepton_host_steps");
    growHostView(hostRecords_, recordCount, sourceHardLimit,
                 "c8_kokkos_resident_lepton_host_records");
    growHostView(hostPhotons_, photonCount, photonHardLimit,
                 "c8_kokkos_resident_lepton_host_photons");
    growHostView(hostFallbacks_, fallbackCount, sourceHardLimit,
                 "c8_kokkos_resident_lepton_host_fallbacks");
    growHostView(hostObservations_, observationCount, sourceHardLimit,
                 "c8_kokkos_resident_lepton_host_observations");
    growHostView(hostDecays_, decayCount, sourceHardLimit,
                 "c8_kokkos_resident_lepton_host_decays");
  }

  template <class TExecutionSpace>
  std::size_t LeptonFrontWorkspace<TExecutionSpace>::capacity() const noexcept {
    return queue_.capacity();
  }

  template <class TExecutionSpace>
  std::size_t LeptonFrontWorkspace<TExecutionSpace>::deviceBytes() const noexcept {
    auto const viewBytes = [](auto const& view) {
      // A default-constructed rank-zero Kokkos::View can report span()==1
      // even though data()==nullptr and no allocation exists.
      return (view.data() == nullptr ? 0 : view.span()) *
             sizeof(typename std::decay_t<decltype(view)>::value_type);
    };
    // error, interaction_count, vertex_interaction_count and scan_totals are
    // unmanaged aliases into front_control and are intentionally omitted.
    return queue_.deviceBytes() + bucketing_.deviceBytes() + viewBytes(selections_) +
           viewBytes(transports_) + viewBytes(transportFallbacks_) +
           viewBytes(vertices_) + viewBytes(finalStates_) + viewBytes(counts_) +
           viewBytes(interactionFlags_) + viewBytes(transportStates_) +
           viewBytes(interactionSources_) + viewBytes(offsets_) + viewBytes(steps_) +
           viewBytes(projectedSteps_) + viewBytes(records_) + viewBytes(photons_) +
           viewBytes(fallbacks_) + viewBytes(observations_) + viewBytes(decays_) +
           viewBytes(firstSnapshots_) + viewBytes(firstInteractionCandidates_) +
           viewBytes(callStatistics_) + viewBytes(frontControl_);
  }

  template <class TExecutionSpace>
  template <class TView>
  std::size_t LeptonFrontWorkspace<TExecutionSpace>::viewBytes(
      TView const& view) noexcept {
    return (view.data() == nullptr ? 0 : view.span()) *
           sizeof(typename TView::value_type);
  }

  template <class TExecutionSpace>
  template <class TElement, class TView>
  void LeptonFrontWorkspace<TExecutionSpace>::replaceViewProjection(
      MemoryProjectionBuilder& projection, TView const& view, std::size_t const extent) {
    projection.replace(viewBytes(view),
                       checked_memory_multiply(extent, sizeof(TElement)));
  }

  template <class TExecutionSpace>
  template <class TElement, class TView>
  void LeptonFrontWorkspace<TExecutionSpace>::appendGrowViewProjection(
      MemoryProjectionBuilder& projection, TView const& view, std::size_t const current,
      std::size_t const requested, std::size_t const hardLimit) {
    if (requested <= current) return;
    replaceViewProjection<TElement>(projection, view,
                                    grownCapacity(current, requested, hardLimit));
  }

  template <class TExecutionSpace>
  void LeptonFrontWorkspace<TExecutionSpace>::appendDiagnosticProjection(
      MemoryProjectionBuilder& projection) const {
    if (firstSnapshots_.extent(0) == 0) {
      projection.replace(0, sizeof(FirstInteractionRecord));
      projection.replace(0, sizeof(std::uint32_t));
    }
    if (callStatistics_.extent(0) == 0)
      projection.replace(
          0, checked_memory_multiply(LeptonCallStatisticCount, sizeof(std::uint64_t)));
    if (frontControl_.data() == nullptr)
      projection.replace(0, sizeof(ResidentLeptonFrontControl));
  }

  template <class TExecutionSpace>
  std::size_t LeptonFrontWorkspace<TExecutionSpace>::grownCapacity(
      std::size_t const current, std::size_t const requested,
      std::size_t const hardLimit) {
    if (requested > hardLimit)
      throw std::length_error(
          "resident Kokkos lepton allocation exceeds its logical limit");
    if (requested <= current) return current;
    if (current == 0) return requested;
    if (current > hardLimit / 2) return hardLimit;
    return std::min(hardLimit, std::max(requested, current * 2));
  }

  template <class TExecutionSpace>
  template <class TView>
  void LeptonFrontWorkspace<TExecutionSpace>::growView(TView& view, std::size_t& current,
                                                       std::size_t const requested,
                                                       std::size_t const hardLimit,
                                                       TExecutionSpace const& execution,
                                                       char const* const label) {
    if (requested <= current) return;
    auto const grown = grownCapacity(current, requested, hardLimit);
    view = TView(
        Kokkos::view_alloc(execution, Kokkos::WithoutInitializing, std::string(label)),
        grown);
    current = grown;
  }

  template <class TExecutionSpace>
  template <class TView>
  void LeptonFrontWorkspace<TExecutionSpace>::growHostView(TView& view,
                                                           std::size_t const requested,
                                                           std::size_t const hardLimit,
                                                           char const* const label) {
    if (requested <= view.extent(0)) return;
    auto const grown = grownCapacity(view.extent(0), requested, hardLimit);
    TView replacement(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                      grown);
    view = std::move(replacement);
  }

  template <class TExecutionSpace>
  void LeptonFrontWorkspace<TExecutionSpace>::ensureDiagnosticViews(
      TExecutionSpace const& execution) {
    // There can be only one generation-zero projectile in a shower.  Keep
    // the same one-snapshot/candidate-counter layout as native CUDA rather
    // than scaling this diagnostic allocation with the lepton wavefront.
    if (firstSnapshots_.extent(0) == 0) {
      firstSnapshots_ = decltype(firstSnapshots_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_first_snapshot"),
          1);
      firstInteractionCandidates_ = decltype(firstInteractionCandidates_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_first_candidates"),
          1);
    }
    if (callStatistics_.extent(0) == 0)
      callStatistics_ = decltype(callStatistics_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_call_statistics"),
          LeptonCallStatisticCount);
    if (hostCallStatistics_.extent(0) == 0)
      hostCallStatistics_ = decltype(hostCallStatistics_)(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_host_call_statistics"),
          LeptonCallStatisticCount);
    if (frontControl_.data() == nullptr) {
      frontControl_ = decltype(frontControl_)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_front_control"));
      hostFrontControl_ =
          decltype(hostFrontControl_)("c8_kokkos_resident_lepton_host_front_control");
      auto* const controlBytes = reinterpret_cast<unsigned char*>(frontControl_.data());
      scanTotals_ = scan_totals_view_type(reinterpret_cast<ResidentLeptonScanValue*>(
          controlBytes + offsetof(ResidentLeptonFrontControl, totals_)));
      interactionCount_ = interaction_count_view_type(reinterpret_cast<std::uint64_t*>(
          controlBytes + offsetof(ResidentLeptonFrontControl, interactionCount_)));
      vertexInteractionCount_ =
          interaction_count_view_type(reinterpret_cast<std::uint64_t*>(
              controlBytes +
              offsetof(ResidentLeptonFrontControl, vertexInteractionCount_)));
      error_ = error_view_type(
          reinterpret_cast<std::uint32_t*>(
              controlBytes + offsetof(ResidentLeptonFrontControl, materializationError_)),
          1);
      hostFirstInteractionCandidates_ = decltype(hostFirstInteractionCandidates_)(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_host_first_candidates"),
          1);
      hostFirstSnapshots_ = decltype(hostFirstSnapshots_)(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_host_first_snapshot"),
          1);
    }
  }

} // namespace corsika::kokkos
