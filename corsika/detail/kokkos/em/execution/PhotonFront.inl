/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace photon_front_detail {
    KOKKOS_INLINE_FUNCTION std::uint64_t photon_fallback_output_offset(
        std::uint32_t const stage, std::uint64_t const selectionOffset,
        std::uint64_t const transportOffset, std::uint64_t const finalStateOffset,
        ResidentPhotonScanValue const& totals) {
      if (stage == PhotonSelectionFallback) return selectionOffset;
      if (stage == PhotonTransportFallback)
        return totals.values_[PhotonSelectionFallbackOffset] + transportOffset;
      if (stage == PhotonFinalStateFallback)
        return totals.values_[PhotonSelectionFallbackOffset] +
               totals.values_[PhotonTransportFallbackOffset] + finalStateOffset;
      return ~std::uint64_t{0};
    }
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentPhotonScanFunctor<TExecutionSpace>::init(
        value_type& value) const {
      value = {};
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentPhotonScanFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      for (std::size_t column = 0; column < PhotonOffsetCount; ++column)
        destination.values_[column] += source.values_[column];
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentPhotonScanFunctor<TExecutionSpace>::operator()(
        std::size_t const source, value_type& update, bool const final) const {
      if (final) {
        for (std::size_t column = 0; column < PhotonOffsetCount; ++column)
          offsets_(source, column) = update.values_[column];
      }
      auto const& sourceCounts = counts_(source);
      update.values_[PhotonChildOffset] += sourceCounts.child_;
      update.values_[PhotonNextOffset] += sourceCounts.next_;
      update.values_[PhotonChargedOffset] += sourceCounts.charged_;
      update.values_[PhotonSelectionFallbackOffset] +=
          sourceCounts.fallbackStage_ == PhotonSelectionFallback;
      update.values_[PhotonTransportFallbackOffset] +=
          sourceCounts.fallbackStage_ == PhotonTransportFallback;
      update.values_[PhotonFinalStateFallbackOffset] +=
          sourceCounts.fallbackStage_ == PhotonFinalStateFallback;
      update.values_[PhotonObservationOffset] += sourceCounts.observation_;
      update.values_[PhotonStepOffset] += sourceCounts.step_;
      update.values_[PhotonRecordOffset] += sourceCounts.record_;
    }
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonCompactionScanFunctor<TExecutionSpace>::init(value_type& value) const {
      value = 0;
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentPhotonCompactionScanFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      destination += source;
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentPhotonCompactionScanFunctor<TExecutionSpace>::
    operator()(std::size_t const source, value_type& update, bool const final) const {
      auto const selected = flags_(source) != 0;
      if (final && selected) sources_(static_cast<std::size_t>(update)) = source;
      update += selected;
    }
    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::fallback(
        FallbackRequest const& value) {
      fallbacks_(fallbackOffset_) = value;
    }

    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::continuation(
        InteractionRecord const& value) {
      next_.store(nextOffset_, value.particle_);
    }

    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::suppression(
        ParticleState const& parent, InteractionRecord const&,
        PhotonFinalStateParameters const&) {
      next_.store(nextOffset_, parent);
    }

    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::firstInteraction(
        ParticleState const& parent, std::int32_t processId, ParticleState const& first,
        ParticleState const* second) {
      if (captureFirstInteraction_ == 0 || parent.generation_ != 0) return;
      auto const candidate =
          Kokkos::atomic_fetch_add(firstInteractionCandidates_, std::uint32_t{1});
      if (candidate != 0) return;
      auto& snapshot = firstSnapshots_(0);
      snapshot.parentAtVertex_ = parent;
      snapshot.processId_ = processId;
      snapshot.secondaryCount_ = second == nullptr ? 1U : 2U;
      snapshot.secondaries_[0] = first;
      snapshot.secondaries_[0].weight_ = parent.weight_;
      if (second != nullptr) {
        snapshot.secondaries_[1] = *second;
        snapshot.secondaries_[1].weight_ = parent.weight_;
      }
    }

    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::secondary(
        std::uint32_t, ParticleState const& particle) {
      if (particle.pid_ == static_cast<std::int32_t>(EmPid::Photon)) {
        next_.store(nextOffset_ + nextWritten_++, particle);
      } else {
        charged_(chargedOffset_ + chargedWritten_++) = particle;
      }
    }

    KOKKOS_INLINE_FUNCTION PhotonFinalStateRecord&
    ResidentPhotonFinalStateWriter::record() {
      return records_(recordOffset_);
    }

    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::finishRecord(
        std::uint32_t) {}

    KOKKOS_INLINE_FUNCTION void ResidentPhotonFinalStateWriter::fail(
        std::uint32_t value) {
      Kokkos::atomic_compare_exchange(error_, std::uint32_t{0}, value);
    }
  } // namespace photon_front_detail
  template <class TExecutionSpace>
  void enqueue_photon_front(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      ParticleSoARawView const current, ParticleSoARawView const next,
      PhotonFrontWorkspace<TExecutionSpace>& workspace, std::size_t const currentCount,
      std::size_t const capacity, std::uint64_t const nextHistoryId,
      bool const projectSteps, bool const captureFirstInteraction,
      TExecutionSpace const& execution, std::size_t const openmpChunkSize) {
    using namespace photon_front_detail;

    auto const policy = [openmpChunkSize](TExecutionSpace const& ex, std::size_t,
                                          std::size_t end) {
      return make_range_policy(ex, end, openmpChunkSize);
    };
    auto const& selections = workspace.selections_;
    auto const& transports = workspace.transports_;
    auto const& transportFallbacks = workspace.transportFallbacks_;
    auto const& finalStates = workspace.finalStates_;
    auto const& counts = workspace.counts_;
    auto const& interactionFlags = workspace.interactionFlags_;
    auto const& interactionSources = workspace.interactionSources_;
    auto const& offsets = workspace.offsets_;
    auto const& steps = workspace.steps_;
    auto const& projectedSteps = workspace.projectedSteps_;
    auto const& records = workspace.records_;
    auto const& charged = workspace.charged_;
    auto const& fallbacks = workspace.fallbacks_;
    auto const& observations = workspace.observations_;
    auto const& firstSnapshots = workspace.firstSnapshots_;
    auto const& firstInteractionCandidates = workspace.firstInteractionCandidates_;
    auto const& error = workspace.error_;
    auto const interactionCountDevice = workspace.interactionCount_;
    auto const scanTotalsDevice = workspace.scanTotals_;
    auto const chargedCapacity = capacity * 2;
    auto const selectionsRaw = raw_device_view(selections);
    auto const transportsRaw = raw_device_view(transports);
    auto const transportFallbacksRaw = raw_device_view(transportFallbacks);
    auto const countsRaw = raw_device_view(counts);
    auto const interactionFlagsRaw = raw_device_view(interactionFlags);
    auto const interactionSourcesRaw = raw_device_view(interactionSources);
    auto const selectKernel = KOKKOS_LAMBDA(std::size_t const source) {
      auto const& context = physicsContext();
      ResidentPhotonSourceCounts sourceCounts{};
      auto const particle = current.load(source);
      auto& selection = selectionsRaw(source);
      selection = select_discrete_interaction(context.physics_, particle, source,
                                              context.randomSeed_, context.showerId_);
      if (selection.fallbackFlag_ != 0) {
        sourceCounts.fallbackStage_ = PhotonSelectionFallback;
      }
      countsRaw(source) = sourceCounts;
    };
    static_assert(sizeof(selectKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_photon_select",
                         policy(execution, 0, currentCount), selectKernel);

    // Transport is a separate kernel so its geometry state does not share
    // registers with rate inversion.  Selection fallbacks return before
    // transport without introducing an additional scan of the full front.
    auto const transportKernel = KOKKOS_LAMBDA(std::size_t const source) {
      auto const& context = physicsContext();
      auto const& selection = selectionsRaw(source);
      auto& sourceCounts = countsRaw(source);
      interactionFlagsRaw(source) = 0;
      if (selection.fallbackFlag_ != 0) return;
      auto const transport = transport_photon(context.environment_, context.geometry_,
                                              selection.interaction_);
      transportsRaw(source) = transport.record_;
      transportFallbacksRaw(source) = transport.fallback_;
      if (transport.fallbackFlag_ != 0) {
        sourceCounts.fallbackStage_ = PhotonTransportFallback;
        return;
      }
      sourceCounts.step_ = 1;
      auto const& step = transport.record_;
      if (step.limit_ == PhotonTransportLimit::LayerBoundary) {
        sourceCounts.next_ = 1;
      } else if (step.limit_ == PhotonTransportLimit::ObservationSurface ||
                 step.limit_ == PhotonTransportLimit::EscapedEnvironment ||
                 (step.limit_ == PhotonTransportLimit::ParticleCut &&
                  step.observationSurfaceReachedBeforeCut_ != 0)) {
        sourceCounts.observation_ = 1;
      } else if (step.limit_ == PhotonTransportLimit::Interaction) {
        interactionFlagsRaw(source) = 1;
      }
    };
    static_assert(sizeof(transportKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_photon_transport",
                         policy(execution, 0, currentCount), transportKernel);

    // Only true interaction vertices execute the comparatively heavy LPM,
    // final-state kinematics and thinning classifier.  A second stable scan
    // keeps their order identical to the original source wavefront.
    Kokkos::parallel_scan("c8_kokkos_scan_photon_interactions",
                          policy(execution, 0, currentCount),
                          ResidentPhotonCompactionScanFunctor<TExecutionSpace>{
                              interactionFlags, interactionSources},
                          workspace.interactionCount_);
    auto const finalStatesRaw = raw_device_view(finalStates);
    auto const interactionCountRaw = interactionCountDevice.data();
    auto const finalStateKernel = KOKKOS_LAMBDA(std::size_t const interactionIndex) {
      auto const& context = physicsContext();
      auto const compactedCount = static_cast<std::size_t>(*interactionCountRaw);
      if (interactionIndex >= compactedCount) return;
      auto const source = interactionSourcesRaw(interactionIndex);
      auto& sourceCounts = countsRaw(source);
      auto const& step = transportsRaw(source);
      auto& final = finalStatesRaw(interactionIndex);
      final = classify_photon_final_state(
          context.physics_.getContinuousView(), context.photonPairLpm_, context.thinning_,
          step.interaction_, context.randomSeed_, context.showerId_);
      sourceCounts.child_ = final.childCount_;
      sourceCounts.record_ = final.recordFlag_;
      if (final.fallbackFlag_ != 0)
        sourceCounts.fallbackStage_ = PhotonFinalStateFallback;
      if (final.suppressionFlag_ || final.continuationFlag_) {
        sourceCounts.next_ = 1;
      } else if (final.recordFlag_) {
        auto const process = final.parameters_.processId_;
        if (process == ComptonProcessId) {
          sourceCounts.next_ = (final.parameters_.thinningKeepMask_ & 0x1U) != 0;
          sourceCounts.charged_ = (final.parameters_.thinningKeepMask_ & 0x2U) != 0;
        } else {
          sourceCounts.charged_ = final.childCount_;
        }
      }
    };
    static_assert(sizeof(finalStateKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_photon_final_state",
                         policy(execution, 0, currentCount), finalStateKernel);

    Kokkos::parallel_scan("c8_kokkos_scan_photon_outputs",
                          policy(execution, 0, currentCount),
                          ResidentPhotonScanFunctor<TExecutionSpace>{counts, offsets},
                          workspace.scanTotals_);

    auto const offsetsRaw = raw_device_view(offsets);
    auto const stepsRaw = raw_device_view(steps);
    auto const projectedStepsRaw = raw_device_view(projectedSteps);
    auto const recordsRaw = raw_device_view(records);
    auto const chargedRaw = raw_device_view(charged);
    auto const fallbacksRaw = raw_device_view(fallbacks);
    auto const observationsRaw = raw_device_view(observations);
    auto const firstSnapshotsRaw = raw_device_view(firstSnapshots);
    auto const firstInteractionCandidatesRaw = firstInteractionCandidates.data();
    auto const scanTotalsRaw = scanTotalsDevice.data();
    auto const errorRaw = error.data();
    auto const materializeEndpoint = KOKKOS_LAMBDA(std::size_t const source) {
      auto const deviceTotals = *scanTotalsRaw;
      if (deviceTotals.values_[PhotonNextOffset] > capacity ||
          deviceTotals.values_[PhotonChargedOffset] > chargedCapacity)
        return;
      auto const& selection = selectionsRaw(source);
      auto const& transport = transportsRaw(source);
      auto const& sourceCounts = countsRaw(source);
      if (sourceCounts.fallbackStage_ > PhotonFinalStateFallback) {
        Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{21});
        return;
      }
      if (sourceCounts.step_) {
        stepsRaw(offsetsRaw(source, PhotonStepOffset)) = transport;
        if (projectSteps) {
          auto const& context = physicsContext();
          projectedStepsRaw(offsetsRaw(source, PhotonStepOffset)) =
              project_photon_step(context.profileProjection_, transport);
        }
      }
      if (selection.fallbackFlag_) {
        if (sourceCounts.fallbackStage_ != PhotonSelectionFallback) {
          Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{22});
          return;
        }
        fallbacksRaw(photon_fallback_output_offset(
            sourceCounts.fallbackStage_,
            offsetsRaw(source, PhotonSelectionFallbackOffset),
            offsetsRaw(source, PhotonTransportFallbackOffset),
            offsetsRaw(source, PhotonFinalStateFallbackOffset), deviceTotals)) =
            selection.fallback_;
        return;
      }
      if (!sourceCounts.step_) {
        if (sourceCounts.fallbackStage_ != PhotonTransportFallback) {
          Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{23});
          return;
        }
        fallbacksRaw(photon_fallback_output_offset(
            sourceCounts.fallbackStage_,
            offsetsRaw(source, PhotonSelectionFallbackOffset),
            offsetsRaw(source, PhotonTransportFallbackOffset),
            offsetsRaw(source, PhotonFinalStateFallbackOffset), deviceTotals)) =
            transportFallbacksRaw(source);
        return;
      }
      auto const& step = transport;
      if (sourceCounts.observation_) {
        ObservationRecord observation{};
        observation.particle_ = step.end_;
        observation.status_ = step.limit_ == PhotonTransportLimit::EscapedEnvironment
                                  ? ObservationStatus::EscapedEnvironment
                                  : ObservationStatus::ReachedObservationSurface;
        observationsRaw(offsetsRaw(source, PhotonObservationOffset)) = observation;
      }
      if (step.limit_ == PhotonTransportLimit::LayerBoundary) {
        next.store(offsetsRaw(source, PhotonNextOffset), step.end_);
      }
    };
    static_assert(sizeof(materializeEndpoint) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_photon_materialize_endpoints",
                         policy(execution, 0, currentCount), materializeEndpoint);

    // Materializing an interaction carries the complete final-state
    // kinematics and therefore substantially more live state than writing
    // a transport endpoint.  Run it only for the stable interaction list;
    // output offsets and history IDs still come from the source-ordered
    // composite scan above.
    auto const materializeKernel = KOKKOS_LAMBDA(std::size_t const interactionIndex) {
      auto const deviceTotals = *scanTotalsRaw;
      if (deviceTotals.values_[PhotonNextOffset] > capacity ||
          deviceTotals.values_[PhotonChargedOffset] > chargedCapacity)
        return;
      auto const compactedCount = static_cast<std::size_t>(*interactionCountRaw);
      if (interactionIndex >= compactedCount) return;
      auto const source = interactionSourcesRaw(interactionIndex);
      auto const& step = transportsRaw(source);
      auto const& final = finalStatesRaw(interactionIndex);
      auto const& sourceCounts = countsRaw(source);
      if ((final.fallbackFlag_ != 0) !=
          (sourceCounts.fallbackStage_ == PhotonFinalStateFallback)) {
        Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{24});
        return;
      }
      ResidentPhotonFinalStateWriter output{
          next,
          recordsRaw,
          chargedRaw,
          fallbacksRaw,
          firstSnapshotsRaw,
          firstInteractionCandidatesRaw,
          errorRaw,
          offsetsRaw(source, PhotonNextOffset),
          offsetsRaw(source, PhotonChargedOffset),
          photon_fallback_output_offset(
              sourceCounts.fallbackStage_,
              offsetsRaw(source, PhotonSelectionFallbackOffset),
              offsetsRaw(source, PhotonTransportFallbackOffset),
              offsetsRaw(source, PhotonFinalStateFallbackOffset), deviceTotals),
          offsetsRaw(source, PhotonRecordOffset),
          captureFirstInteraction ? 1U : 0U};
      materialize_photon_final_state_in_place(step.interaction_, final,
                                              offsetsRaw(source, PhotonChildOffset),
                                              nextHistoryId, output);
    };
    static_assert(sizeof(materializeKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_photon_materialize_interactions",
                         policy(execution, 0, currentCount), materializeKernel);
  }

} // namespace corsika::kokkos
