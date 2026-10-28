/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace lepton_front_detail {
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentLeptonScanFunctor<TExecutionSpace>::init(
        value_type& value) const {
      value = {};
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentLeptonScanFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      for (std::size_t column = 0; column < LeptonOffsetCount; ++column)
        destination.values_[column] += source.values_[column];
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentLeptonScanFunctor<TExecutionSpace>::operator()(
        std::size_t const source, value_type& update, bool const final) const {
      if (final) {
        for (std::size_t column = 0; column < LeptonOffsetCount; ++column)
          offsets_(source, column) = update.values_[column];
      }
      auto const& sourceCounts = counts_(source);
      update.values_[LeptonChildOffset] += sourceCounts.child_;
      update.values_[LeptonNextOffset] += sourceCounts.next_;
      update.values_[LeptonPhotonOffset] += sourceCounts.photon_;
      update.values_[LeptonSelectionFallbackOffset] +=
          sourceCounts.fallbackStage_ == LeptonSelectionFallback;
      update.values_[LeptonTransportFallbackOffset] +=
          sourceCounts.fallbackStage_ == LeptonTransportFallback;
      update.values_[LeptonVertexFallbackOffset] +=
          sourceCounts.fallbackStage_ == LeptonVertexFallback;
      update.values_[LeptonFinalStateFallbackOffset] +=
          sourceCounts.fallbackStage_ == LeptonFinalStateFallback;
      update.values_[LeptonObservationOffset] += sourceCounts.observation_;
      update.values_[LeptonDecayOffset] += sourceCounts.decay_;
      update.values_[LeptonStepOffset] += sourceCounts.step_;
      update.values_[LeptonRecordOffset] += sourceCounts.record_;
    }
    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentLeptonInteractionScanFunctor<TExecutionSpace>::init(value_type& value) const {
      value = 0;
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void
    ResidentLeptonInteractionScanFunctor<TExecutionSpace>::join(
        value_type& destination, value_type const& source) const {
      destination += source;
    }

    template <class TExecutionSpace>
    KOKKOS_INLINE_FUNCTION void ResidentLeptonInteractionScanFunctor<TExecutionSpace>::
    operator()(std::size_t const source, value_type& update, bool const final) const {
      auto const selected = flags_(source) != 0;
      if (final && selected) sources_(static_cast<std::size_t>(update)) = source;
      update += selected;
    }
    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::fallback(
        FallbackRequest const& value) {
      fallbacks_(fallbackOffset_) = value;
    }

    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::continuation(
        InteractionRecord const& value) {
      next_.store(nextOffset_, value.particle_);
    }

    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::suppression(
        ParticleState const& parent, InteractionRecord const&,
        LeptonFinalStateParameters const&) {
      next_.store(nextOffset_, parent);
    }

    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::firstInteraction(
        ParticleState const& parent, std::int32_t processId, ParticleState const& first,
        ParticleState const* second, ParticleState const* third) {
      if (captureFirstInteraction_ == 0 || parent.generation_ != 0) return;
      auto const candidate =
          Kokkos::atomic_fetch_add(firstInteractionCandidates_, std::uint32_t{1});
      if (candidate != 0) return;
      auto& snapshot = firstSnapshots_(0);
      snapshot.parentAtVertex_ = parent;
      snapshot.processId_ = processId;
      snapshot.secondaryCount_ = third ? 3U : (second ? 2U : 1U);
      snapshot.secondaries_[0] = first;
      snapshot.secondaries_[0].weight_ = parent.weight_;
      if (second) {
        snapshot.secondaries_[1] = *second;
        snapshot.secondaries_[1].weight_ = parent.weight_;
      }
      if (third) {
        snapshot.secondaries_[2] = *third;
        snapshot.secondaries_[2].weight_ = parent.weight_;
      }
    }

    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::secondary(
        std::uint32_t, ParticleState const& particle) {
      if (is_charged_lepton_pid(particle.pid_)) {
        next_.store(nextOffset_ + nextWritten_++, particle);
      } else if (particle.pid_ == static_cast<std::int32_t>(EmPid::Photon)) {
        photons_(photonOffset_ + photonWritten_++) = particle;
      } else {
        fail(2);
      }
    }

    KOKKOS_INLINE_FUNCTION LeptonFinalStateRecord&
    ResidentLeptonFinalStateWriter::record() {
      return records_(recordOffset_);
    }

    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::finishRecord(
        std::uint32_t) {}

    KOKKOS_INLINE_FUNCTION void ResidentLeptonFinalStateWriter::fail(
        std::uint32_t value) {
      Kokkos::atomic_compare_exchange(error_, std::uint32_t{0}, value);
    }
    KOKKOS_INLINE_FUNCTION std::uint64_t lepton_fallback_output_offset(
        std::uint32_t const stage, std::uint64_t const selectionOffset,
        std::uint64_t const transportOffset, std::uint64_t const vertexOffset,
        std::uint64_t const finalStateOffset, ResidentLeptonScanValue const& totals) {
      if (stage == LeptonSelectionFallback) return selectionOffset;
      if (stage == LeptonTransportFallback)
        return totals.values_[LeptonSelectionFallbackOffset] + transportOffset;
      if (stage == LeptonVertexFallback)
        return totals.values_[LeptonSelectionFallbackOffset] +
               totals.values_[LeptonTransportFallbackOffset] + vertexOffset;
      if (stage == LeptonFinalStateFallback)
        return totals.values_[LeptonSelectionFallbackOffset] +
               totals.values_[LeptonTransportFallbackOffset] +
               totals.values_[LeptonVertexFallbackOffset] + finalStateOffset;
      return ~std::uint64_t{0};
    }
    KOKKOS_INLINE_FUNCTION void classify_lepton_transport_limit(
        LeptonTransportRecord const& step, ResidentLeptonSourceCounts& sourceCounts) {

      sourceCounts.step_ = 1;
      switch (step.limit_) {
        case LeptonTransportLimit::ContinuousStep:
        case LeptonTransportLimit::MaterialBoundary:
        case LeptonTransportLimit::LayerBoundary:
        case LeptonTransportLimit::MagneticStep:
          sourceCounts.next_ = 1;
          return;
        case LeptonTransportLimit::ObservationSurface:
        case LeptonTransportLimit::EscapedEnvironment:
          sourceCounts.observation_ = 1;
          return;
        case LeptonTransportLimit::ParticleCut:
          sourceCounts.observation_ = step.observationSurfaceReachedBeforeCut_ != 0;
          return;
        case LeptonTransportLimit::DecayCandidate:
          sourceCounts.decay_ = 1;
          return;
        case LeptonTransportLimit::InteractionCandidate:
          break;
      }
    }
    KOKKOS_INLINE_FUNCTION void classify_lepton_interaction_source(
        InteractionPhysicsView const& physics, BremsLpmSnapshot const& bremsLpm,
        EmThinningConfig const& thinning, std::uint64_t const randomSeed,
        std::uint64_t const showerId, LeptonTransportRecord const& step,
        LeptonVertexSelectionOutcome& vertex, LeptonFinalStateClassification& finalState,
        ResidentLeptonSourceCounts& sourceCounts) {

      vertex = select_lepton_vertex(physics, step.interaction_, randomSeed, showerId);
      if (vertex.fallbackFlag_ != 0) {
        sourceCounts.fallbackStage_ = LeptonVertexFallback;
        return;
      }
      if (vertex.continuationFlag_ != 0) {
        sourceCounts.next_ = 1;
        return;
      }
      if (vertex.interactionFlag_ == 0) {
        sourceCounts.fallbackStage_ = LeptonVertexFallback;
        vertex.fallback_ = lepton_vertex_detail::vertex_fallback(
            step.interaction_, FallbackReason::InvalidFinalState);
        vertex.fallbackFlag_ = 1;
        return;
      }

      finalState = classify_lepton_final_state(bremsLpm, thinning, vertex.record_,
                                               randomSeed, showerId);
      auto const& final = finalState;
      sourceCounts.child_ = final.childCount_;
      sourceCounts.record_ = final.recordFlag_;
      if (final.fallbackFlag_ != 0)
        sourceCounts.fallbackStage_ = LeptonFinalStateFallback;
      if (final.continuationFlag_ || final.suppressionFlag_) {
        sourceCounts.next_ = 1;
        return;
      }
      if (!final.recordFlag_) return;

      auto const process = final.parameters_.processId_;
      if (process == BremsProcessId) {
        sourceCounts.next_ = (final.parameters_.thinningKeepMask_ & 0x1U) != 0;
        sourceCounts.photon_ = (final.parameters_.thinningKeepMask_ & 0x2U) != 0;
      } else if (process == AnnihilationProcessId) {
        sourceCounts.photon_ = final.childCount_;
      } else if (process == IonizationProcessId) {
        sourceCounts.next_ = final.childCount_;
      } else if (process == ElectronPairProcessId) {
        sourceCounts.next_ = final.childCount_;
      }
    }
  } // namespace lepton_front_detail
  template <class TExecutionSpace>
  void enqueue_lepton_front(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      double const electronMassGeV, bool const airMoliereFastPath,
      ParticleSoARawView const current,
      LeptonFrontWorkspace<TExecutionSpace>& activeWorkspace,
      std::size_t const currentCount, std::size_t const capacity,
      std::size_t const photonCapacity, std::uint64_t const nextHistoryId,
      std::uint64_t const secondaryHistoryIdLimitExclusive, bool const projectSteps,
      bool const captureFirstInteraction, TExecutionSpace const& execution,
      std::size_t const openmpChunkSize) {
    using namespace lepton_front_detail;

    auto const policy = [openmpChunkSize](TExecutionSpace const& ex, std::size_t begin,
                                          std::size_t end) {
      if (begin != 0 || end < begin)
        throw std::invalid_argument("resident lepton RangePolicy requires a zero origin");
      return make_range_policy(ex, end, openmpChunkSize);
    };
    auto& queue = activeWorkspace.queue_;
    auto const& selections = activeWorkspace.selections_;
    auto const& transports = activeWorkspace.transports_;
    auto const& transportFallbacks = activeWorkspace.transportFallbacks_;
    auto const& vertices = activeWorkspace.vertices_;
    auto const& finalStates = activeWorkspace.finalStates_;
    auto const& counts = activeWorkspace.counts_;
    auto const& interactionFlags = activeWorkspace.interactionFlags_;
    auto const& transportStates = activeWorkspace.transportStates_;
    auto const& interactionSources = activeWorkspace.interactionSources_;
    auto const& offsets = activeWorkspace.offsets_;
    auto const& steps = activeWorkspace.steps_;
    auto const& projectedSteps = activeWorkspace.projectedSteps_;
    auto const& records = activeWorkspace.records_;
    auto const& photons = activeWorkspace.photons_;
    auto const& fallbacks = activeWorkspace.fallbacks_;
    auto const& observations = activeWorkspace.observations_;
    auto const& decays = activeWorkspace.decays_;
    auto const& firstSnapshots = activeWorkspace.firstSnapshots_;
    auto const& firstInteractionCandidates = activeWorkspace.firstInteractionCandidates_;
    auto const& error = activeWorkspace.error_;
    auto const interactionCountDevice = activeWorkspace.interactionCount_;
    auto const scanTotalsDevice = activeWorkspace.scanTotals_;
    auto const frontControlDevice = activeWorkspace.frontControl_;

    // Keep rate inversion separate from geometry and multiple scattering.
    // The three kernels execute in source order on the same execution-space
    // instance, so splitting their register frames does not change source
    // indices, history allocation, or any history-keyed Philox draw.
    auto const selectionsRaw = raw_device_view(selections);
    auto const transportsRaw = raw_device_view(transports);
    auto const transportFallbacksRaw = raw_device_view(transportFallbacks);
    auto const verticesRaw = raw_device_view(vertices);
    auto const finalStatesRaw = raw_device_view(finalStates);
    auto const countsRaw = raw_device_view(counts);
    auto const interactionFlagsRaw = raw_device_view(interactionFlags);
    auto const transportStatesRaw = raw_device_view(transportStates);
    auto const interactionSourcesRaw = raw_device_view(interactionSources);
    auto const selectKernel = KOKKOS_LAMBDA(std::size_t const source) {
      auto const& context = physicsContext();
      ResidentLeptonSourceCounts sourceCounts{};
      auto& selection = selectionsRaw(source);
      selection =
          select_discrete_interaction(context.physics_, current.load(source), source,
                                      context.randomSeed_, context.showerId_);
      if (selection.fallbackFlag_ != 0)
        sourceCounts.fallbackStage_ = LeptonSelectionFallback;
      countsRaw(source) = sourceCounts;
      interactionFlagsRaw(source) = 0;
      transportStatesRaw(source) = selection.fallbackFlag_ != 0 ? 1U : 0U;
    };
    static_assert(sizeof(selectKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_lepton_select",
                         policy(execution, 0, currentCount), selectKernel);

    auto const transportKernel = KOKKOS_LAMBDA(std::size_t const source) {
      auto const& context = physicsContext();
      auto const& selection = selectionsRaw(source);
      if (selection.fallbackFlag_ != 0) return;
      transportStatesRaw(source) = transport_lepton(
          context.physics_.getContinuousView(), true, context.environment_,
          context.geometry_, selection.interaction_, transportsRaw(source),
          transportFallbacksRaw(source));
    };
    static_assert(sizeof(transportKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_lepton_transport",
                         policy(execution, 0, currentCount), transportKernel);

    if (airMoliereFastPath) {
      auto const moliereAirKernel = KOKKOS_LAMBDA(std::size_t const source) {
        auto const& context = physicsContext();
        auto const& selection = selectionsRaw(source);
        auto& transport = transportsRaw(source);
        auto& transportFallback = transportFallbacksRaw(source);
        auto& sourceCounts = countsRaw(source);
        if (selection.fallbackFlag_ != 0) return;
        auto const state = apply_moliere_scattering_stage<4>(
            context.electronMoliere_, context.muonMoliere_, context.moliereInterpolation_,
            context.muonMoliereAvailable_ != 0, context.randomSeed_, context.showerId_,
            transport, transportFallback, transportStatesRaw(source));
        transportStatesRaw(source) = state;
        if (state != 0) {
          sourceCounts.fallbackStage_ = LeptonTransportFallback;
          return;
        }
        classify_lepton_transport_limit(transport, sourceCounts);
        interactionFlagsRaw(source) =
            transport.limit_ == LeptonTransportLimit::InteractionCandidate ||
            transport.interaction_.status_ == InteractionStatus::AtRestAnnihilation;
      };
      static_assert(sizeof(moliereAirKernel) < 512);
      Kokkos::parallel_for("c8_kokkos_resident_lepton_moliere_air",
                           policy(execution, 0, currentCount), moliereAirKernel);
    } else {
      auto const moliereGeneralKernel = KOKKOS_LAMBDA(std::size_t const source) {
        auto const& context = physicsContext();
        auto const& selection = selectionsRaw(source);
        auto& transport = transportsRaw(source);
        auto& transportFallback = transportFallbacksRaw(source);
        auto& sourceCounts = countsRaw(source);
        if (selection.fallbackFlag_ != 0) return;
        auto const state = apply_moliere_scattering_stage<MaxMoliereComponents>(
            context.electronMoliere_, context.muonMoliere_, context.moliereInterpolation_,
            context.muonMoliereAvailable_ != 0, context.randomSeed_, context.showerId_,
            transport, transportFallback, transportStatesRaw(source));
        transportStatesRaw(source) = state;
        if (state != 0) {
          sourceCounts.fallbackStage_ = LeptonTransportFallback;
          return;
        }
        classify_lepton_transport_limit(transport, sourceCounts);
        interactionFlagsRaw(source) =
            transport.limit_ == LeptonTransportLimit::InteractionCandidate ||
            transport.interaction_.status_ == InteractionStatus::AtRestAnnihilation;
      };
      static_assert(sizeof(moliereGeneralKernel) < 512);
      Kokkos::parallel_for("c8_kokkos_resident_lepton_moliere_general",
                           policy(execution, 0, currentCount), moliereGeneralKernel);
    }

    // Stable interaction-candidate compaction.  This preserves increasing
    // source order exactly, while ensuring that vertex selection and the
    // comparatively register-heavy final-state classifier run only for
    // actual interaction limits.
    Kokkos::parallel_scan("c8_kokkos_scan_lepton_interactions",
                          policy(execution, 0, currentCount),
                          ResidentLeptonInteractionScanFunctor<TExecutionSpace>{
                              interactionFlags, interactionSources},
                          activeWorkspace.interactionCount_);
    auto const interactionCountRaw = interactionCountDevice.data();
    auto const vertexInteractionCountDevice = activeWorkspace.vertexInteractionCount_;
    auto const classifyKernel =
        KOKKOS_LAMBDA(std::size_t const candidate, std::uint64_t& vertexInteractions) {
      auto const& context = physicsContext();
      auto const compactedCount = static_cast<std::size_t>(*interactionCountRaw);
      if (candidate >= compactedCount) return;
      auto const source = interactionSourcesRaw(candidate);
      auto& sourceCounts = countsRaw(source);
      classify_lepton_interaction_source(
          context.physics_, context.bremsLpm_, context.thinning_, context.randomSeed_,
          context.showerId_, transportsRaw(source), verticesRaw(candidate),
          finalStatesRaw(candidate), sourceCounts);
      vertexInteractions += verticesRaw(candidate).interactionFlag_ != 0;
    };
    static_assert(sizeof(classifyKernel) < 512);
    Kokkos::parallel_reduce("c8_kokkos_classify_lepton_interactions",
                            policy(execution, 0, currentCount), classifyKernel,
                            vertexInteractionCountDevice);

    Kokkos::parallel_scan("c8_kokkos_scan_lepton_outputs",
                          policy(execution, 0, currentCount),
                          ResidentLeptonScanFunctor<TExecutionSpace>{counts, offsets},
                          activeWorkspace.scanTotals_);

    auto const next = queue.next().rawDeviceView();
    auto const offsetsRaw = raw_device_view(offsets);
    auto const stepsRaw = raw_device_view(steps);
    auto const projectedStepsRaw = raw_device_view(projectedSteps);
    auto const recordsRaw = raw_device_view(records);
    auto const photonsRaw = raw_device_view(photons);
    auto const fallbacksRaw = raw_device_view(fallbacks);
    auto const observationsRaw = raw_device_view(observations);
    auto const decaysRaw = raw_device_view(decays);
    auto const firstSnapshotsRaw = raw_device_view(firstSnapshots);
    auto const firstInteractionCandidatesRaw = firstInteractionCandidates.data();
    auto const scanTotalsRaw = scanTotalsDevice.data();
    auto const errorRaw = error.data();
    auto const materializeEndpoint = KOKKOS_LAMBDA(std::size_t const source) {
      auto const deviceTotals = *scanTotalsRaw;
      if (deviceTotals.values_[LeptonNextOffset] > capacity ||
          deviceTotals.values_[LeptonPhotonOffset] > photonCapacity ||
          deviceTotals.values_[LeptonChildOffset] >
              secondaryHistoryIdLimitExclusive - nextHistoryId)
        return;
      auto const& selection = selectionsRaw(source);
      auto const& transport = transportsRaw(source);
      auto const& sourceCounts = countsRaw(source);
      if (sourceCounts.fallbackStage_ > LeptonFinalStateFallback) {
        Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{31});
        return;
      }
      if (sourceCounts.step_) {
        stepsRaw(offsetsRaw(source, LeptonStepOffset)) = transport;
        if (projectSteps) {
          auto const& context = physicsContext();
          projectedStepsRaw(offsetsRaw(source, LeptonStepOffset)) =
              project_lepton_step(context.profileProjection_, transport);
        }
      }
      if (selection.fallbackFlag_) {
        if (sourceCounts.fallbackStage_ != LeptonSelectionFallback) {
          Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{32});
          return;
        }
        fallbacksRaw(lepton_fallback_output_offset(
            sourceCounts.fallbackStage_,
            offsetsRaw(source, LeptonSelectionFallbackOffset),
            offsetsRaw(source, LeptonTransportFallbackOffset),
            offsetsRaw(source, LeptonVertexFallbackOffset),
            offsetsRaw(source, LeptonFinalStateFallbackOffset), deviceTotals)) =
            selection.fallback_;
        return;
      }
      if (!sourceCounts.step_) {
        if (sourceCounts.fallbackStage_ != LeptonTransportFallback) {
          Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{33});
          return;
        }
        fallbacksRaw(lepton_fallback_output_offset(
            sourceCounts.fallbackStage_,
            offsetsRaw(source, LeptonSelectionFallbackOffset),
            offsetsRaw(source, LeptonTransportFallbackOffset),
            offsetsRaw(source, LeptonVertexFallbackOffset),
            offsetsRaw(source, LeptonFinalStateFallbackOffset), deviceTotals)) =
            transportFallbacksRaw(source);
        return;
      }

      auto const& step = transport;
      if (sourceCounts.observation_) {
        ObservationRecord observation{};
        observation.particle_ = step.end_;
        observation.status_ = step.limit_ == LeptonTransportLimit::EscapedEnvironment
                                  ? ObservationStatus::EscapedEnvironment
                                  : ObservationStatus::ReachedObservationSurface;
        observationsRaw(offsetsRaw(source, LeptonObservationOffset)) = observation;
      }
      if (sourceCounts.decay_)
        decaysRaw(offsetsRaw(source, LeptonDecayOffset)) = step.end_;
      if (step.limit_ == LeptonTransportLimit::ContinuousStep ||
          step.limit_ == LeptonTransportLimit::LayerBoundary ||
          step.limit_ == LeptonTransportLimit::MaterialBoundary ||
          step.limit_ == LeptonTransportLimit::MagneticStep) {
        next.store(offsetsRaw(source, LeptonNextOffset), step.end_);
      }
    };
    static_assert(sizeof(materializeEndpoint) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_lepton_materialize_endpoints",
                         policy(execution, 0, currentCount), materializeEndpoint);

    auto const materializeKernel = KOKKOS_LAMBDA(std::size_t const candidate) {
      auto const deviceTotals = *scanTotalsRaw;
      if (deviceTotals.values_[LeptonNextOffset] > capacity ||
          deviceTotals.values_[LeptonPhotonOffset] > photonCapacity ||
          deviceTotals.values_[LeptonChildOffset] >
              secondaryHistoryIdLimitExclusive - nextHistoryId)
        return;
      auto const compactedCount = static_cast<std::size_t>(*interactionCountRaw);
      if (candidate >= compactedCount) return;
      auto const source = interactionSourcesRaw(candidate);
      auto const& vertex = verticesRaw(candidate);
      auto const& finalState = finalStatesRaw(candidate);
      auto const& sourceCounts = countsRaw(source);
      if (vertex.fallbackFlag_) {
        if (sourceCounts.fallbackStage_ != LeptonVertexFallback) {
          Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{34});
          return;
        }
        fallbacksRaw(lepton_fallback_output_offset(
            sourceCounts.fallbackStage_,
            offsetsRaw(source, LeptonSelectionFallbackOffset),
            offsetsRaw(source, LeptonTransportFallbackOffset),
            offsetsRaw(source, LeptonVertexFallbackOffset),
            offsetsRaw(source, LeptonFinalStateFallbackOffset), deviceTotals)) =
            vertex.fallback_;
        return;
      }
      if (vertex.continuationFlag_) {
        if (sourceCounts.fallbackStage_ != LeptonNoFallback) {
          Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{35});
          return;
        }
        next.store(offsetsRaw(source, LeptonNextOffset), vertex.record_.particle_);
        return;
      }
      if ((finalState.fallbackFlag_ != 0) !=
          (sourceCounts.fallbackStage_ == LeptonFinalStateFallback)) {
        Kokkos::atomic_compare_exchange(errorRaw, std::uint32_t{0}, std::uint32_t{36});
        return;
      }

      ResidentLeptonFinalStateWriter output{
          next,
          recordsRaw,
          photonsRaw,
          fallbacksRaw,
          firstSnapshotsRaw,
          firstInteractionCandidatesRaw,
          errorRaw,
          offsetsRaw(source, LeptonNextOffset),
          offsetsRaw(source, LeptonPhotonOffset),
          lepton_fallback_output_offset(
              sourceCounts.fallbackStage_,
              offsetsRaw(source, LeptonSelectionFallbackOffset),
              offsetsRaw(source, LeptonTransportFallbackOffset),
              offsetsRaw(source, LeptonVertexFallbackOffset),
              offsetsRaw(source, LeptonFinalStateFallbackOffset), deviceTotals),
          offsetsRaw(source, LeptonRecordOffset),
          captureFirstInteraction ? 1U : 0U};
      materialize_lepton_final_state_in_place(vertex.record_, finalState,
                                              offsetsRaw(source, LeptonChildOffset),
                                              nextHistoryId, electronMassGeV, output);
    };
    static_assert(sizeof(materializeKernel) < 512);
    Kokkos::parallel_for("c8_kokkos_resident_lepton_materialize_interactions",
                         policy(execution, 0, currentCount), materializeKernel);
  }

} // namespace corsika::kokkos
