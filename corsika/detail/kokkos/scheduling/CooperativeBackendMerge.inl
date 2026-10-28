/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class T>
  T cooperative_add(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
      if (b > std::numeric_limits<T>::max() - a)
        throw std::overflow_error("cooperative counter overflow");
    }
    T r = a + b;
    if constexpr (std::is_floating_point_v<T>)
      if (!std::isfinite(r))
        throw std::overflow_error("nonfinite cooperative diagnostic");
    return r;
  }

  inline void merge_cooperative(ProcessStatistics& a, ProcessStatistics b) {
    a.gpuFinalStates_ = cooperative_add(a.gpuFinalStates_, b.gpuFinalStates_);
    a.physicalSecondariesGenerated_ =
        cooperative_add(a.physicalSecondariesGenerated_, b.physicalSecondariesGenerated_);
    a.photonPairFinalStates_ =
        cooperative_add(a.photonPairFinalStates_, b.photonPairFinalStates_);
    a.photonMuonPairFinalStates_ =
        cooperative_add(a.photonMuonPairFinalStates_, b.photonMuonPairFinalStates_);
    a.bremsFinalStates_ = cooperative_add(a.bremsFinalStates_, b.bremsFinalStates_);
    a.comptonFinalStates_ = cooperative_add(a.comptonFinalStates_, b.comptonFinalStates_);
    a.photoelectricFinalStates_ =
        cooperative_add(a.photoelectricFinalStates_, b.photoelectricFinalStates_);
    a.annihilationFinalStates_ =
        cooperative_add(a.annihilationFinalStates_, b.annihilationFinalStates_);
    a.ionizationFinalStates_ =
        cooperative_add(a.ionizationFinalStates_, b.ionizationFinalStates_);
    a.electronPairFinalStates_ =
        cooperative_add(a.electronPairFinalStates_, b.electronPairFinalStates_);
    a.photonPairLpmTrials_ =
        cooperative_add(a.photonPairLpmTrials_, b.photonPairLpmTrials_);
    a.photonPairLpmSuppressions_ =
        cooperative_add(a.photonPairLpmSuppressions_, b.photonPairLpmSuppressions_);
    a.bremsLpmTrials_ = cooperative_add(a.bremsLpmTrials_, b.bremsLpmTrials_);
    a.bremsLpmSuppressions_ =
        cooperative_add(a.bremsLpmSuppressions_, b.bremsLpmSuppressions_);
    a.electronPairLpmTrials_ =
        cooperative_add(a.electronPairLpmTrials_, b.electronPairLpmTrials_);
    a.electronPairLpmSuppressions_ =
        cooperative_add(a.electronPairLpmSuppressions_, b.electronPairLpmSuppressions_);
    a.electronPairRejectionTrials_ =
        cooperative_add(a.electronPairRejectionTrials_, b.electronPairRejectionTrials_);
    a.electronPairZeroWeightSamples_ = cooperative_add(a.electronPairZeroWeightSamples_,
                                                       b.electronPairZeroWeightSamples_);
    a.electronPairRejectionFallbacks_ = cooperative_add(
        a.electronPairRejectionFallbacks_, b.electronPairRejectionFallbacks_);
    a.electronPairEnvelopeViolations_ = cooperative_add(
        a.electronPairEnvelopeViolations_, b.electronPairEnvelopeViolations_);
    a.thinningHillasVertices_ =
        cooperative_add(a.thinningHillasVertices_, b.thinningHillasVertices_);
    a.thinningStatisticalVertices_ =
        cooperative_add(a.thinningStatisticalVertices_, b.thinningStatisticalVertices_);
    a.thinningParticlesDiscarded_ =
        cooperative_add(a.thinningParticlesDiscarded_, b.thinningParticlesDiscarded_);
    a.moliereTrials_ = cooperative_add(a.moliereTrials_, b.moliereTrials_);
    a.moliereDeflections_ = cooperative_add(a.moliereDeflections_, b.moliereDeflections_);
    a.moliereZeroDeflections_ =
        cooperative_add(a.moliereZeroDeflections_, b.moliereZeroDeflections_);
    a.moliereNewtonIterations_ =
        cooperative_add(a.moliereNewtonIterations_, b.moliereNewtonIterations_);
    a.moliereMaxNewtonIterations_ =
        std::max(a.moliereMaxNewtonIterations_, b.moliereMaxNewtonIterations_);
    for (std::size_t i = 0; i < a.leptonTransportLimits_.size(); ++i)
      a.leptonTransportLimits_[i] =
          cooperative_add(a.leptonTransportLimits_[i], b.leptonTransportLimits_[i]);
  }

  inline void merge_cooperative(PhotonCascadeResult& a, PhotonCascadeResult b) {
    a.inputParticles_ = cooperative_add(a.inputParticles_, b.inputParticles_);
    a.wavefronts_ = cooperative_add(a.wavefronts_, b.wavefronts_);
    a.peakResidentPhotons_ = std::max(a.peakResidentPhotons_, b.peakResidentPhotons_);
    a.selectedInteractions_ =
        cooperative_add(a.selectedInteractions_, b.selectedInteractions_);
    a.interactionBearingWavefronts_ =
        cooperative_add(a.interactionBearingWavefronts_, b.interactionBearingWavefronts_);
    a.transportRecords_ = cooperative_add(a.transportRecords_, b.transportRecords_);
    a.interactionVertices_ =
        cooperative_add(a.interactionVertices_, b.interactionVertices_);
    a.layerBoundaries_ = cooperative_add(a.layerBoundaries_, b.layerBoundaries_);
    a.particleCuts_ = cooperative_add(a.particleCuts_, b.particleCuts_);
    a.lpmSuppressions_ = cooperative_add(a.lpmSuppressions_, b.lpmSuppressions_);
    a.wavefrontBucketingBatches_ =
        cooperative_add(a.wavefrontBucketingBatches_, b.wavefrontBucketingBatches_);
    a.wavefrontBucketingParticles_ =
        cooperative_add(a.wavefrontBucketingParticles_, b.wavefrontBucketingParticles_);
    a.wavefrontBucketingSmallBatches_ = cooperative_add(
        a.wavefrontBucketingSmallBatches_, b.wavefrontBucketingSmallBatches_);
    a.wavefrontBucketingSmallParticles_ = cooperative_add(
        a.wavefrontBucketingSmallParticles_, b.wavefrontBucketingSmallParticles_);
    a.nativeNewtonIterations_ =
        cooperative_add(a.nativeNewtonIterations_, b.nativeNewtonIterations_);
    a.nativeBisectionIterations_ =
        cooperative_add(a.nativeBisectionIterations_, b.nativeBisectionIterations_);
    a.nativeInverseFailures_ =
        cooperative_add(a.nativeInverseFailures_, b.nativeInverseFailures_);
    merge_cooperative(a.processStatistics_, b.processStatistics_);
    a.belowMinimumBatchCheckpoint_ =
        a.belowMinimumBatchCheckpoint_ || b.belowMinimumBatchCheckpoint_;
    a.workspaceLimitCheckpoint_ =
        a.workspaceLimitCheckpoint_ || b.workspaceLimitCheckpoint_;
    a.completed_ = a.completed_ && b.completed_;
    a.electromagneticSecondaries_.insert(
        a.electromagneticSecondaries_.end(),
        std::make_move_iterator(b.electromagneticSecondaries_.begin()),
        std::make_move_iterator(b.electromagneticSecondaries_.end()));
    a.cpuSpillParticles_.insert(a.cpuSpillParticles_.end(),
                                std::make_move_iterator(b.cpuSpillParticles_.begin()),
                                std::make_move_iterator(b.cpuSpillParticles_.end()));
    a.fallbackEvents_.insert(a.fallbackEvents_.end(),
                             std::make_move_iterator(b.fallbackEvents_.begin()),
                             std::make_move_iterator(b.fallbackEvents_.end()));
    a.observations_.insert(a.observations_.end(),
                           std::make_move_iterator(b.observations_.begin()),
                           std::make_move_iterator(b.observations_.end()));
    a.interactionRecords_.insert(a.interactionRecords_.end(),
                                 std::make_move_iterator(b.interactionRecords_.begin()),
                                 std::make_move_iterator(b.interactionRecords_.end()));
    a.stepRecords_.insert(a.stepRecords_.end(),
                          std::make_move_iterator(b.stepRecords_.begin()),
                          std::make_move_iterator(b.stepRecords_.end()));
    a.projectedStepRecords_.insert(
        a.projectedStepRecords_.end(),
        std::make_move_iterator(b.projectedStepRecords_.begin()),
        std::make_move_iterator(b.projectedStepRecords_.end()));
    a.finalStateRecords_.insert(a.finalStateRecords_.end(),
                                std::make_move_iterator(b.finalStateRecords_.begin()),
                                std::make_move_iterator(b.finalStateRecords_.end()));
    a.remainingPhotons_.insert(a.remainingPhotons_.end(),
                               std::make_move_iterator(b.remainingPhotons_.begin()),
                               std::make_move_iterator(b.remainingPhotons_.end()));
  }

  inline void merge_cooperative(LeptonCascadeResult& a, LeptonCascadeResult b) {
    a.inputParticles_ = cooperative_add(a.inputParticles_, b.inputParticles_);
    a.wavefronts_ = cooperative_add(a.wavefronts_, b.wavefronts_);
    a.peakResidentLeptons_ = std::max(a.peakResidentLeptons_, b.peakResidentLeptons_);
    a.selectedInteractions_ =
        cooperative_add(a.selectedInteractions_, b.selectedInteractions_);
    a.interactionBearingWavefronts_ =
        cooperative_add(a.interactionBearingWavefronts_, b.interactionBearingWavefronts_);
    a.vertexInteractionsSelected_ =
        cooperative_add(a.vertexInteractionsSelected_, b.vertexInteractionsSelected_);
    a.vertexNoInteractionContinuations_ = cooperative_add(
        a.vertexNoInteractionContinuations_, b.vertexNoInteractionContinuations_);
    a.finalStateBearingWavefronts_ =
        cooperative_add(a.finalStateBearingWavefronts_, b.finalStateBearingWavefronts_);
    a.transportRecords_ = cooperative_add(a.transportRecords_, b.transportRecords_);
    a.interactionVertices_ =
        cooperative_add(a.interactionVertices_, b.interactionVertices_);
    a.lpmSuppressions_ = cooperative_add(a.lpmSuppressions_, b.lpmSuppressions_);
    a.wavefrontBucketingBatches_ =
        cooperative_add(a.wavefrontBucketingBatches_, b.wavefrontBucketingBatches_);
    a.wavefrontBucketingParticles_ =
        cooperative_add(a.wavefrontBucketingParticles_, b.wavefrontBucketingParticles_);
    a.wavefrontBucketingSmallBatches_ = cooperative_add(
        a.wavefrontBucketingSmallBatches_, b.wavefrontBucketingSmallBatches_);
    a.wavefrontBucketingSmallParticles_ = cooperative_add(
        a.wavefrontBucketingSmallParticles_, b.wavefrontBucketingSmallParticles_);
    merge_cooperative(a.processStatistics_, b.processStatistics_);
    a.secondaryHistoryIdsUsed_ =
        cooperative_add(a.secondaryHistoryIdsUsed_, b.secondaryHistoryIdsUsed_);
    a.historyRangeExhausted_ = a.historyRangeExhausted_ || b.historyRangeExhausted_;
    a.belowMinimumBatchCheckpoint_ =
        a.belowMinimumBatchCheckpoint_ || b.belowMinimumBatchCheckpoint_;
    a.workspaceLimitCheckpoint_ =
        a.workspaceLimitCheckpoint_ || b.workspaceLimitCheckpoint_;
    a.completed_ = a.completed_ && b.completed_;
    a.stepRecords_.insert(a.stepRecords_.end(),
                          std::make_move_iterator(b.stepRecords_.begin()),
                          std::make_move_iterator(b.stepRecords_.end()));
    a.projectedStepRecords_.insert(
        a.projectedStepRecords_.end(),
        std::make_move_iterator(b.projectedStepRecords_.begin()),
        std::make_move_iterator(b.projectedStepRecords_.end()));
    a.generatedPhotons_.insert(a.generatedPhotons_.end(),
                               std::make_move_iterator(b.generatedPhotons_.begin()),
                               std::make_move_iterator(b.generatedPhotons_.end()));
    a.cpuSpillParticles_.insert(a.cpuSpillParticles_.end(),
                                std::make_move_iterator(b.cpuSpillParticles_.begin()),
                                std::make_move_iterator(b.cpuSpillParticles_.end()));
    a.fallbackEvents_.insert(a.fallbackEvents_.end(),
                             std::make_move_iterator(b.fallbackEvents_.begin()),
                             std::make_move_iterator(b.fallbackEvents_.end()));
    a.observations_.insert(a.observations_.end(),
                           std::make_move_iterator(b.observations_.begin()),
                           std::make_move_iterator(b.observations_.end()));
    a.decayCandidates_.insert(a.decayCandidates_.end(),
                              std::make_move_iterator(b.decayCandidates_.begin()),
                              std::make_move_iterator(b.decayCandidates_.end()));
    a.interactionRecords_.insert(a.interactionRecords_.end(),
                                 std::make_move_iterator(b.interactionRecords_.begin()),
                                 std::make_move_iterator(b.interactionRecords_.end()));
    a.finalStateRecords_.insert(a.finalStateRecords_.end(),
                                std::make_move_iterator(b.finalStateRecords_.begin()),
                                std::make_move_iterator(b.finalStateRecords_.end()));
    a.remainingLeptons_.insert(a.remainingLeptons_.end(),
                               std::make_move_iterator(b.remainingLeptons_.begin()),
                               std::make_move_iterator(b.remainingLeptons_.end()));
  }

  inline BackendStatistics merge_cooperative_statistics(BackendStatistics a,
                                                        BackendStatistics const& b) {
    if (a.nativeTableHash_ != b.nativeTableHash_ ||
        a.auxiliaryCacheHash_ != b.auxiliaryCacheHash_ ||
        a.showerOrdinal_ != b.showerOrdinal_)
      throw std::logic_error(
          "cooperative endpoints have incompatible physics/shower identity");
    // Identity, GPU capacities/VRAM and tuning describe the CUDA endpoint.
    // Physical counters and phase work are summed; wall time is measured outside.
    a.staticHostToDeviceBytes_ =
        cooperative_add(a.staticHostToDeviceBytes_, b.staticHostToDeviceBytes_);
    a.nativeNewtonIterations_ =
        cooperative_add(a.nativeNewtonIterations_, b.nativeNewtonIterations_);
    a.nativeBisectionIterations_ =
        cooperative_add(a.nativeBisectionIterations_, b.nativeBisectionIterations_);
    a.nativeInverseFailures_ =
        cooperative_add(a.nativeInverseFailures_, b.nativeInverseFailures_);
    a.wavefronts_ = cooperative_add(a.wavefronts_, b.wavefronts_);
    a.particlesEnqueued_ = cooperative_add(a.particlesEnqueued_, b.particlesEnqueued_);
    a.particlesAdvanced_ = cooperative_add(a.particlesAdvanced_, b.particlesAdvanced_);
    a.particlesProduced_ = cooperative_add(a.particlesProduced_, b.particlesProduced_);
    a.proposalFallbacks_ = cooperative_add(a.proposalFallbacks_, b.proposalFallbacks_);
    a.interactionSelectionBatches_ =
        cooperative_add(a.interactionSelectionBatches_, b.interactionSelectionBatches_);
    a.interactionsSelected_ =
        cooperative_add(a.interactionsSelected_, b.interactionsSelected_);
    a.finalStateBatches_ = cooperative_add(a.finalStateBatches_, b.finalStateBatches_);
    a.gpuFinalStates_ = cooperative_add(a.gpuFinalStates_, b.gpuFinalStates_);
    a.firstInteractionCandidates_ =
        cooperative_add(a.firstInteractionCandidates_, b.firstInteractionCandidates_);
    a.physicalSecondariesGenerated_ =
        cooperative_add(a.physicalSecondariesGenerated_, b.physicalSecondariesGenerated_);
    a.photonPairLpmTrials_ =
        cooperative_add(a.photonPairLpmTrials_, b.photonPairLpmTrials_);
    a.photonPairLpmSuppressions_ =
        cooperative_add(a.photonPairLpmSuppressions_, b.photonPairLpmSuppressions_);
    a.photonPairFinalStates_ =
        cooperative_add(a.photonPairFinalStates_, b.photonPairFinalStates_);
    a.photonMuonPairFinalStates_ =
        cooperative_add(a.photonMuonPairFinalStates_, b.photonMuonPairFinalStates_);
    a.bremsFinalStates_ = cooperative_add(a.bremsFinalStates_, b.bremsFinalStates_);
    a.comptonFinalStates_ = cooperative_add(a.comptonFinalStates_, b.comptonFinalStates_);
    a.photoelectricFinalStates_ =
        cooperative_add(a.photoelectricFinalStates_, b.photoelectricFinalStates_);
    a.annihilationFinalStates_ =
        cooperative_add(a.annihilationFinalStates_, b.annihilationFinalStates_);
    a.ionizationFinalStates_ =
        cooperative_add(a.ionizationFinalStates_, b.ionizationFinalStates_);
    a.electronPairFinalStates_ =
        cooperative_add(a.electronPairFinalStates_, b.electronPairFinalStates_);
    a.bremsLpmTrials_ = cooperative_add(a.bremsLpmTrials_, b.bremsLpmTrials_);
    a.bremsLpmSuppressions_ =
        cooperative_add(a.bremsLpmSuppressions_, b.bremsLpmSuppressions_);
    a.electronPairLpmTrials_ =
        cooperative_add(a.electronPairLpmTrials_, b.electronPairLpmTrials_);
    a.electronPairLpmSuppressions_ =
        cooperative_add(a.electronPairLpmSuppressions_, b.electronPairLpmSuppressions_);
    a.electronPairRejectionTrials_ =
        cooperative_add(a.electronPairRejectionTrials_, b.electronPairRejectionTrials_);
    a.electronPairZeroWeightSamples_ = cooperative_add(a.electronPairZeroWeightSamples_,
                                                       b.electronPairZeroWeightSamples_);
    a.electronPairRejectionFallbacks_ = cooperative_add(
        a.electronPairRejectionFallbacks_, b.electronPairRejectionFallbacks_);
    a.electronPairEnvelopeViolations_ = cooperative_add(
        a.electronPairEnvelopeViolations_, b.electronPairEnvelopeViolations_);
    a.thinningHillasVertices_ =
        cooperative_add(a.thinningHillasVertices_, b.thinningHillasVertices_);
    a.thinningStatisticalVertices_ =
        cooperative_add(a.thinningStatisticalVertices_, b.thinningStatisticalVertices_);
    a.thinningParticlesDiscarded_ =
        cooperative_add(a.thinningParticlesDiscarded_, b.thinningParticlesDiscarded_);
    a.photonTransportBatches_ =
        cooperative_add(a.photonTransportBatches_, b.photonTransportBatches_);
    a.photonTransportInteractions_ =
        cooperative_add(a.photonTransportInteractions_, b.photonTransportInteractions_);
    a.photonTransportBoundaries_ =
        cooperative_add(a.photonTransportBoundaries_, b.photonTransportBoundaries_);
    a.photonTransportObservations_ =
        cooperative_add(a.photonTransportObservations_, b.photonTransportObservations_);
    a.photonTransportEscapes_ =
        cooperative_add(a.photonTransportEscapes_, b.photonTransportEscapes_);
    a.photonTransportCuts_ =
        cooperative_add(a.photonTransportCuts_, b.photonTransportCuts_);
    a.leptonTransportBatches_ =
        cooperative_add(a.leptonTransportBatches_, b.leptonTransportBatches_);
    a.leptonTransportInteractionCandidates_ = cooperative_add(
        a.leptonTransportInteractionCandidates_, b.leptonTransportInteractionCandidates_);
    a.leptonTransportContinuousSteps_ = cooperative_add(
        a.leptonTransportContinuousSteps_, b.leptonTransportContinuousSteps_);
    a.leptonTransportCuts_ =
        cooperative_add(a.leptonTransportCuts_, b.leptonTransportCuts_);
    a.leptonTransportBoundaries_ =
        cooperative_add(a.leptonTransportBoundaries_, b.leptonTransportBoundaries_);
    a.leptonTransportObservations_ =
        cooperative_add(a.leptonTransportObservations_, b.leptonTransportObservations_);
    a.leptonTransportEscapes_ =
        cooperative_add(a.leptonTransportEscapes_, b.leptonTransportEscapes_);
    a.leptonTransportMagneticSteps_ =
        cooperative_add(a.leptonTransportMagneticSteps_, b.leptonTransportMagneticSteps_);
    a.leptonTransportDecayCandidates_ = cooperative_add(
        a.leptonTransportDecayCandidates_, b.leptonTransportDecayCandidates_);
    a.leptonVertexSelectionBatches_ =
        cooperative_add(a.leptonVertexSelectionBatches_, b.leptonVertexSelectionBatches_);
    a.leptonVertexInteractionsSelected_ = cooperative_add(
        a.leptonVertexInteractionsSelected_, b.leptonVertexInteractionsSelected_);
    a.leptonVertexNoInteractionContinuations_ =
        cooperative_add(a.leptonVertexNoInteractionContinuations_,
                        b.leptonVertexNoInteractionContinuations_);
    a.moliereTrials_ = cooperative_add(a.moliereTrials_, b.moliereTrials_);
    a.moliereDeflections_ = cooperative_add(a.moliereDeflections_, b.moliereDeflections_);
    a.moliereZeroDeflections_ =
        cooperative_add(a.moliereZeroDeflections_, b.moliereZeroDeflections_);
    a.moliereNewtonIterations_ =
        cooperative_add(a.moliereNewtonIterations_, b.moliereNewtonIterations_);
    a.moliereMaxNewtonIterations_ =
        std::max(a.moliereMaxNewtonIterations_, b.moliereMaxNewtonIterations_);
    a.moliereStatisticsReserved_ =
        cooperative_add(a.moliereStatisticsReserved_, b.moliereStatisticsReserved_);
    a.physicalPhotonWavefronts_ =
        cooperative_add(a.physicalPhotonWavefronts_, b.physicalPhotonWavefronts_);
    a.physicalLeptonWavefronts_ =
        cooperative_add(a.physicalLeptonWavefronts_, b.physicalLeptonWavefronts_);
    a.photonSelectionTransportSummaryFusions_ =
        cooperative_add(a.photonSelectionTransportSummaryFusions_,
                        b.photonSelectionTransportSummaryFusions_);
    a.leptonSelectionTransportSummaryFusions_ =
        cooperative_add(a.leptonSelectionTransportSummaryFusions_,
                        b.leptonSelectionTransportSummaryFusions_);
    a.photonTransportFinalStateSummaryFusions_ =
        cooperative_add(a.photonTransportFinalStateSummaryFusions_,
                        b.photonTransportFinalStateSummaryFusions_);
    a.leptonTransportVertexSummaryFusions_ = cooperative_add(
        a.leptonTransportVertexSummaryFusions_, b.leptonTransportVertexSummaryFusions_);
    a.leptonVertexFinalStateSummaryFusions_ = cooperative_add(
        a.leptonVertexFinalStateSummaryFusions_, b.leptonVertexFinalStateSummaryFusions_);
    a.photonFinalStateEndpointSummaryFusions_ =
        cooperative_add(a.photonFinalStateEndpointSummaryFusions_,
                        b.photonFinalStateEndpointSummaryFusions_);
    a.leptonFinalStateEndpointSummaryFusions_ =
        cooperative_add(a.leptonFinalStateEndpointSummaryFusions_,
                        b.leptonFinalStateEndpointSummaryFusions_);
    a.pipelineHostSynchronizationsEliminated_ =
        cooperative_add(a.pipelineHostSynchronizationsEliminated_,
                        b.pipelineHostSynchronizationsEliminated_);
    a.pipelineDeviceToHostBytesEliminated_ = cooperative_add(
        a.pipelineDeviceToHostBytesEliminated_, b.pipelineDeviceToHostBytesEliminated_);
    a.queueOverflows_ = cooperative_add(a.queueOverflows_, b.queueOverflows_);
    a.crossSpeciesParticlesKeptOnDevice_ = cooperative_add(
        a.crossSpeciesParticlesKeptOnDevice_, b.crossSpeciesParticlesKeptOnDevice_);
    a.crossSpeciesDeviceToDeviceBytes_ = cooperative_add(
        a.crossSpeciesDeviceToDeviceBytes_, b.crossSpeciesDeviceToDeviceBytes_);
    a.crossSpeciesHostSpills_ =
        cooperative_add(a.crossSpeciesHostSpills_, b.crossSpeciesHostSpills_);
    a.crossSpeciesSpillRebalances_ =
        cooperative_add(a.crossSpeciesSpillRebalances_, b.crossSpeciesSpillRebalances_);
    a.crossSpeciesParticlesSpilledToCpu_ = cooperative_add(
        a.crossSpeciesParticlesSpilledToCpu_, b.crossSpeciesParticlesSpilledToCpu_);
    a.crossSpeciesLowEnergyOrderingChecks_ = cooperative_add(
        a.crossSpeciesLowEnergyOrderingChecks_, b.crossSpeciesLowEnergyOrderingChecks_);
    a.wavefrontBucketingBatches_ =
        cooperative_add(a.wavefrontBucketingBatches_, b.wavefrontBucketingBatches_);
    a.wavefrontBucketingParticles_ =
        cooperative_add(a.wavefrontBucketingParticles_, b.wavefrontBucketingParticles_);
    a.wavefrontBucketingSmallBatches_ = cooperative_add(
        a.wavefrontBucketingSmallBatches_, b.wavefrontBucketingSmallBatches_);
    a.wavefrontBucketingSmallParticles_ = cooperative_add(
        a.wavefrontBucketingSmallParticles_, b.wavefrontBucketingSmallParticles_);
    a.physicalHostToDeviceBytes_ =
        cooperative_add(a.physicalHostToDeviceBytes_, b.physicalHostToDeviceBytes_);
    a.physicalDeviceToHostBytes_ =
        cooperative_add(a.physicalDeviceToHostBytes_, b.physicalDeviceToHostBytes_);
    a.kernelTimeMs_ = cooperative_add(a.kernelTimeMs_, b.kernelTimeMs_);
    a.transferTimeMs_ = cooperative_add(a.transferTimeMs_, b.transferTimeMs_);
    a.profile_.steps_ = cooperative_add(a.profile_.steps_, b.profile_.steps_);
    a.profile_.depositedSteps_ =
        cooperative_add(a.profile_.depositedSteps_, b.profile_.depositedSteps_);
    a.profile_.fixedPointOverflows_ =
        cooperative_add(a.profile_.fixedPointOverflows_, b.profile_.fixedPointOverflows_);
    a.profile_.invalidRecords_ =
        cooperative_add(a.profile_.invalidRecords_, b.profile_.invalidRecords_);
    a.profile_.hostToDeviceBytes_ =
        cooperative_add(a.profile_.hostToDeviceBytes_, b.profile_.hostToDeviceBytes_);
    a.profile_.deviceToHostBytes_ =
        cooperative_add(a.profile_.deviceToHostBytes_, b.profile_.deviceToHostBytes_);
    a.profile_.kernelTimeMs_ =
        cooperative_add(a.profile_.kernelTimeMs_, b.profile_.kernelTimeMs_);
    a.profile_.transferTimeMs_ =
        cooperative_add(a.profile_.transferTimeMs_, b.profile_.transferTimeMs_);
    a.radio_.leptonTracks_ =
        cooperative_add(a.radio_.leptonTracks_, b.radio_.leptonTracks_);
    a.radio_.trackObserverPairs_ =
        cooperative_add(a.radio_.trackObserverPairs_, b.radio_.trackObserverPairs_);
    a.radio_.fusedTrackObserverPairs_ = cooperative_add(
        a.radio_.fusedTrackObserverPairs_, b.radio_.fusedTrackObserverPairs_);
    a.radio_.coreasContributions_ =
        cooperative_add(a.radio_.coreasContributions_, b.radio_.coreasContributions_);
    a.radio_.zhsContributions_ =
        cooperative_add(a.radio_.zhsContributions_, b.radio_.zhsContributions_);
    a.radio_.zhsSubtracks_ =
        cooperative_add(a.radio_.zhsSubtracks_, b.radio_.zhsSubtracks_);
    a.radio_.fixedPointOverflows_ =
        cooperative_add(a.radio_.fixedPointOverflows_, b.radio_.fixedPointOverflows_);
    a.radio_.weightedSegmentCount_ =
        cooperative_add(a.radio_.weightedSegmentCount_, b.radio_.weightedSegmentCount_);
    a.radio_.trackLengthM_ =
        cooperative_add(a.radio_.trackLengthM_, b.radio_.trackLengthM_);
    a.radio_.weightedTrackLengthM_ =
        cooperative_add(a.radio_.weightedTrackLengthM_, b.radio_.weightedTrackLengthM_);
    a.radio_.electronWeightedTrackLengthM_ = cooperative_add(
        a.radio_.electronWeightedTrackLengthM_, b.radio_.electronWeightedTrackLengthM_);
    a.radio_.positronWeightedTrackLengthM_ = cooperative_add(
        a.radio_.positronWeightedTrackLengthM_, b.radio_.positronWeightedTrackLengthM_);
    a.radio_.signedChargeWeightedTrackLengthM_ =
        cooperative_add(a.radio_.signedChargeWeightedTrackLengthM_,
                        b.radio_.signedChargeWeightedTrackLengthM_);
    a.radio_.energyWeightedTrackLengthGeVM_ = cooperative_add(
        a.radio_.energyWeightedTrackLengthGeVM_, b.radio_.energyWeightedTrackLengthGeVM_);
    a.radio_.maximumSegmentLengthM_ =
        std::max(a.radio_.maximumSegmentLengthM_, b.radio_.maximumSegmentLengthM_);
    a.radio_.weightedDirectionChangeRad_ = cooperative_add(
        a.radio_.weightedDirectionChangeRad_, b.radio_.weightedDirectionChangeRad_);
    a.radio_.weightedDirectionChangeSquaredRad2_ =
        cooperative_add(a.radio_.weightedDirectionChangeSquaredRad2_,
                        b.radio_.weightedDirectionChangeSquaredRad2_);
    a.radio_.weightedBetaDeficitTrackLengthM_ =
        cooperative_add(a.radio_.weightedBetaDeficitTrackLengthM_,
                        b.radio_.weightedBetaDeficitTrackLengthM_);
    a.radio_.weightedTimeResidualS_ =
        cooperative_add(a.radio_.weightedTimeResidualS_, b.radio_.weightedTimeResidualS_);
    a.radio_.maximumDirectionChangeRad_ = std::max(a.radio_.maximumDirectionChangeRad_,
                                                   b.radio_.maximumDirectionChangeRad_);
    for (std::size_t i = 0; i < a.radio_.signedChargeWeightedDirectionChange_.size(); ++i)
      a.radio_.signedChargeWeightedDirectionChange_[i] =
          cooperative_add(a.radio_.signedChargeWeightedDirectionChange_[i],
                          b.radio_.signedChargeWeightedDirectionChange_[i]);
    for (std::size_t i = 0; i < a.radio_.weightedTrackLengthByKineticEnergyM_.size(); ++i)
      a.radio_.weightedTrackLengthByKineticEnergyM_[i] =
          cooperative_add(a.radio_.weightedTrackLengthByKineticEnergyM_[i],
                          b.radio_.weightedTrackLengthByKineticEnergyM_[i]);
    a.radio_.hostToDeviceBytes_ =
        cooperative_add(a.radio_.hostToDeviceBytes_, b.radio_.hostToDeviceBytes_);
    a.radio_.hostTracksEnqueued_ =
        cooperative_add(a.radio_.hostTracksEnqueued_, b.radio_.hostTracksEnqueued_);
    a.radio_.hostTrackBatches_ =
        cooperative_add(a.radio_.hostTrackBatches_, b.radio_.hostTrackBatches_);
    a.radio_.deviceToHostBytes_ =
        cooperative_add(a.radio_.deviceToHostBytes_, b.radio_.deviceToHostBytes_);
    a.radio_.deviceTimeMs_ =
        cooperative_add(a.radio_.deviceTimeMs_, b.radio_.deviceTimeMs_);
    a.radio_.kernelTimeMs_ =
        cooperative_add(a.radio_.kernelTimeMs_, b.radio_.kernelTimeMs_);
    a.radio_.transferTimeMs_ =
        cooperative_add(a.radio_.transferTimeMs_, b.radio_.transferTimeMs_);
    a.radio_.inputSlotWaits_ =
        cooperative_add(a.radio_.inputSlotWaits_, b.radio_.inputSlotWaits_);
    a.radio_.inputSlotHostWaitTimeMs_ = cooperative_add(
        a.radio_.inputSlotHostWaitTimeMs_, b.radio_.inputSlotHostWaitTimeMs_);
    a.radio_.trackPrecomputeBatches_ = cooperative_add(a.radio_.trackPrecomputeBatches_,
                                                       b.radio_.trackPrecomputeBatches_);
    a.radio_.trackPrecomputedRecords_ = cooperative_add(
        a.radio_.trackPrecomputedRecords_, b.radio_.trackPrecomputedRecords_);
    a.radio_.directProjectionBatches_ = cooperative_add(
        a.radio_.directProjectionBatches_, b.radio_.directProjectionBatches_);
    a.radio_.directProjectionRecords_ = cooperative_add(
        a.radio_.directProjectionRecords_, b.radio_.directProjectionRecords_);
    a.radio_.projectionTiles_ =
        cooperative_add(a.radio_.projectionTiles_, b.radio_.projectionTiles_);
    a.radio_.maximumTrackBatch_ =
        std::max(a.radio_.maximumTrackBatch_, b.radio_.maximumTrackBatch_);
    a.radio_.trackPrecomputeDeviceTimeMs_ = cooperative_add(
        a.radio_.trackPrecomputeDeviceTimeMs_, b.radio_.trackPrecomputeDeviceTimeMs_);
    a.radio_.projectionDeviceTimeMs_ = cooperative_add(a.radio_.projectionDeviceTimeMs_,
                                                       b.radio_.projectionDeviceTimeMs_);
    a.transferTiming_.operations_ =
        cooperative_add(a.transferTiming_.operations_, b.transferTiming_.operations_);
    a.transferTiming_.hostToDeviceOperations_ =
        cooperative_add(a.transferTiming_.hostToDeviceOperations_,
                        b.transferTiming_.hostToDeviceOperations_);
    a.transferTiming_.deviceToHostOperations_ =
        cooperative_add(a.transferTiming_.deviceToHostOperations_,
                        b.transferTiming_.deviceToHostOperations_);
    a.transferTiming_.deviceToDeviceOperations_ =
        cooperative_add(a.transferTiming_.deviceToDeviceOperations_,
                        b.transferTiming_.deviceToDeviceOperations_);
    a.transferTiming_.hostApiTimeMs_ = cooperative_add(a.transferTiming_.hostApiTimeMs_,
                                                       b.transferTiming_.hostApiTimeMs_);
    a.transferTiming_.deviceCopyTimeMs_ = cooperative_add(
        a.transferTiming_.deviceCopyTimeMs_, b.transferTiming_.deviceCopyTimeMs_);
    a.transferTiming_.hostWaitUpperBoundMs_ = cooperative_add(
        a.transferTiming_.hostWaitUpperBoundMs_, b.transferTiming_.hostWaitUpperBoundMs_);
    a.synchronizationTiming_.physicalPipelineWaits_ =
        cooperative_add(a.synchronizationTiming_.physicalPipelineWaits_,
                        b.synchronizationTiming_.physicalPipelineWaits_);
    a.synchronizationTiming_.profileInputWaits_ =
        cooperative_add(a.synchronizationTiming_.profileInputWaits_,
                        b.synchronizationTiming_.profileInputWaits_);
    a.synchronizationTiming_.physicalPipelineWaitTimeMs_ =
        cooperative_add(a.synchronizationTiming_.physicalPipelineWaitTimeMs_,
                        b.synchronizationTiming_.physicalPipelineWaitTimeMs_);
    a.synchronizationTiming_.profileInputWaitTimeMs_ =
        cooperative_add(a.synchronizationTiming_.profileInputWaitTimeMs_,
                        b.synchronizationTiming_.profileInputWaitTimeMs_);
    a.leptonPipelineTiming_.wavefronts_ = cooperative_add(
        a.leptonPipelineTiming_.wavefronts_, b.leptonPipelineTiming_.wavefronts_);
    a.leptonPipelineTiming_.selectionMs_ = cooperative_add(
        a.leptonPipelineTiming_.selectionMs_, b.leptonPipelineTiming_.selectionMs_);
    a.leptonPipelineTiming_.transportMs_ = cooperative_add(
        a.leptonPipelineTiming_.transportMs_, b.leptonPipelineTiming_.transportMs_);
    a.leptonPipelineTiming_.transportPhysicsMs_ =
        cooperative_add(a.leptonPipelineTiming_.transportPhysicsMs_,
                        b.leptonPipelineTiming_.transportPhysicsMs_);
    a.leptonPipelineTiming_.moliereMs_ = cooperative_add(
        a.leptonPipelineTiming_.moliereMs_, b.leptonPipelineTiming_.moliereMs_);
    a.leptonPipelineTiming_.transportControlMs_ =
        cooperative_add(a.leptonPipelineTiming_.transportControlMs_,
                        b.leptonPipelineTiming_.transportControlMs_);
    a.leptonPipelineTiming_.transportCompactionMs_ =
        cooperative_add(a.leptonPipelineTiming_.transportCompactionMs_,
                        b.leptonPipelineTiming_.transportCompactionMs_);
    a.leptonPipelineTiming_.interactionExtractionMs_ =
        cooperative_add(a.leptonPipelineTiming_.interactionExtractionMs_,
                        b.leptonPipelineTiming_.interactionExtractionMs_);
    a.leptonPipelineTiming_.vertexSelectionMs_ =
        cooperative_add(a.leptonPipelineTiming_.vertexSelectionMs_,
                        b.leptonPipelineTiming_.vertexSelectionMs_);
    a.leptonPipelineTiming_.finalStateMs_ = cooperative_add(
        a.leptonPipelineTiming_.finalStateMs_, b.leptonPipelineTiming_.finalStateMs_);
    a.leptonPipelineTiming_.finalStateClassificationMs_ =
        cooperative_add(a.leptonPipelineTiming_.finalStateClassificationMs_,
                        b.leptonPipelineTiming_.finalStateClassificationMs_);
    a.leptonPipelineTiming_.finalStateScanMs_ =
        cooperative_add(a.leptonPipelineTiming_.finalStateScanMs_,
                        b.leptonPipelineTiming_.finalStateScanMs_);
    a.leptonPipelineTiming_.finalStateSummaryMs_ =
        cooperative_add(a.leptonPipelineTiming_.finalStateSummaryMs_,
                        b.leptonPipelineTiming_.finalStateSummaryMs_);
    a.leptonPipelineTiming_.finalStateWriteMs_ =
        cooperative_add(a.leptonPipelineTiming_.finalStateWriteMs_,
                        b.leptonPipelineTiming_.finalStateWriteMs_);
    a.leptonPipelineTiming_.endpointCompactionMs_ =
        cooperative_add(a.leptonPipelineTiming_.endpointCompactionMs_,
                        b.leptonPipelineTiming_.endpointCompactionMs_);
    a.leptonPipelineTiming_.postEndpointMs_ = cooperative_add(
        a.leptonPipelineTiming_.postEndpointMs_, b.leptonPipelineTiming_.postEndpointMs_);
    return a;
  }
} // namespace corsika::kokkos
