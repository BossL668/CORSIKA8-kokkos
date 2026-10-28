/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <corsika/kokkos/em/execution/BackendConfig.hpp>
#include <corsika/kokkos/em/execution/BackendTimingStatistics.hpp>
#include <corsika/kokkos/em/execution/CooperativeStatistics.hpp>
#include <corsika/kokkos/em/execution/ProfileStatistics.hpp>
#include <corsika/kokkos/radio/RadioStatistics.hpp>

namespace corsika::kokkos {
  struct BackendStatistics {
    CooperativeStatistics cooperative_{};
    /**
     * Lifecycle metadata. shower_ordinal starts at one for initialize() and
     * increases on every beginShower() call. One-time initialization metrics
     * are reported separately from per-shower transfer and kernel counters.
     */
    std::uint64_t showerOrdinal_{};
    bool reusedForShower_{};
    std::uint64_t staticHostToDeviceBytes_{};
    double oneTimeInitializationMs_{};
    std::string acceleratorBackend_;
    std::string acceleratorDeviceName_;
    std::string acceleratorArchitecture_;
    std::string acceleratorDriverVersion_;
    std::string acceleratorRuntimeVersion_;
    std::string acceleratorCompilerVersion_;
    std::string acceleratorProjectRevision_;
    int acceleratorDevice_{};
    int acceleratorConcurrency_{};
    int acceleratorHostThreads_{1};
    bool acceleratorOpenmp_{};
    bool acceleratorGpu_{};
    bool tuningCacheMatched_{};
    bool tuningCacheRequired_{};
    std::string tuningCacheHash_;
    std::size_t tuningBatchSize_{};
    std::size_t tuningChunkSize_{};
    std::size_t tuningTeamSize_{};
    std::size_t tuningTrackTileSize_{};
    std::size_t tuningObserverTileSize_{};
    std::size_t tuningDeviceQueues_{};
    PhysicsSource physicsSource_{PhysicsSource::ProposalNative};
    std::string nativeProposalVersion_;
    std::string nativeCubicInterpolationVersion_;
    std::array<std::uint8_t, 32> nativeTableHash_{};
    std::array<std::uint8_t, 32> auxiliaryCacheHash_{};
    std::uint64_t nativeTableNodes_{};
    std::size_t nativeTableDeviceBytes_{};
    std::uint64_t nativeNewtonIterations_{};
    std::uint64_t nativeBisectionIterations_{};
    std::uint64_t nativeInverseFailures_{};
    std::uint64_t proposalCacheTableCount_{};
    std::uint64_t proposalCacheHitCount_{};
    bool proposalCacheAllHit_{};
    bool auxiliaryCacheHit_{};
    std::uint64_t wavefronts_{};
    std::uint64_t particlesEnqueued_{};
    std::uint64_t particlesAdvanced_{};
    std::uint64_t particlesProduced_{};
    std::uint64_t proposalFallbacks_{};
    std::uint64_t interactionSelectionBatches_{};
    std::uint64_t interactionsSelected_{};
    std::uint64_t finalStateBatches_{};
    std::uint64_t gpuFinalStates_{};
    std::uint64_t firstInteractionCandidates_{};
    std::uint64_t physicalSecondariesGenerated_{};
    std::uint64_t photonPairLpmTrials_{};
    std::uint64_t photonPairLpmSuppressions_{};
    std::uint64_t photonPairFinalStates_{};
    std::uint64_t photonMuonPairFinalStates_{};
    std::uint64_t bremsFinalStates_{};
    std::uint64_t comptonFinalStates_{};
    std::uint64_t photoelectricFinalStates_{};
    std::uint64_t annihilationFinalStates_{};
    std::uint64_t ionizationFinalStates_{};
    std::uint64_t electronPairFinalStates_{};
    std::uint64_t bremsLpmTrials_{};
    std::uint64_t bremsLpmSuppressions_{};
    std::uint64_t electronPairLpmTrials_{};
    std::uint64_t electronPairLpmSuppressions_{};
    std::uint64_t electronPairRejectionTrials_{};
    std::uint64_t electronPairZeroWeightSamples_{};
    std::uint64_t electronPairRejectionFallbacks_{};
    std::uint64_t electronPairEnvelopeViolations_{};
    std::uint64_t thinningHillasVertices_{};
    std::uint64_t thinningStatisticalVertices_{};
    std::uint64_t thinningParticlesDiscarded_{};
    std::uint64_t photonTransportBatches_{};
    std::uint64_t photonTransportInteractions_{};
    std::uint64_t photonTransportBoundaries_{};
    std::uint64_t photonTransportObservations_{};
    std::uint64_t photonTransportEscapes_{};
    std::uint64_t photonTransportCuts_{};
    std::uint64_t leptonTransportBatches_{};
    std::uint64_t leptonTransportInteractionCandidates_{};
    std::uint64_t leptonTransportContinuousSteps_{};
    std::uint64_t leptonTransportCuts_{};
    std::uint64_t leptonTransportBoundaries_{};
    std::uint64_t leptonTransportObservations_{};
    std::uint64_t leptonTransportEscapes_{};
    std::uint64_t leptonTransportMagneticSteps_{};
    std::uint64_t leptonTransportDecayCandidates_{};
    std::uint64_t leptonVertexSelectionBatches_{};
    std::uint64_t leptonVertexInteractionsSelected_{};
    std::uint64_t leptonVertexNoInteractionContinuations_{};
    std::uint64_t moliereTrials_{};
    std::uint64_t moliereDeflections_{};
    std::uint64_t moliereZeroDeflections_{};
    std::uint64_t moliereNewtonIterations_{};
    std::uint32_t moliereMaxNewtonIterations_{};
    std::uint32_t moliereStatisticsReserved_{};
    std::uint64_t physicalPhotonWavefronts_{};
    std::uint64_t physicalLeptonWavefronts_{};
    std::uint64_t photonSelectionTransportSummaryFusions_{};
    std::uint64_t leptonSelectionTransportSummaryFusions_{};
    std::uint64_t photonTransportFinalStateSummaryFusions_{};
    std::uint64_t leptonTransportVertexSummaryFusions_{};
    std::uint64_t leptonVertexFinalStateSummaryFusions_{};
    std::uint64_t photonFinalStateEndpointSummaryFusions_{};
    std::uint64_t leptonFinalStateEndpointSummaryFusions_{};
    std::uint64_t pipelineHostSynchronizationsEliminated_{};
    std::uint64_t pipelineDeviceToHostBytesEliminated_{};
    std::uint64_t queueOverflows_{};
    std::size_t currentParticles_{};
    std::size_t peakParticles_{};
    std::size_t reservedParticles_{};
    std::size_t peakDeviceBytes_{};
    std::size_t tableDeviceBytes_{};
    std::size_t physicalWorkspaceBytes_{};
    std::size_t maximumResidentPhotonBatch_{};
    std::size_t maximumResidentLeptonBatch_{};
    // Nonzero only for the Kokkos GPU automatic resident-capacity policy.
    std::size_t automaticCapacityBudgetBytes_{};
    std::size_t automaticCapacityPlannedBytes_{};
    std::size_t automaticCapacityPeakBytes_{};
    std::size_t automaticCapacityAllocatorReserveBytes_{};
    std::size_t crossSpeciesQueueDeviceBytes_{};
    std::size_t crossSpeciesQueueCapacityPerPid_{};
    std::size_t peakPendingPhotons_{};
    std::size_t peakPendingLeptons_{};
    std::uint64_t crossSpeciesParticlesKeptOnDevice_{};
    std::uint64_t crossSpeciesDeviceToDeviceBytes_{};
    std::uint64_t crossSpeciesHostSpills_{};
    std::uint64_t crossSpeciesSpillRebalances_{};
    std::uint64_t crossSpeciesParticlesSpilledToCpu_{};
    std::uint64_t crossSpeciesLowEnergyOrderingChecks_{};
    std::uint64_t wavefrontBucketingBatches_{};
    std::uint64_t wavefrontBucketingParticles_{};
    std::uint64_t wavefrontBucketingSmallBatches_{};
    std::uint64_t wavefrontBucketingSmallParticles_{};
    std::uint64_t physicalHostToDeviceBytes_{};
    std::uint64_t physicalDeviceToHostBytes_{};
    double kernelTimeMs_{};
    /**
     * Legacy host-wall measurement around synchronous CUDA copies. This field
     * is kept for output compatibility; use `transfer_timing` to distinguish
     * actual device copy time from synchronization wait.
     */
    double transferTimeMs_{};
    TransferTimingStatistics transferTiming_{};
    SynchronizationTimingStatistics synchronizationTiming_{};
    LeptonPipelineTimingStatistics leptonPipelineTiming_{};
    ProfileStatistics profile_{};
    radio::RadioStatistics radio_{};
  };
} // namespace corsika::kokkos
