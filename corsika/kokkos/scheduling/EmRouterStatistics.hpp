/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <map>

namespace corsika::kokkos {
  struct EmRouterStatistics {
    std::uint64_t routeAttempts_{};
    std::uint64_t routeNonEmRejections_{};
    std::uint64_t routeBackendRejections_{};
    std::uint64_t routeEnvironmentRejections_{};
    std::uint64_t particlesStaged_{};
    std::uint64_t photonsAdvanced_{};
    std::uint64_t leptonsAdvanced_{};
    std::uint64_t residentPhotonWavefronts_{};
    std::uint64_t residentLeptonWavefronts_{};
    std::uint64_t historyRangeCheckpoints_{};
    std::uint64_t belowMinimumBatchCheckpoints_{};
    std::uint64_t workspaceLimitCheckpoints_{};
    std::uint64_t inputBatchSplits_{};
    std::uint64_t electromagneticParticlesProduced_{};
    std::uint64_t particlesReturnedForCpuFallback_{};
    std::uint64_t particlesReturnedForCpuMemorySpill_{};
    std::uint64_t particlesReturnedForCpuDecay_{};
    std::uint64_t specifiedCpuFinalStates_{};
    std::uint64_t deferredCpuFallbacksQueued_{};
    std::uint64_t deferredCpuFallbacksFlushed_{};
    std::uint64_t deferredCpuFallbackFlushes_{};
    std::uint64_t deferredFallbackScalarExpansionRounds_{};
    std::uint64_t deferredFrontGpuFlushes_{};
    std::uint64_t deferredProductGpuFlushes_{};
    std::uint64_t cpuFallbackStepsExecuted_{};
    std::uint64_t cpuMemorySpillStepsExecuted_{};
    std::uint64_t forcedCpuDecaysExecuted_{};
    std::uint64_t cpuWavefrontExpansionStepsExecuted_{};
    std::uint64_t particlesReturnedForCpuWavefrontExpansion_{};
    std::uint64_t smallBatchExpansions_{};
    std::uint64_t stalledBoundaryGpuFlushes_{};
    std::uint64_t scalarExpansionBudgetGpuFlushes_{};
    std::uint64_t particlesObserved_{};
    std::uint64_t particlesEscaped_{};
    std::uint64_t particlesCut_{};
    std::uint64_t wavefronts_{};
    std::uint64_t reservedHistoryIds_{};
    std::size_t maximumInputBatch_{};
    std::size_t maximumDeferredCpuFallbackBatch_{};
    double depositedEnergyGeV_{};
    double mediumRestMassInputGeV_{};
    double cutRestMassEnergyGeV_{};
    double observedTotalEnergyGeV_{};
    double escapedTotalEnergyGeV_{};
    double unwrittenPhotoelectricBindingEnergyGeV_{};
    double observationCutOverlapEnergyGeV_{};
    double massConventionCorrectionGeV_{};
    double specifiedCpuFallbackTimeMs_{};
    double photonBackendWallTimeMs_{};
    double leptonBackendWallTimeMs_{};
    double photonHostPostprocessTimeMs_{};
    double leptonHostPostprocessTimeMs_{};
    std::map<std::int32_t, std::uint64_t> cpuFallbacksByProcess_{};
    std::map<std::int32_t, std::uint64_t> cpuFallbacksByReason_{};
  };
} // namespace corsika::kokkos
