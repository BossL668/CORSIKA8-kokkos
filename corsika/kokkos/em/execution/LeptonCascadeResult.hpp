/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/ProcessStatistics.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/LeptonFinalStateRecord.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <corsika/kokkos/output/ObservationRecord.hpp>
#include <corsika/kokkos/output/ProjectedStepRecord.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace corsika::kokkos {
  /** Host outputs and lossless checkpoints from charged-lepton fronts.
   * CPU fallbacks, decays, spill particles and remaining leptons require a caller.
   * Records owned by a device profile accumulator must not be accumulated again.
   */
  struct LeptonCascadeResult {
    std::size_t inputParticles_{};
    std::size_t wavefronts_{};
    std::size_t peakResidentLeptons_{};
    /** Successful selection records, including later-stage fallbacks. */
    std::size_t selectedInteractions_{};
    /** Wavefronts containing at least one transported interaction candidate. */
    std::size_t interactionBearingWavefronts_{};
    /** Candidates accepted as physical interactions by vertex selection. */
    std::size_t vertexInteractionsSelected_{};
    /** Candidates returned as no-interaction charged continuations. */
    std::size_t vertexNoInteractionContinuations_{};
    /** Wavefronts containing at least one selected final-state input. */
    std::size_t finalStateBearingWavefronts_{};
    std::size_t transportRecords_{};
    std::size_t interactionVertices_{};
    std::size_t lpmSuppressions_{};
    std::uint64_t wavefrontBucketingBatches_{};
    std::uint64_t wavefrontBucketingParticles_{};
    std::uint64_t wavefrontBucketingSmallBatches_{};
    std::uint64_t wavefrontBucketingSmallParticles_{};
    ProcessStatistics processStatistics_{};
    std::uint64_t secondaryHistoryIdsUsed_{};
    bool historyRangeExhausted_{};
    bool belowMinimumBatchCheckpoint_{};
    bool workspaceLimitCheckpoint_{};
    bool completed_{};
    std::vector<LeptonTransportRecord> stepRecords_{};
    std::vector<ProjectedStepRecord> projectedStepRecords_{};
    std::vector<ParticleState> generatedPhotons_{};
    std::vector<ParticleState> cpuSpillParticles_{};
    std::vector<FallbackRequest> fallbackEvents_{};
    std::vector<ObservationRecord> observations_{};
    std::vector<ParticleState> decayCandidates_{};
    /** Final process/component/v decisions after continuous-loss reselection. */
    std::vector<InteractionRecord> interactionRecords_{};
    std::vector<LeptonFinalStateRecord> finalStateRecords_{};
    std::vector<ParticleState> remainingLeptons_{};
  };
} // namespace corsika::kokkos
