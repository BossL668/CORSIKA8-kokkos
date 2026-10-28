/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/ProcessStatistics.hpp>
#include <corsika/kokkos/em/PhotonFinalStateRecord.hpp>
#include <corsika/kokkos/em/PhotonTransportRecord.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <corsika/kokkos/output/ObservationRecord.hpp>
#include <corsika/kokkos/output/ProjectedStepRecord.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace corsika::kokkos {
  /** Host results and lossless checkpoints from the production photon loop.
   * Charged children are returned or appended to the supplied pending queue.
   * Profile-owned records may stay on the execution space to avoid double counting.
   * CPU spill and remaining photons are outputs the caller must continue. */
  struct PhotonCascadeResult {
    std::size_t inputParticles_{};
    std::size_t wavefronts_{};
    std::size_t peakResidentPhotons_{};
    /** Successful selection records, including later transport fallbacks. */
    std::size_t selectedInteractions_{};
    /** Wavefronts containing at least one transported interaction vertex. */
    std::size_t interactionBearingWavefronts_{};
    std::size_t transportRecords_{};
    std::size_t interactionVertices_{};
    std::size_t layerBoundaries_{};
    std::size_t particleCuts_{};
    std::size_t lpmSuppressions_{};
    std::uint64_t wavefrontBucketingBatches_{};
    std::uint64_t wavefrontBucketingParticles_{};
    std::uint64_t wavefrontBucketingSmallBatches_{};
    std::uint64_t wavefrontBucketingSmallParticles_{};
    std::uint64_t nativeNewtonIterations_{};
    std::uint64_t nativeBisectionIterations_{};
    std::uint64_t nativeInverseFailures_{};
    ProcessStatistics processStatistics_{};
    bool belowMinimumBatchCheckpoint_{};
    bool workspaceLimitCheckpoint_{};
    bool completed_{};
    std::vector<ParticleState> electromagneticSecondaries_{};
    std::vector<ParticleState> cpuSpillParticles_{};
    std::vector<FallbackRequest> fallbackEvents_{};
    std::vector<ObservationRecord> observations_{};
    /** Final process/component/v decisions at transported vertices. */
    std::vector<InteractionRecord> interactionRecords_{};
    std::vector<PhotonTransportRecord> stepRecords_{};
    std::vector<ProjectedStepRecord> projectedStepRecords_{};
    std::vector<PhotonFinalStateRecord> finalStateRecords_{};
    std::vector<ParticleState> remainingPhotons_{};
  };
} // namespace corsika::kokkos
