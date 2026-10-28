/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstdint>

namespace corsika::kokkos {
  /** Production process counters, including unchanged compatibility fields.
   * A counter does not enable a process or a sampling algorithm. */
  struct ProcessStatistics {
    std::uint64_t gpuFinalStates_{};
    std::uint64_t physicalSecondariesGenerated_{};
    std::uint64_t photonPairFinalStates_{};
    std::uint64_t photonMuonPairFinalStates_{};
    std::uint64_t bremsFinalStates_{};
    std::uint64_t comptonFinalStates_{};
    std::uint64_t photoelectricFinalStates_{};
    std::uint64_t annihilationFinalStates_{};
    std::uint64_t ionizationFinalStates_{};
    std::uint64_t electronPairFinalStates_{};
    std::uint64_t photonPairLpmTrials_{};
    std::uint64_t photonPairLpmSuppressions_{};
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
    std::uint64_t moliereTrials_{};
    std::uint64_t moliereDeflections_{};
    std::uint64_t moliereZeroDeflections_{};
    std::uint64_t moliereNewtonIterations_{};
    std::uint32_t moliereMaxNewtonIterations_{};
    std::uint32_t reserved_{};
    std::array<std::uint64_t, 8> leptonTransportLimits_{};
    // Non-absorbing interfaces are separate from the original eight limits.
    std::uint64_t leptonMaterialBoundaries_{};
  };
} // namespace corsika::kokkos
