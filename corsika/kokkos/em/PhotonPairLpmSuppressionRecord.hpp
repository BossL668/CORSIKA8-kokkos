/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {
  /** Photon state and diagnostics for an LPM-suppressed pair candidate.
   * The producing stage increments the step ID before storing the unchanged
   * physical state, giving the continuation fresh keyed draws. This value record
   * alone does not change any particle or random-number state.
   */
  // Units match ParticleState (GeV, metres, seconds); weights are dimensionless.
  // Preserve field order, widths and defaults. Native layout is not a wire format.
  struct PhotonPairLpmSuppressionRecord {
    ParticleState particle_{};
    std::uint64_t inputIndex_{};
    std::uint64_t componentHash_{};
    double survivalProbability_{};
    double uniform_{};
    std::uint64_t drawId_{};
  };
  static_assert(std::is_standard_layout_v<PhotonPairLpmSuppressionRecord>);
  static_assert(std::is_trivially_copyable_v<PhotonPairLpmSuppressionRecord>);
} // namespace corsika::kokkos
