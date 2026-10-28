/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <type_traits>
namespace corsika::kokkos {
  /** Production fixed-point output counters, separate from particle dynamics.
   * Signed energy sums retain the production mass/observation/closure conventions.
   * leptonLimits_ retains the original eight slots. Non-absorbing material
   * interfaces have a separate appended counter; unknown limits remain invalid.
   */
  struct ProfileCounters {
    unsigned long long steps_{};
    unsigned long long depositedSteps_{};
    unsigned long long photonCuts_{};
    unsigned long long leptonLimits_[8]{};
    unsigned long long moliereTrials_{};
    unsigned long long moliereDeflections_{};
    unsigned long long moliereZeroDeflections_{};
    unsigned long long moliereNewtonIterations_{};
    unsigned long long moliereMaxNewtonIterations_{};
    unsigned long long thinningHillasVertices_{};
    unsigned long long thinningStatisticalVertices_{};
    unsigned long long thinningParticlesDiscarded_{};
    unsigned long long fixedPointOverflows_{};
    unsigned long long invalidRecords_{};
    long long weightedMediumRestMassInput_{};
    long long weightedCutRestMassEnergy_{};
    long long weightedObservedTotalEnergy_{};
    long long weightedEscapedTotalEnergy_{};
    long long weightedUnwrittenPhotoelectricBindingEnergy_{};
    long long weightedObservationCutOverlapEnergy_{};
    long long weightedMassConventionCorrection_{};
    long long weightedNativeMuonPairCutEnergy_{};
    unsigned long long leptonMaterialBoundaries_{};
  };
  static_assert(std::is_standard_layout_v<ProfileCounters>);
  static_assert(std::is_trivially_copyable_v<ProfileCounters>);
} // namespace corsika::kokkos
