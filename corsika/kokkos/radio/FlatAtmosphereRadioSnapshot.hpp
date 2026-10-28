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
#include <vector>

namespace corsika::kokkos::radio {
  /** Host-owned production refractivity table, with nearest-bin lookup and endpoint
   * extrapolation. A positive homogeneousRefractiveIndex_ selects n R/c instead;
   * the caller must ensure the complete ray remains in that uniform medium.
   * A firn particle-density snapshot does not provide a curved radio ray model. */
  struct FlatAtmosphereRadioSnapshot {
    /** Zero retains the released atmospheric lookup. Positive values opt in
     * to exact homogeneous propagation (n R/c), with no altitude table.
     * The caller must ensure every entire ray stays in that homogeneous medium.
     */
    double homogeneousRefractiveIndex_{};
    double minimumHeightM_{};
    double maximumHeightM_{};
    double stepM_{};
    double inverseStepPerM_{};
    double slopeRefractivityLower_{};
    double slopeIntegratedRefractivityLower_{};
    double slopeRefractivityUpper_{};
    double slopeIntegratedRefractivityUpper_{};
    std::vector<double> refractivity_{};
    std::vector<double> integratedRefractivity_{};
  };
} // namespace corsika::kokkos::radio
