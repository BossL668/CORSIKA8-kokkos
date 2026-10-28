/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  /// Exact exponential-profile integrals across the monotonic/turning junction.
  struct ExponentialEndpointData {
    /// Coordinate-switch threshold, not an exclusion of physical ray directions.
    static constexpr double cosineLimit_ = 0.002;
    double invariantP_;
    double horizontalM_;
    double geometricLengthM_;
    double opticalPathM_;
    double horizontalDerivativeM_;
    double deepCosine_;
  };
} // namespace corsika::kokkos::radio::firn
