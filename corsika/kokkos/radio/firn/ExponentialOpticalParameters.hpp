/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  /** Exact numerical export of the nondispersive exponential optical profile.
   * Coordinates are in the same local, upward-z system as the CPU GO backend.
   * No host pointers or fitted/interpolated replacement profile are exported.
   */
  struct ExponentialOpticalParameters {
    double deepIndex_;
    double deltaIndex_;
    double scaleDepthM_;
    double surfaceZM_;
    double attenuationPerM_;
    double minimumZM_;
  };
} // namespace corsika::kokkos::radio::firn
