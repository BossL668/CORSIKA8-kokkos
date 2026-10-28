/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/Types.hpp>

namespace corsika::kokkos::radio::firn {

  /// One phase-controlled finite-track child in surface-centred coordinates.
  struct FullWaveChildSegment {
    vector_type midpointM_{};
    vector_type displacementM_{};
    double midpointTimeS_{};
    double chargeWeight_{};
  };
} // namespace corsika::kokkos::radio::firn
