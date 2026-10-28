/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <corsika/kokkos/radio/firn/FrequencyGrid.hpp>
#include <corsika/kokkos/radio/firn/PathDiagnostics.hpp>
#include <corsika/kokkos/radio/firn/VectorOperations.hpp>

namespace corsika::kokkos::radio::firn {
  /** Coupling contract between one finite charged step and a Green response. */
  enum class FiniteTrackCoupling {
    // A unique launch eikonal exists, so the centered finite-track sinc is exact.
    EikonalSinc,
    // The dyadic contains a continuum of wave vectors.  The online process must
    // subdivide to the certified point-current limit and must not add one ray sinc.
    PointCurrentDyadic,
  };
} // namespace corsika::kokkos::radio::firn
