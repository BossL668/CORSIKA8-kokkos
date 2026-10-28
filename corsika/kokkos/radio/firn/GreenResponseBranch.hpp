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
#include <corsika/kokkos/radio/firn/FiniteTrackCoupling.hpp>
#include <corsika/kokkos/radio/firn/VectorOperations.hpp>

namespace corsika::kokkos::radio::firn {
  /** One physically distinct propagation contribution.
   *
   * A finite charged track must be projected and phase-integrated separately
   * for every ray branch because emitDirection and travelTimeS are branch
   * dependent.  Summing dyadics before applying the finite-track sinc would be
   * incorrect for multipath propagation.
   */
  struct GreenResponseBranch {
    std::vector<complex_matrix_type> dyadicPerM_;
    PathDiagnostics path_;
    FiniteTrackCoupling finiteTrackCoupling_{FiniteTrackCoupling::EikonalSinc};
  };
} // namespace corsika::kokkos::radio::firn
