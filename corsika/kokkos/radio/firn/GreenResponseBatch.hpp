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
#include <corsika/kokkos/radio/firn/GreenResponseBranch.hpp>
#include <corsika/kokkos/radio/firn/VectorOperations.hpp>

namespace corsika::kokkos::radio::firn {
  /** GreenResponseBatch from the validated stratified-radio implementation. */
  struct GreenResponseBatch {
    // Aggregate response retained for single-path callers and table oracles.
    // When branches is non-empty, OnlineRadioProcess uses the branches and does
    // not apply one representative finite-track factor to this aggregate.
    std::vector<complex_matrix_type> dyadicPerM_;
    PathDiagnostics path_;
    FiniteTrackCoupling finiteTrackCoupling_{FiniteTrackCoupling::EikonalSinc};
    std::vector<GreenResponseBranch> branches_;
  };
} // namespace corsika::kokkos::radio::firn
