/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <complex>
#include <vector>

namespace corsika::kokkos::radio::firn {

  /// Total and branch-resolved Fourier spectra, with the archived FFT convention.
  struct FullWaveSpectra {
    using components_type = std::array<std::vector<std::complex<double>>, 3>;
    std::vector<components_type> total_;
    std::vector<components_type> direct_;
    std::vector<components_type> boundary_;
  };
} // namespace corsika::kokkos::radio::firn
