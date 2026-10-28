/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/FrequencyGrid.hpp>
#include <complex>
#include <vector>

namespace corsika::kokkos::radio::firn {
  /// Radix-2 transform; the inverse includes the 1/N normalization.
  void fft_in_place(std::vector<std::complex<double>>& values, bool inverse);
  /// Reconstruct the real waveform from its positive-frequency SI spectrum.
  std::vector<double> inverse_real_spectrum(
      std::vector<std::complex<double>> positiveSpectrumVpmS, FrequencyGrid const& grid,
      double* maximumImaginary = nullptr);
  /// Relative Parseval residual with the physical time/frequency normalization.
  double parseval_relative(std::vector<double> const& waveform,
                           std::vector<std::complex<double>> const& positiveSpectrumVpmS,
                           FrequencyGrid const& grid);
} // namespace corsika::kokkos::radio::firn
#include <corsika/detail/kokkos/radio/firn/Fft.inl>
