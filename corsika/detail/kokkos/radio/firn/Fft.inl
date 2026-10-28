/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/Types.hpp>
#include <algorithm>

namespace corsika::kokkos::radio::firn {

  inline void fft_in_place(std::vector<std::complex<double>>& values, bool inverse) {
    std::size_t const count = values.size();
    if (count < 2 || (count & (count - 1)) != 0) {
      throw std::runtime_error("radix-2 FFT requires a power-of-two length");
    }
    for (std::size_t index = 1, componentIndex = 0; index < count; ++index) {
      std::size_t bit = count >> 1;
      for (; componentIndex & bit; bit >>= 1) componentIndex ^= bit;
      componentIndex ^= bit;
      if (index < componentIndex) std::swap(values[index], values[componentIndex]);
    }
    for (std::size_t length = 2; length <= count; length <<= 1) {
      double const angle = (inverse ? 2.0 : -2.0) * pi / static_cast<double>(length);
      std::complex<double> const root(std::cos(angle), std::sin(angle));
      for (std::size_t offset = 0; offset < count; offset += length) {
        std::complex<double> phase(1.0, 0.0);
        for (std::size_t componentIndex = 0; componentIndex < length / 2;
             ++componentIndex) {
          auto const even = values[offset + componentIndex];
          auto const odd = values[offset + componentIndex + length / 2] * phase;
          values[offset + componentIndex] = even + odd;
          values[offset + componentIndex + length / 2] = even - odd;
          phase *= root;
        }
      }
    }
    if (inverse) {
      for (auto& value : values) value /= static_cast<double>(count);
    }
  }

  inline std::vector<double> inverse_real_spectrum(
      std::vector<std::complex<double>> positiveSpectrumVpmS, FrequencyGrid const& grid,
      double* maximumImaginary) {
    grid.validate();
    if (positiveSpectrumVpmS.size() != grid.frequencyCount()) {
      throw std::runtime_error("positive spectrum has wrong length");
    }
    positiveSpectrumVpmS.front() = {positiveSpectrumVpmS.front().real(), 0.0};
    positiveSpectrumVpmS.back() = {positiveSpectrumVpmS.back().real(), 0.0};
    std::vector<std::complex<double>> full(grid.sampleCount_, 0.0);
    for (std::size_t index = 0; index < positiveSpectrumVpmS.size(); ++index) {
      full[index] = positiveSpectrumVpmS[index] / grid.sampleIntervalS_;
    }
    for (std::size_t index = 1; index + 1 < positiveSpectrumVpmS.size(); ++index) {
      full[grid.sampleCount_ - index] = std::conj(full[index]);
    }
    fft_in_place(full, true);
    std::vector<double> result(full.size());
    double maxImag = 0.0;
    for (std::size_t index = 0; index < full.size(); ++index) {
      result[index] = full[index].real();
      maxImag = std::max(maxImag, std::abs(full[index].imag()));
    }
    if (maximumImaginary) *maximumImaginary = maxImag;
    return result;
  }

  inline double parseval_relative(
      std::vector<double> const& waveform,
      std::vector<std::complex<double>> const& positiveSpectrumVpmS,
      FrequencyGrid const& grid) {
    if (waveform.size() != grid.sampleCount_ ||
        positiveSpectrumVpmS.size() != grid.frequencyCount()) {
      throw std::runtime_error("Parseval arrays have incompatible size");
    }
    double timeEnergy = 0.0;
    for (double value : waveform) timeEnergy += value * value;
    timeEnergy *= grid.sampleIntervalS_;
    double frequencyEnergy =
        std::norm(positiveSpectrumVpmS.front()) + std::norm(positiveSpectrumVpmS.back());
    for (std::size_t index = 1; index + 1 < positiveSpectrumVpmS.size(); ++index) {
      frequencyEnergy += 2.0 * std::norm(positiveSpectrumVpmS[index]);
    }
    frequencyEnergy *= grid.spacingHz();
    double const scale = std::max({timeEnergy, frequencyEnergy, 1.0e-300});
    return std::abs(timeEnergy - frequencyEnergy) / scale;
  }

} // namespace corsika::kokkos::radio::firn
