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
#include <corsika/kokkos/radio/firn/VectorOperations.hpp>

namespace corsika::kokkos::radio::firn {
  /** Real-signal frequency grid: N/2+1 bins from zero through Nyquist. */
  struct FrequencyGrid {
    double sampleIntervalS_{1.0e-9};
    std::size_t sampleCount_{4096};

    void validate() const;

    std::size_t frequencyCount() const;

    double spacingHz() const;

    double frequencyHz(std::size_t index) const;

    double nyquistHz() const;
  };
} // namespace corsika::kokkos::radio::firn

#include <corsika/detail/kokkos/radio/firn/FrequencyGrid.inl>
