/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace corsika::kokkos::radio {
  /** Three waveform components. Units follow the selected radio algorithm. */
  struct RadioWaveform {
    std::vector<double> x_{};
    std::vector<double> y_{};
    std::vector<double> z_{};
  };
} // namespace corsika::kokkos::radio
