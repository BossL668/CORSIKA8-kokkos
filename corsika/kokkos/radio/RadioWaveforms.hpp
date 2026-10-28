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
#include <corsika/kokkos/radio/RadioWaveform.hpp>

namespace corsika::kokkos::radio {
  /** CoREAS stores electric-field samples; ZHS stores vector-potential samples.
   * The output layer must retain C8's adjacent-sample differentiation for ZHS. */
  struct RadioWaveforms {
    std::vector<RadioWaveform> coreas_{};
    std::vector<RadioWaveform> zhs_{};
  };
} // namespace corsika::kokkos::radio
