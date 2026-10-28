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
  /** Observer position in the common coordinate system (m), times (s) and rate (Hz). */
  struct RadioObserverSnapshot {
    double positionM_[3]{};
    double startTimeS_{};
    double durationS_{};
    double sampleRateHz_{};
    std::uint64_t numberOfBins_{};
  };
} // namespace corsika::kokkos::radio
