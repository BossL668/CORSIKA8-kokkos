/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/FrequencyGrid.hpp>

namespace corsika::kokkos::radio::firn {

  /// Resource limits for immutable full-wave tables and bounded segment batches.
  struct FullWaveDeviceConfig {
    FrequencyGrid grid_{};
    double timeOriginS_{};
    std::size_t batchSegments_{4096};
    std::size_t memoryLimitBytes_{512ULL * 1024ULL * 1024ULL};
    bool retainBranchSpectra_{true};
    void validate() const;
  };
} // namespace corsika::kokkos::radio::firn
#include <corsika/detail/kokkos/radio/firn/FullWaveDeviceConfig.inl>
