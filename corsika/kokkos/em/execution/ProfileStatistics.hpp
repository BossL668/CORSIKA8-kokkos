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
#include <string>
#include <vector>

namespace corsika::kokkos {
  struct ProfileStatistics {
    bool enabled_{};
    bool deterministic_{};
    std::uint64_t steps_{};
    std::uint64_t depositedSteps_{};
    std::uint64_t fixedPointOverflows_{};
    std::uint64_t invalidRecords_{};
    std::size_t bins_{};
    std::size_t deviceBytes_{};
    std::uint64_t hostToDeviceBytes_{};
    std::uint64_t deviceToHostBytes_{};
    double kernelTimeMs_{};
    double transferTimeMs_{};
  };
} // namespace corsika::kokkos
