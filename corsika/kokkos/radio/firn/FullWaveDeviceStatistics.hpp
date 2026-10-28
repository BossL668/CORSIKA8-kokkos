/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace corsika::kokkos::radio::firn {

  /// Transfer and execution counters; times include completion fences.
  struct FullWaveDeviceStatistics {
    std::uint64_t inputSegments_{};
    std::uint64_t batches_{};
    std::uint64_t responseQueries_{};
    std::uint64_t tableDomainFailures_{};
    std::size_t tableBytes_{};
    std::size_t spectrumBytes_{};
    std::size_t workspaceBytes_{};
    std::size_t totalDeviceBytes_{};
    std::size_t maximumBatchSegments_{};
    double hostToDeviceMiB_{};
    double deviceToHostMiB_{};
    double geometryKernelMs_{};
    double spectrumKernelMs_{};
    double transferMs_{};
  };
} // namespace corsika::kokkos::radio::firn
