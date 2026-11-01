/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::egs4::application {
  inline GpuMemoryBudget plan_gpu_memory(double fraction, std::size_t total,
                                         std::size_t free, std::size_t bytes_per_history,
                                         std::size_t reserve) {
    if (!std::isfinite(fraction) || fraction <= 0. || fraction > 1. || !total ||
        free > total || !bytes_per_history)
      throw std::invalid_argument("Invalid native EGS4 GPU memory budget");
    auto requested =
        fraction == 1.
            ? total
            : static_cast<std::size_t>(static_cast<long double>(total) * fraction);
    auto available = std::min(requested, free);
    if (available <= reserve)
      throw std::length_error("Insufficient GPU memory after the native EGS4 reserve");
    auto working = available - reserve;
    auto capacity = std::min(working / bytes_per_history,
                             std::size_t(std::numeric_limits<int>::max() / 5));
    if (capacity < 8192)
      throw std::length_error(
          "Insufficient GPU memory for the minimum native EGS4 queue");
    return {fraction,          total,        free, requested, working, reserve,
            bytes_per_history, int(capacity)};
  }
} // namespace corsika::kokkos::egs4::application
