/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline std::size_t CooperativeHostPolicy::capacity(std::size_t threads) noexcept {
    auto target = 64 * std::clamp<std::size_t>(threads, 1, 256);
    std::size_t result = 2048;
    while (result < target) result *= 2;
    return result;
  }

  inline double CooperativeHostPolicy::initialShare(std::size_t threads) noexcept {
    return std::clamp(static_cast<double>(threads) / 320., 1. / 16., .5);
  }
} // namespace corsika::kokkos
