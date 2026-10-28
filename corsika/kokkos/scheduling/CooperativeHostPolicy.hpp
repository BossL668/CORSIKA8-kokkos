/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace corsika::kokkos {

  struct CooperativeHostPolicy {
    // Keep the laptop footprint; grow the shared arena (not one arena/thread)
    // for large machines, with a strict upper bound even for huge thread counts.
    static std::size_t capacity(std::size_t threads) noexcept;
    static double initialShare(std::size_t threads) noexcept;
  };

} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/CooperativeHostPolicy.inl>
