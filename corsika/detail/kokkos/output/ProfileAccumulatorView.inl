/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline constexpr std::size_t profile_histogram_bytes(
      std::size_t const bins) {
    return ProfileHistogramCount * bins * sizeof(long long);
  }

} // namespace corsika::kokkos
