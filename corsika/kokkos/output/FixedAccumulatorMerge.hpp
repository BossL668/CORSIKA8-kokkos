/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::kokkos {

  struct FixedAccumulatorLayout {
    // Identity covers algorithm, observers, binning and units, not just length.
    std::string identity_;
    double unitsPerInteger_{};
    std::size_t bins_{};
  };
  /** Add compatible integer histograms after checking every bin for overflow.
   * On a layout mismatch or overflow, throw without changing destination.
   */
  inline void merge_fixed_accumulators(std::vector<std::int64_t>& destination,
                                       FixedAccumulatorLayout const& a,
                                       std::vector<std::int64_t> const& source,
                                       FixedAccumulatorLayout const& b);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/FixedAccumulatorMerge.inl>
