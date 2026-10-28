/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline void merge_fixed_accumulators(std::vector<std::int64_t>& destination,
                                       FixedAccumulatorLayout const& a,
                                       std::vector<std::int64_t> const& source,
                                       FixedAccumulatorLayout const& b) {
    if (a.identity_.empty() || a.identity_ != b.identity_ || a.bins_ != b.bins_ ||
        !std::isfinite(a.unitsPerInteger_) || a.unitsPerInteger_ <= 0. ||
        a.unitsPerInteger_ != b.unitsPerInteger_ || destination.size() != a.bins_ ||
        source.size() != b.bins_)
      throw std::invalid_argument("incompatible fixed-point accumulators");
    // Preflight every bin before writing: failed merges cannot partially commit.
    for (std::size_t i = 0; i < source.size(); ++i) {
      auto x = destination[i], y = source[i];
      if ((y > 0 && x > std::numeric_limits<std::int64_t>::max() - y) ||
          (y < 0 && x < std::numeric_limits<std::int64_t>::min() - y))
        throw std::overflow_error("cooperative fixed-point accumulation overflow");
    }
    for (std::size_t i = 0; i < source.size(); ++i) destination[i] += source[i];
  }

} // namespace corsika::kokkos
