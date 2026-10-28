/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline MemoryProjectionBuilder::MemoryProjectionBuilder(std::size_t const current)
      : retained_{current}
      , peak_{current} {}

  inline void MemoryProjectionBuilder::replace(std::size_t const old_bytes,
                                               std::size_t const new_bytes) {
    if (old_bytes > retained_)
      throw std::logic_error("Kokkos memory projection exceeds current retained bytes");
    peak_ = std::max(peak_, checked_memory_add(retained_, new_bytes));
    retained_ = checked_memory_add(retained_ - old_bytes, new_bytes);
  }

  inline void MemoryProjectionBuilder::merge(std::size_t const old_bytes,
                                             MemoryProjection const& nested) {
    if (old_bytes > retained_)
      throw std::logic_error("Kokkos nested memory projection exceeds retained bytes");
    auto const base = retained_ - old_bytes;
    peak_ = std::max(peak_, checked_memory_add(base, nested.transientPeakBytes_));
    retained_ = checked_memory_add(base, nested.retainedBytes_);
  }

  inline void MemoryProjectionBuilder::appendDisjointGrowth(
      std::size_t const initial_owner_bytes, MemoryProjection const& projected) {
    if (projected.retainedBytes_ < initial_owner_bytes ||
        projected.transientPeakBytes_ < projected.retainedBytes_)
      throw std::logic_error("invalid disjoint Kokkos growth projection");
    peak_ = std::max(peak_, checked_memory_add(retained_, projected.transientPeakBytes_ -
                                                              initial_owner_bytes));
    retained_ =
        checked_memory_add(retained_, projected.retainedBytes_ - initial_owner_bytes);
  }

  inline MemoryProjection MemoryProjectionBuilder::result() const noexcept {
    return {retained_, peak_};
  }
  inline bool ResidentMemoryBudgetGate::permits(std::size_t const current_owner_bytes,
                                                MemoryProjection const& projection,
                                                char const* const stage) const {
    return callback_ == nullptr ||
           callback_(context_, current_owner_bytes, projection, stage);
  }

} // namespace corsika::kokkos
