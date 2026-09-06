/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace corsika::accelerator::em::kokkos_detail {

  inline std::size_t checkedMemoryAdd(std::size_t const left,
                                      std::size_t const right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
      throw std::overflow_error("Kokkos resident byte count overflow");
    return left + right;
  }

  inline std::size_t checkedMemoryMultiply(std::size_t const count,
                                           std::size_t const element_bytes) {
    if (element_bytes != 0 &&
        count > std::numeric_limits<std::size_t>::max() / element_bytes)
      throw std::overflow_error("Kokkos resident byte count overflow");
    return count * element_bytes;
  }

  /** Retained bytes after growth and the peak while old/new storage overlaps. */
  struct KokkosMemoryProjection {
    std::size_t retained_bytes{};
    std::size_t transient_peak_bytes{};
  };

  /**
   * Compose a projection in allocation order.
   *
   * replace() models `member = newly_allocated_view`: the complete new
   * allocation exists before the old member is released. merge() embeds a
   * projection produced by a nested owner such as a queue or bucket workspace.
   */
  class KokkosMemoryProjectionBuilder {
  public:
    explicit KokkosMemoryProjectionBuilder(std::size_t const current)
        : retained_{current}, peak_{current} {}

    void replace(std::size_t const old_bytes,
                 std::size_t const new_bytes) {
      if (old_bytes > retained_)
        throw std::logic_error(
            "Kokkos memory projection exceeds current retained bytes");
      peak_ = std::max(peak_, checkedMemoryAdd(retained_, new_bytes));
      retained_ = checkedMemoryAdd(retained_ - old_bytes, new_bytes);
    }

    void merge(std::size_t const old_bytes,
               KokkosMemoryProjection const& nested) {
      if (old_bytes > retained_)
        throw std::logic_error(
            "Kokkos nested memory projection exceeds retained bytes");
      auto const base = retained_ - old_bytes;
      peak_ = std::max(
          peak_, checkedMemoryAdd(base, nested.transient_peak_bytes));
      retained_ = checkedMemoryAdd(base, nested.retained_bytes);
    }

    /** Compose disjoint allocations projected against the same initial owner. */
    void appendDisjointGrowth(std::size_t const initial_owner_bytes,
                              KokkosMemoryProjection const& projected) {
      if (projected.retained_bytes < initial_owner_bytes ||
          projected.transient_peak_bytes < projected.retained_bytes)
        throw std::logic_error("invalid disjoint Kokkos growth projection");
      peak_ = std::max(peak_, checkedMemoryAdd(
          retained_, projected.transient_peak_bytes - initial_owner_bytes));
      retained_ = checkedMemoryAdd(
          retained_, projected.retained_bytes - initial_owner_bytes);
    }

    KokkosMemoryProjection result() const noexcept {
      return {retained_, peak_};
    }

  private:
    std::size_t retained_{};
    std::size_t peak_{};
  };

  /**
   * Host-only callback used immediately before a grow-only device allocation.
   * The callback receives the current byte count of the owner being replaced,
   * so the backend can combine the projection with every other retained owner.
   */
  struct KokkosResidentMemoryBudgetGate {
    using Callback = bool (*)(void*, std::size_t,
                              KokkosMemoryProjection const&, char const*);

    void* context{};
    Callback callback{};

    bool permits(std::size_t const current_owner_bytes,
                 KokkosMemoryProjection const& projection,
                 char const* const stage) const {
      return callback == nullptr ||
             callback(context, current_owner_bytes, projection, stage);
    }
  };

} // namespace corsika::accelerator::em::kokkos_detail
