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
#include <corsika/kokkos/runtime/CheckedMemory.hpp>

namespace corsika::kokkos {

  /** Retained allocation size and temporary growth peak, both in bytes. */
  struct MemoryProjection {
    std::size_t retainedBytes_{};
    std::size_t transientPeakBytes_{};
  };
  /** Compose allocation growth in order without allocating or querying a GPU. */
  class MemoryProjectionBuilder {
  public:
    explicit MemoryProjectionBuilder(std::size_t const current);

    void replace(std::size_t const old_bytes, std::size_t const new_bytes);

    void merge(std::size_t const old_bytes, MemoryProjection const& nested);

    /** Compose disjoint allocations projected against the same initial owner. */
    void appendDisjointGrowth(std::size_t const initial_owner_bytes,
                              MemoryProjection const& projected);

    MemoryProjection result() const noexcept;

  private:
    std::size_t retained_{};
    std::size_t peak_{};
  };
  /** Borrow an optional owner callback; a null callback permits the projection.
   * The context must outlive calls to permits(). No memory is reserved here.
   */
  struct ResidentMemoryBudgetGate {
    using callback_type = bool (*)(void*, std::size_t, MemoryProjection const&,
                                   char const*);

    void* context_{};
    callback_type callback_{};

    bool permits(std::size_t const current_owner_bytes,
                 MemoryProjection const& projection, char const* const stage) const;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/ResidentMemoryBudget.inl>
