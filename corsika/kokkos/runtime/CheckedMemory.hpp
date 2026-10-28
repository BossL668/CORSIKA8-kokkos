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

  /** Add byte counts, throwing overflow_error if size_t would overflow. */
  inline std::size_t checked_memory_add(std::size_t const left, std::size_t const right);
  /** Compute allocation bytes with the production size_t overflow check. */
  inline std::size_t checked_memory_multiply(std::size_t const count,
                                             std::size_t const element_bytes);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/CheckedMemory.inl>
