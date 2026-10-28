/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline std::size_t checked_memory_add(std::size_t const left, std::size_t const right) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
      throw std::overflow_error("Kokkos resident byte count overflow");
    return left + right;
  }

  inline std::size_t checked_memory_multiply(std::size_t const count,
                                             std::size_t const element_bytes) {
    if (element_bytes != 0 &&
        count > std::numeric_limits<std::size_t>::max() / element_bytes)
      throw std::overflow_error("Kokkos resident byte count overflow");
    return count * element_bytes;
  }

} // namespace corsika::kokkos
