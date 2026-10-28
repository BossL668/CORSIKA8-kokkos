/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline std::size_t checked_geometric_capacity(std::size_t const current,
                                                std::size_t const requested,
                                                std::size_t const maximum) {
    if (requested > maximum)
      throw std::length_error("Kokkos grow-only allocation exceeds configured capacity");
    if (requested <= current) return current;
    if (current == 0) return requested;
    if (current > maximum / 2) return maximum;
    return std::min(maximum, std::max(requested, current * 2));
  }

} // namespace corsika::kokkos
