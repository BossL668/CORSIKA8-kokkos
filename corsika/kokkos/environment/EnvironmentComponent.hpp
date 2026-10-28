/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Pointer-free copy of one native nuclear-composition entry.
   *
   * code_ is a C8 Code enum value, not a PDG code. fraction_ is the native
   * dimensionless number fraction. Export preserves entry order, duplicates and
   * stored values without normalizing, sorting or combining components.
   */
  struct EnvironmentComponent {
    std::int32_t code_{};
    double fraction_{};
  };

} // namespace corsika::kokkos
