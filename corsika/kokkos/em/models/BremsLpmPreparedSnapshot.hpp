/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/models/BremsLpm.hpp>
#include <type_traits>
namespace corsika::kokkos {
  /** Production component terms prepared once in the execution space that uses
   * them. Preserve that space's pow/log results; these records own no memory.
   * The execution layer must fill valid_, componentCount_, s1_ and logarithmS1_.
   */
  struct BremsLpmPreparedComponent {
    std::uint64_t componentHash_{};
    double s1_{};
    double logarithmS1_{};
  };
  struct BremsLpmPreparedSnapshot {
    std::uint32_t valid_{};
    std::uint32_t componentCount_{};
    BremsLpmPreparedComponent components_[MaxBremsLpmComponents]{};
  };
  static_assert(std::is_standard_layout_v<BremsLpmPreparedSnapshot>);
  static_assert(std::is_trivially_copyable_v<BremsLpmPreparedSnapshot>);
} // namespace corsika::kokkos
