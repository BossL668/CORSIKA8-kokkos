/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /**
   * Borrowed numeric arrays of one exported environment, without native pointers.
   *
   * Nodes and shapes have the same count and index order. All arrays must belong
   * to one valid snapshot and be accessible from the execution space using them.
   * This record neither owns storage nor extends its lifetime. Its owner must
   * remain alive and unchanged until all work using the record has completed.
   * Native byte layout is not a portable serialization format.
   */
  struct EnvironmentDataView {
    EnvironmentNode const* nodes_{};
    EnvironmentShape const* shapes_{};
    std::int32_t const* children_{};
    std::int32_t const* exclusions_{};
    EnvironmentModel const* models_{};
    EnvironmentComponent const* composition_{};
    std::uint32_t nodeCount_{};
    std::uint32_t childCount_{};
    std::uint32_t exclusionCount_{};
    std::uint32_t modelCount_{};
    std::uint32_t componentCount_{};
  };

  static_assert(std::is_trivially_copyable_v<EnvironmentDataView>);
  static_assert(std::is_standard_layout_v<EnvironmentDataView>);

} // namespace corsika::kokkos
