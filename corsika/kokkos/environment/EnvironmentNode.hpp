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
   * Pointer-free topology of one environment node, suitable for device copies.
   *
   * Child and exclusion offsets address separate arrays of node indices. Their
   * order is the original C8 order, not a sorted or deduplicated neighbour list.
   * The root has index zero and parent -1. Model indices identify shared host
   * property objects; -1 means no properties on this node, without inheritance.
   * A model index is not yet a transport medium or a physical model payload.
   *
   * This record contains no volume, density, field or observation surface. It
   * cannot perform geometry queries by itself. Native padding and byte order
   * are not a portable serialization format.
   */
  struct EnvironmentNode {
    std::int32_t parentIndex_{-1};
    std::int32_t modelIndex_{-1};
    std::uint32_t childOffset_{};
    std::uint32_t childCount_{};
    std::uint32_t excludedOffset_{};
    std::uint32_t excludedCount_{};
  };

} // namespace corsika::kokkos
