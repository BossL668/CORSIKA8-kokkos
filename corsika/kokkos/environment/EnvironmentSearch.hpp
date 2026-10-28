/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

namespace corsika::kokkos {

  /**
   * Evaluate the production containment predicate on an exported shape.
   *
   * \param shape Supported shape from export_shape(); Invalid is outside the contract.
   * \param positionM Position in metres in the shape's common coordinate system.
   * \return Universe always contains the point; sphere and box surfaces are
   * excluded, while a separation plane includes its surface.
   *
   * Preserves production arithmetic and comparisons without adding tolerances.
   * Physical queries require finite positions. Exceptional IEEE inputs preserve
   * the production predicate, not necessarily native C8 behavior on such inputs.
   */
  CORSIKA_HOST_DEVICE inline bool contains(EnvironmentShape const& shape,
                                           double const (&positionM)[3]);

  /**
   * Locate the responsible node using the production iterative C8 search order.
   *
   * \param tree Valid exported arrays with the EnvironmentTreeView preconditions.
   * \param positionM Position in metres in the snapshot's common coordinate system.
   * \param start Starting node index, normally the root (zero).
   * \return The selected node index, -1 when outside the starting volume, or -2
   * for an invalid starting index, an empty tree or an exhausted cycle limit.
   *
   * Tests the current volume, then the first containing child in list order,
   * then (only if no child contains the point) the first containing exclusion.
   * A node without model properties may be returned. Never climbs to a parent,
   * skips a model-free node or substitutes another material. At most node-count
   * iterations are used, as in production; a reachable exclusion cycle fails
   * explicitly. Child/exclusion backing arrays are not checked per query.
   * No distance, boundary intersection or observation-plane calculation is done.
   */
  template <typename TTreeView>
  [[nodiscard]] CORSIKA_HOST_DEVICE inline std::int32_t locate_node(
      TTreeView const& tree, double const (&positionM)[3], std::int32_t start = 0);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentSearch.inl>
