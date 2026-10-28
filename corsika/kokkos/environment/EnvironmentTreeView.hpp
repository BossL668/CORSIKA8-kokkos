/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace corsika::kokkos {

  /**
   * Read-only Kokkos views of an exported environment's geometry and topology.
   *
   * The node and shape views have the same extent and share preorder indices.
   * Child and exclusion indices address these views, with the ordering and
   * offsets from EnvironmentTopology. Model indices are not used for location.
   * These views contain device-accessible values, not native C8 object pointers.
   * Kokkos retains the shared allocations; no data are copied or validated here.
   * Keep the arrays unmodified while a search or a captured kernel uses them.
   * Retain an owning host view until asynchronous kernels finish, and release
   * the views before Kokkos finalization; a device capture is not a host owner.
   *
   * Only the selected execution space may dereference its accessible memory.
   * The default empty view is allowed and produces an invalid-location result.
   * Construction, copying and destruction follow the normal Kokkos lifecycle.
   */
  template <typename TMemorySpace>
  class EnvironmentTreeView {
  public:
    using node_index_type = std::int32_t;
    using node_view_type = Kokkos::View<EnvironmentNode const*, TMemorySpace>;
    using shape_view_type = Kokkos::View<EnvironmentShape const*, TMemorySpace>;
    using index_view_type = Kokkos::View<node_index_type const*, TMemorySpace>;

    EnvironmentTreeView() = default;

    /**
     * Retain the four numerical arrays without changing their values or order.
     *
     * All arrays must come from the same valid exported snapshot. Each node
     * needs a supported shape; adjacency ranges and indices must be valid.
     * The node count must fit the exported signed 32-bit index range. No search
     * validates backing arrays or repairs malformed topology on the device.
     *
     * \param nodes Exported topology records.
     * \param shapes One shape per node in the same common coordinate system.
     * \param children Ordered child-node index array.
     * \param exclusions Ordered, directed excluded-node index array.
     */
    EnvironmentTreeView(node_view_type const& nodes, shape_view_type const& shapes,
                        index_view_type const& children,
                        index_view_type const& exclusions);

    KOKKOS_INLINE_FUNCTION std::uint32_t getNodeCount() const;
    KOKKOS_INLINE_FUNCTION node_index_type getParent(node_index_type node) const;
    KOKKOS_INLINE_FUNCTION EnvironmentShape const& getShape(node_index_type node) const;
    KOKKOS_INLINE_FUNCTION std::uint32_t getChildCount(node_index_type node) const;
    KOKKOS_INLINE_FUNCTION node_index_type getChild(node_index_type node,
                                                    std::uint32_t child) const;
    KOKKOS_INLINE_FUNCTION std::uint32_t getExcludedCount(node_index_type node) const;
    KOKKOS_INLINE_FUNCTION node_index_type getExcluded(node_index_type node,
                                                       std::uint32_t excluded) const;

    /// Model index of a valid node; a negative index means no properties.
    KOKKOS_INLINE_FUNCTION std::int32_t getModelIndex(node_index_type node) const;
    /// Whether the node view has backing storage; no topology is traversed.
    KOKKOS_INLINE_FUNCTION bool hasNodeStorage() const;
    /// Nonempty child and exclusion views must have backing storage.
    KOKKOS_INLINE_FUNCTION bool hasAdjacencyStorage() const;

  private:
    node_view_type nodes_;
    shape_view_type shapes_;
    index_view_type children_;
    index_view_type exclusions_;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentTreeView.inl>
