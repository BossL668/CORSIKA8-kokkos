/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/NextEnvironmentBoundary.hpp>
#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/environment/NextEnvironmentBoundary.hpp>
#include <corsika/kokkos/environment/StraightIntersections.hpp>

namespace corsika::kokkos {

  /**
   * Adapt read-only numerical geometry to the shared ordered boundary search.
   *
   * The trajectory is straight. Its parameter is time for a velocity in metres
   * per second, or distance for a unit direction. No magnetic update, model
   * lookup, containing-node search or observation-plane comparison is done.
   * The view's allocation, accessibility and valid-index preconditions apply.
   * Construct on the host or locally for each trajectory in a device kernel.
   * The adapter shares the tree view and copies six trajectory values, without
   * allocation. Retain owning host views until asynchronous kernels finish;
   * a device-local copy is not a host owner of the tree allocations.
   */
  template <typename TTreeView>
  class StraightBoundaryAccess {
  public:
    using node_index_type = typename TTreeView::node_index_type;

    /// Bind an exported tree and a trajectory in its common coordinate system.
    CORSIKA_HOST_DEVICE inline StraightBoundaryAccess(TTreeView const& tree,
                                                      double const (&positionM)[3],
                                                      double const (&velocity)[3]);

    /// Evaluate the production straight intersection for a valid node index.
    [[nodiscard]] CORSIKA_HOST_DEVICE inline straight_intersections_type intersect(
        node_index_type node) const;

    /// Borrow the parent handle, which may be the outside-environment sentinel.
    CORSIKA_HOST_DEVICE inline node_index_type getParent(node_index_type node) const;
    /// Number of ordered child handles of this node.
    CORSIKA_HOST_DEVICE inline std::uint32_t getChildCount(node_index_type node) const;
    /// Child handle at a valid local offset in the original list order.
    CORSIKA_HOST_DEVICE inline node_index_type getChild(node_index_type node,
                                                        std::uint32_t child) const;
    /// Number of ordered, directed exclusion handles of this node.
    CORSIKA_HOST_DEVICE inline std::uint32_t getExcludedCount(node_index_type node) const;
    /// Exclusion handle at a valid local offset in the original list order.
    CORSIKA_HOST_DEVICE inline node_index_type getExcluded(node_index_type node,
                                                           std::uint32_t excluded) const;

  private:
    TTreeView tree_;
    double positionM_[3];
    double velocity_[3];
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/StraightBoundaryAccess.inl>
