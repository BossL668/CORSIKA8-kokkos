/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/BoundaryResult.hpp>

namespace corsika {

  /**
   * Select the next boundary using the production C8 ordered tree search.
   *
   * Check the current volume's exit, then children in list order, then exclusions
   * in list order. Equal parameters replace the previous result. Only children
   * use the additional negative-entry guard; clamp a final negative parameter
   * to zero without changing the selected node. This is not a global nearest-
   * surface search or a containing-node lookup.
   *
   * \param access Supplies intersect(node), getParent(node), getChildCount(node),
   *   getChild(node, index), getExcludedCount(node) and getExcluded(node, index).
   *   Intersections provide hasIntersections(), getEntry() and getExit().
   * \param current Valid current logical node; all adjacency handles must be valid.
   * \param limit Maximum trajectory parameter in the same units as intersections.
   * \return Selected parameter and next logical node; a parent handle may denote
   *   leaving the environment. Shape failures must be checked on the access object.
   *
   * Access owns or borrows its geometry; the search neither copies a tree nor
   * validates it. Preserve its original child and exclusion ordering. This host
   * entry retains native units and host-only node interfaces; the Kokkos entry
   * compiles the same implementation body for numerical device accessors.
   */
  template <typename TAccess, typename TNode, typename TParameter>
  [[nodiscard]] inline BoundaryResult<TParameter, TNode> next_environment_boundary(
      TAccess const& access, TNode current, TParameter limit);

} // namespace corsika

#include <corsika/detail/framework/geometry/NextEnvironmentBoundary.inl>
