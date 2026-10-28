/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline StraightBoundaryAccess<TTreeView>::StraightBoundaryAccess(
      TTreeView const& tree, double const (&positionM)[3], double const (&velocity)[3])
      : tree_(tree)
      , positionM_{positionM[0], positionM[1], positionM[2]}
      , velocity_{velocity[0], velocity[1], velocity[2]} {}

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline straight_intersections_type
  StraightBoundaryAccess<TTreeView>::intersect(node_index_type node) const {
    return straight_intersect(tree_.getShape(node), positionM_, velocity_);
  }

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline auto StraightBoundaryAccess<TTreeView>::getParent(
      node_index_type node) const -> node_index_type {
    return tree_.getParent(node);
  }

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline std::uint32_t
  StraightBoundaryAccess<TTreeView>::getChildCount(node_index_type node) const {
    return tree_.getChildCount(node);
  }

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline auto StraightBoundaryAccess<TTreeView>::getChild(
      node_index_type node, std::uint32_t child) const -> node_index_type {
    return tree_.getChild(node, child);
  }

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline std::uint32_t
  StraightBoundaryAccess<TTreeView>::getExcludedCount(node_index_type node) const {
    return tree_.getExcludedCount(node);
  }

  template <typename TTreeView>
  CORSIKA_HOST_DEVICE inline auto StraightBoundaryAccess<TTreeView>::getExcluded(
      node_index_type node, std::uint32_t excluded) const -> node_index_type {
    return tree_.getExcluded(node, excluded);
  }

} // namespace corsika::kokkos
