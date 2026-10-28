/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TMemorySpace>
  EnvironmentTreeView<TMemorySpace>::EnvironmentTreeView(
      node_view_type const& nodes, shape_view_type const& shapes,
      index_view_type const& children, index_view_type const& exclusions)
      : nodes_(nodes)
      , shapes_(shapes)
      , children_(children)
      , exclusions_(exclusions) {}

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION std::uint32_t EnvironmentTreeView<TMemorySpace>::getNodeCount()
      const {
    return static_cast<std::uint32_t>(nodes_.extent(0));
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION auto EnvironmentTreeView<TMemorySpace>::getParent(
      node_index_type node) const -> node_index_type {
    return nodes_(node).parentIndex_;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION EnvironmentShape const&
  EnvironmentTreeView<TMemorySpace>::getShape(node_index_type node) const {
    return shapes_(node);
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION std::uint32_t EnvironmentTreeView<TMemorySpace>::getChildCount(
      node_index_type node) const {
    return nodes_(node).childCount_;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION auto EnvironmentTreeView<TMemorySpace>::getChild(
      node_index_type node, std::uint32_t child) const -> node_index_type {
    return children_(nodes_(node).childOffset_ + child);
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION std::uint32_t
  EnvironmentTreeView<TMemorySpace>::getExcludedCount(node_index_type node) const {
    return nodes_(node).excludedCount_;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION auto EnvironmentTreeView<TMemorySpace>::getExcluded(
      node_index_type node, std::uint32_t excluded) const -> node_index_type {
    return exclusions_(nodes_(node).excludedOffset_ + excluded);
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION std::int32_t EnvironmentTreeView<TMemorySpace>::getModelIndex(
      node_index_type node) const {
    return nodes_(node).modelIndex_;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION bool EnvironmentTreeView<TMemorySpace>::hasNodeStorage() const {
    return nodes_.data() != nullptr;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION bool EnvironmentTreeView<TMemorySpace>::hasAdjacencyStorage()
      const {
    return (!children_.extent(0) || children_.data()) &&
           (!exclusions_.extent(0) || exclusions_.data());
  }

} // namespace corsika::kokkos
