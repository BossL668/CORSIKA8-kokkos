/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TMemorySpace>
  EnvironmentView<TMemorySpace>::EnvironmentView(tree_type const& tree,
                                                 model_view_type const& models,
                                                 component_view_type const& composition)
      : tree_(tree)
      , models_(models)
      , composition_(composition) {}

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION auto EnvironmentView<TMemorySpace>::getTree() const
      -> tree_type const& {
    return tree_;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION EnvironmentModel const* EnvironmentView<TMemorySpace>::getModel(
      node_index_type node) const {
    if (!tree_.hasNodeStorage() || !models_.data() || node < 0 ||
        static_cast<std::uint32_t>(node) >= tree_.getNodeCount()) {
      return nullptr;
    }
    auto index = tree_.getModelIndex(node);
    return index >= 0 && static_cast<std::uint32_t>(index) < getModelCount()
               ? models_.data() + index
               : nullptr;
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION std::uint32_t EnvironmentView<TMemorySpace>::getModelCount()
      const {
    return static_cast<std::uint32_t>(models_.extent(0));
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION std::uint32_t EnvironmentView<TMemorySpace>::getComponentCount()
      const {
    return static_cast<std::uint32_t>(composition_.extent(0));
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION EnvironmentComponent const&
  EnvironmentView<TMemorySpace>::getComponent(std::uint32_t component) const {
    return composition_(component);
  }

  template <typename TMemorySpace>
  KOKKOS_INLINE_FUNCTION bool EnvironmentView<TMemorySpace>::hasModelStorage() const {
    return models_.data() != nullptr;
  }

} // namespace corsika::kokkos
