/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>

namespace corsika::kokkos {

  /**
   * Read-only geometry, model and composition arrays for an exported node tree.
   *
   * All arrays must belong to the same valid EnvironmentSnapshot, with counts
   * representable by its 32-bit indices. A node addresses its own model index;
   * properties are not inherited from parents. Model indices, node indices and
   * physical medium IDs are distinct. Shared models remain shared records.
   *
   * Construction retains Kokkos views without copying or validating their data.
   * Only an execution space that can access the memory may dereference them.
   * Keep owning host views alive and arrays unmodified until kernels finish,
   * and release the views before Kokkos finalization. A device capture is not
   * a host owner. No native environment pointers are stored in this view.
   */
  template <typename TMemorySpace>
  class EnvironmentView {
  public:
    using tree_type = EnvironmentTreeView<TMemorySpace>;
    using node_index_type = typename tree_type::node_index_type;
    using model_view_type = Kokkos::View<EnvironmentModel const*, TMemorySpace>;
    using component_view_type = Kokkos::View<EnvironmentComponent const*, TMemorySpace>;

    EnvironmentView() = default;

    /// Bind the already-exported geometry, models and ordered number fractions.
    EnvironmentView(tree_type const& tree, model_view_type const& models,
                    component_view_type const& composition);

    KOKKOS_INLINE_FUNCTION tree_type const& getTree() const;

    /**
     * Return this node's model, or nullptr for unavailable storage or indices.
     *
     * This is an indexed lookup, not a location search or parent traversal.
     * The returned pointer is valid only in a space that can access the view.
     */
    KOKKOS_INLINE_FUNCTION EnvironmentModel const* getModel(node_index_type node) const;

    KOKKOS_INLINE_FUNCTION std::uint32_t getModelCount() const;
    KOKKOS_INLINE_FUNCTION std::uint32_t getComponentCount() const;
    /// Read a component at a valid global offset, without reordering fractions.
    KOKKOS_INLINE_FUNCTION EnvironmentComponent const& getComponent(
        std::uint32_t component) const;
    /// Whether the model view has backing storage; no model is evaluated.
    KOKKOS_INLINE_FUNCTION bool hasModelStorage() const;

  private:
    tree_type tree_;
    model_view_type models_;
    component_view_type composition_;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentView.inl>
