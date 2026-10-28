/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentDataView.hpp>

#include <corsika/framework/core/HostDevice.hpp>
#include <type_traits>

namespace corsika::kokkos {

  /**
   * Non-owning access to the same numeric environment arrays used by managed views.
   *
   * Suitable for copying into a device context. This class only accesses arrays;
   * node search, boundary competition and model calculations use the shared query
   * implementations. The array owner must remain alive and unchanged until all
   * queries finish. Accessible memory and valid exported indices are preconditions.
   */
  class EnvironmentBorrowedView {
  public:
    using tree_type = EnvironmentBorrowedView;
    using node_index_type = std::int32_t;

    CORSIKA_HOST_DEVICE inline EnvironmentBorrowedView() = default;
    CORSIKA_HOST_DEVICE inline explicit EnvironmentBorrowedView(EnvironmentDataView data);

    CORSIKA_HOST_DEVICE inline tree_type const& getTree() const;
    CORSIKA_HOST_DEVICE inline std::uint32_t getNodeCount() const;
    CORSIKA_HOST_DEVICE inline node_index_type getParent(node_index_type node) const;
    CORSIKA_HOST_DEVICE inline EnvironmentShape const& getShape(
        node_index_type node) const;
    CORSIKA_HOST_DEVICE inline std::uint32_t getChildCount(node_index_type node) const;
    CORSIKA_HOST_DEVICE inline node_index_type getChild(node_index_type node,
                                                        std::uint32_t child) const;
    CORSIKA_HOST_DEVICE inline std::uint32_t getExcludedCount(node_index_type node) const;
    CORSIKA_HOST_DEVICE inline node_index_type getExcluded(node_index_type node,
                                                           std::uint32_t excluded) const;
    CORSIKA_HOST_DEVICE inline std::int32_t getModelIndex(node_index_type node) const;
    CORSIKA_HOST_DEVICE inline bool hasNodeStorage() const;
    CORSIKA_HOST_DEVICE inline bool hasAdjacencyStorage() const;
    CORSIKA_HOST_DEVICE inline EnvironmentModel const* getModel(
        node_index_type node) const;
    CORSIKA_HOST_DEVICE inline std::uint32_t getModelCount() const;
    CORSIKA_HOST_DEVICE inline std::uint32_t getComponentCount() const;
    CORSIKA_HOST_DEVICE inline EnvironmentComponent const& getComponent(
        std::uint32_t component) const;
    CORSIKA_HOST_DEVICE inline bool hasModelStorage() const;

  private:
    EnvironmentDataView data_;
  };

  static_assert(std::is_trivially_copyable_v<EnvironmentBorrowedView>);
  static_assert(std::is_standard_layout_v<EnvironmentBorrowedView>);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentBorrowedView.inl>
