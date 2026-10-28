/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline EnvironmentBorrowedView::EnvironmentBorrowedView(
      EnvironmentDataView data)
      : data_(data) {}

  CORSIKA_HOST_DEVICE inline auto EnvironmentBorrowedView::getTree() const
      -> tree_type const& {
    return *this;
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t EnvironmentBorrowedView::getNodeCount() const {
    return data_.nodeCount_;
  }

  CORSIKA_HOST_DEVICE inline auto EnvironmentBorrowedView::getParent(
      node_index_type node) const -> node_index_type {
    return data_.nodes_[node].parentIndex_;
  }

  CORSIKA_HOST_DEVICE inline EnvironmentShape const& EnvironmentBorrowedView::getShape(
      node_index_type node) const {
    return data_.shapes_[node];
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t EnvironmentBorrowedView::getChildCount(
      node_index_type node) const {
    return data_.nodes_[node].childCount_;
  }

  CORSIKA_HOST_DEVICE inline auto EnvironmentBorrowedView::getChild(
      node_index_type node, std::uint32_t child) const -> node_index_type {
    return data_.children_[data_.nodes_[node].childOffset_ + child];
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t EnvironmentBorrowedView::getExcludedCount(
      node_index_type node) const {
    return data_.nodes_[node].excludedCount_;
  }

  CORSIKA_HOST_DEVICE inline auto EnvironmentBorrowedView::getExcluded(
      node_index_type node, std::uint32_t excluded) const -> node_index_type {
    return data_.exclusions_[data_.nodes_[node].excludedOffset_ + excluded];
  }

  CORSIKA_HOST_DEVICE inline std::int32_t EnvironmentBorrowedView::getModelIndex(
      node_index_type node) const {
    return data_.nodes_[node].modelIndex_;
  }

  CORSIKA_HOST_DEVICE inline bool EnvironmentBorrowedView::hasNodeStorage() const {
    return data_.nodes_ != nullptr;
  }

  CORSIKA_HOST_DEVICE inline bool EnvironmentBorrowedView::hasAdjacencyStorage() const {
    return (!data_.childCount_ || data_.children_) &&
           (!data_.exclusionCount_ || data_.exclusions_);
  }

  CORSIKA_HOST_DEVICE inline EnvironmentModel const* EnvironmentBorrowedView::getModel(
      node_index_type node) const {
    if (!hasNodeStorage() || !data_.models_ || node < 0 ||
        static_cast<std::uint32_t>(node) >= getNodeCount()) {
      return nullptr;
    }
    auto index = getModelIndex(node);
    return index >= 0 && static_cast<std::uint32_t>(index) < getModelCount()
               ? data_.models_ + index
               : nullptr;
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t EnvironmentBorrowedView::getModelCount()
      const {
    return data_.modelCount_;
  }

  CORSIKA_HOST_DEVICE inline std::uint32_t EnvironmentBorrowedView::getComponentCount()
      const {
    return data_.componentCount_;
  }

  CORSIKA_HOST_DEVICE inline EnvironmentComponent const&
  EnvironmentBorrowedView::getComponent(std::uint32_t component) const {
    return data_.composition_[component];
  }

  CORSIKA_HOST_DEVICE inline bool EnvironmentBorrowedView::hasModelStorage() const {
    return data_.models_ != nullptr;
  }

} // namespace corsika::kokkos
