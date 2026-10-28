/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "HostEnvironmentTree.hpp"

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>

#include <Kokkos_Core.hpp>

namespace environment_test {
  namespace {
    template <typename TValue>
    auto host_view(std::vector<TValue> const& values) {
      Kokkos::View<TValue*, Kokkos::HostSpace> result("native_comparison", values.size());
      for (std::size_t index = 0; index < values.size(); ++index) {
        result(index) = values[index];
      }
      return result;
    }
  } // namespace

  struct HostEnvironmentTree::State {
    corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace> tree_;
  };

  HostEnvironmentTree::HostEnvironmentTree()
      : state_(std::make_unique<State>()) {}

  HostEnvironmentTree::HostEnvironmentTree(
      std::vector<corsika::kokkos::EnvironmentNode> const& nodes,
      std::vector<corsika::kokkos::EnvironmentShape> const& shapes,
      std::vector<node_index_type> const& children,
      std::vector<node_index_type> const& exclusions)
      : state_(std::make_unique<State>(
            State{{host_view(nodes), host_view(shapes), host_view(children),
                   host_view(exclusions)}})) {}

  HostEnvironmentTree::HostEnvironmentTree(HostEnvironmentTree const& other)
      : state_(std::make_unique<State>(*other.state_)) {}

  HostEnvironmentTree::~HostEnvironmentTree() = default;

  std::uint32_t HostEnvironmentTree::getNodeCount() const {
    return state_->tree_.getNodeCount();
  }
  auto HostEnvironmentTree::getParent(node_index_type node) const -> node_index_type {
    return state_->tree_.getParent(node);
  }
  corsika::kokkos::EnvironmentShape const& HostEnvironmentTree::getShape(
      node_index_type node) const {
    return state_->tree_.getShape(node);
  }
  std::uint32_t HostEnvironmentTree::getChildCount(node_index_type node) const {
    return state_->tree_.getChildCount(node);
  }
  auto HostEnvironmentTree::getChild(node_index_type node, std::uint32_t child) const
      -> node_index_type {
    return state_->tree_.getChild(node, child);
  }
  std::uint32_t HostEnvironmentTree::getExcludedCount(node_index_type node) const {
    return state_->tree_.getExcludedCount(node);
  }
  auto HostEnvironmentTree::getExcluded(node_index_type node,
                                        std::uint32_t excluded) const -> node_index_type {
    return state_->tree_.getExcluded(node, excluded);
  }
  bool is_initialized() { return Kokkos::is_initialized(); }
  bool is_finalized() { return Kokkos::is_finalized(); }
} // namespace environment_test
