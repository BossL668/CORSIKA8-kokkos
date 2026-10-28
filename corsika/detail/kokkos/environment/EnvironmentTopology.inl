/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace corsika::kokkos {

  template <typename TModelProperties>
  EnvironmentTopology<TModelProperties>::EnvironmentTopology(node_type const& root) {
    if (root.getParent() != nullptr) {
      throw std::invalid_argument("Environment topology requires the complete root");
    }
    // Reverse-push children to preserve C8 preorder without recursive stack growth.
    std::vector<node_type const*> pending{&root};
    while (!pending.empty()) {
      auto const* node = pending.back();
      pending.pop_back();
      if (nativeNodes_.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("Environment node index range exceeded");
      }
      auto const index = static_cast<std::int32_t>(nativeNodes_.size());
      if (!nodeIndices_.emplace(node, index).second) {
        throw std::invalid_argument("Repeated node in environment ownership tree");
      }
      nativeNodes_.push_back(node);
      auto const& children = node->getChildNodes();
      for (auto child = children.rbegin(); child != children.rend(); ++child) {
        if (!*child || (*child)->getParent() != node) {
          throw std::invalid_argument("Invalid environment child ownership");
        }
        pending.push_back(child->get());
      }
    }

    nodes_.resize(nativeNodes_.size());
    for (std::size_t index = 0; index < nativeNodes_.size(); ++index) {
      auto const& native = *nativeNodes_[index];
      auto& node = nodes_[index];
      node.parentIndex_ = native.getParent() ? getNodeIndex(*native.getParent()) : -1;
      auto const& children = native.getChildNodes();
      auto const& exclusions = native.getExcludedNodes();
      auto const maxEdges = std::numeric_limits<std::uint32_t>::max();
      if (children.size() > maxEdges - children_.size() ||
          exclusions.size() > maxEdges - exclusions_.size()) {
        throw std::length_error("Environment adjacency index range exceeded");
      }
      node.childOffset_ = static_cast<std::uint32_t>(children_.size());
      node.childCount_ = static_cast<std::uint32_t>(children.size());
      for (auto const& child : children) children_.push_back(getNodeIndex(*child));
      node.excludedOffset_ = static_cast<std::uint32_t>(exclusions_.size());
      node.excludedCount_ = static_cast<std::uint32_t>(exclusions.size());
      for (auto const* excluded : exclusions) {
        auto const found = nodeIndices_.find(excluded);
        if (found == nodeIndices_.end()) {
          throw std::invalid_argument("Null or external environment exclusion");
        }
        exclusions_.push_back(found->second);
      }
      if (native.hasModelProperties()) {
        auto const* model = std::addressof(native.getModelProperties());
        if (modelIndices_.find(model) == modelIndices_.end()) {
          // At most one new model per node, so the checked node range bounds this.
          auto const modelIndex = static_cast<std::int32_t>(nativeModels_.size());
          modelIndices_.emplace(model, modelIndex);
          nativeModels_.push_back(model);
        }
        node.modelIndex_ = getModelIndex(*model);
      }
    }
  }

  template <typename TModelProperties>
  std::int32_t EnvironmentTopology<TModelProperties>::getNodeIndex(
      node_reference_type node) const {
    return node ? nodeIndices_.at(std::addressof(*node)) : -1;
  }

  template <typename TModelProperties>
  typename EnvironmentTopology<TModelProperties>::node_reference_type
  EnvironmentTopology<TModelProperties>::getNativeNode(std::int32_t const index) const {
    if (index == -1) return boost::none;
    return *nativeNodes_.at(static_cast<std::size_t>(index));
  }

  template <typename TModelProperties>
  std::int32_t EnvironmentTopology<TModelProperties>::getModelIndex(
      model_reference_type model) const {
    return model ? modelIndices_.at(std::addressof(*model)) : -1;
  }

  template <typename TModelProperties>
  typename EnvironmentTopology<TModelProperties>::model_reference_type
  EnvironmentTopology<TModelProperties>::getNativeModel(std::int32_t const index) const {
    if (index == -1) return boost::none;
    return *nativeModels_.at(static_cast<std::size_t>(index));
  }

} // namespace corsika::kokkos
