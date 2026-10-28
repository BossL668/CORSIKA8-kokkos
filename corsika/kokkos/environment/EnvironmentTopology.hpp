/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/media/VolumeTreeNode.hpp>

#include <boost/optional.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace corsika::kokkos {

  /**
   * Host-only export of C8's ownership tree and ordered exclusion references.
   *
   * Owns the numeric arrays, but borrows the native node and model pointers.
   * The source tree and its model assignments must remain alive and unchanged
   * while this mapping is used; rebuild it after modifying the environment.
   * Indices are deterministic preorder indices within this snapshot, not IDs
   * that survive edits of the source tree. Copies retain the same borrowed
   * source. Only the numeric arrays, never this object, may be copied to a GPU.
   *
   * All nodes are retained, including nodes without model properties. Distinct
   * nodes sharing a property object share a model index but keep separate node
   * indices. Equal-valued but distinct models are not merged. Exclusion edges
   * remain directed and ordered, including repetitions or reciprocal edges.
   * No containment search or geometrical validity check is performed here.
   */
  template <typename TModelProperties>
  class EnvironmentTopology {
  public:
    using node_type = VolumeTreeNode<TModelProperties>;
    using node_reference_type = boost::optional<node_type const&>;
    using model_reference_type = boost::optional<TModelProperties const&>;

    /**
     * Export a complete rooted ownership tree without modifying it.
     *
     * \param root Root with no parent (normally the C8 universe).
     * \throws std::invalid_argument For a non-root input, inconsistent child
     * ownership, or null/external exclusion references. No partial snapshot
     * is returned. Excluded nodes must belong to this same ownership tree.
     * \throws std::length_error If the fixed-width index range is exceeded.
     * \note Does not inspect volumes or model parameters; support for a shape
     * or a physical model is not implied by successful topology export.
     */
    explicit EnvironmentTopology(node_type const& root);

    /** Exported node records, in deterministic ownership-tree order. */
    std::vector<EnvironmentNode> const& getNodes() const { return nodes_; }
    /** Child indices grouped by each node's offset and count. */
    std::vector<std::int32_t> const& getChildren() const { return children_; }
    /** Ordered exclusion indices, preserving directed native references. */
    std::vector<std::int32_t> const& getExclusions() const { return exclusions_; }

    /** Borrowed node to index; an absent reference maps to -1 (outside).
     * \throws std::out_of_range If the referenced node is not in this snapshot.
     */
    std::int32_t getNodeIndex(node_reference_type node) const;

    /** Node index to a borrowed reference; only -1 yields no reference.
     * \throws std::out_of_range For every other invalid index, including the
     * unassigned ParticleState sentinel -3. Never silently select the root.
     */
    node_reference_type getNativeNode(std::int32_t index) const;

    /** Borrowed model to index; an absent reference maps to -1.
     * \throws std::out_of_range If the referenced properties were not exported.
     */
    std::int32_t getModelIndex(model_reference_type model) const;

    /** Model index to a borrowed reference; only -1 yields no reference.
     * \throws std::out_of_range For every other invalid index.
     */
    model_reference_type getNativeModel(std::int32_t index) const;

  private:
    std::vector<EnvironmentNode> nodes_;
    std::vector<std::int32_t> children_;
    std::vector<std::int32_t> exclusions_;
    std::vector<node_type const*> nativeNodes_;
    std::vector<TModelProperties const*> nativeModels_;
    std::unordered_map<node_type const*, std::int32_t> nodeIndices_;
    std::unordered_map<TModelProperties const*, std::int32_t> modelIndices_;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentTopology.inl>
