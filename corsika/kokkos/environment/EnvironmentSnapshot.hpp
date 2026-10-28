/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/kokkos/environment/EnvironmentComponent.hpp>
#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>
#include <corsika/kokkos/environment/EnvironmentTopology.hpp>

#include <vector>

namespace corsika::kokkos {

  /**
   * Host-owned assembly of topology, shapes and exact native model parameters.
   *
   * Each node has one shape at the same index in getShapes(). A nonnegative
   * EnvironmentNode::modelIndex_ addresses getModels(); -1 means no model, with
   * no automatic inheritance from the parent. Model indices follow first
   * occurrence in native preorder. Nodes sharing the same property object share
   * one model and composition range; equal-valued distinct objects stay distinct.
   * Child and exclusion ordering, including repeated references, is preserved.
   *
   * Numeric records are independent copies in one caller-selected coordinate
   * system. The topology still borrows native node/model pointers: keep the
   * source tree and model assignments alive and unchanged while using those
   * mappings. Copies retain the same borrowed source and shared coordinate
   * system. Rebuild after changing geometry, models or their coordinate systems.
   * Only the numeric arrays, not this host object or its pointers, may be copied
   * to a device. Their layout is not a portable serialization format.
   *
   * Performs no containment search, density evaluation, boundary intersection
   * or observation-surface construction. Unsupported inputs fail explicitly;
   * nodes and models are never silently omitted or replaced by approximations.
   */
  template <typename TModelProperties>
  class EnvironmentSnapshot {
  public:
    using topology_type = EnvironmentTopology<TModelProperties>;
    using node_type = typename topology_type::node_type;

    /**
     * Assemble the complete rooted environment without modifying native objects.
     *
     * Uses export_shape() and export_model() with their existing support and
     * unit/frame conversion contracts. Construction either succeeds completely
     * or throws; no partial snapshot or component array is published.
     *
     * \param root Complete ownership-tree root, normally the C8 universe.
     * \param coordinateSystem Common coordinate system retained by this object.
     * \throws std::invalid_argument For a null coordinate system, a node without
     * a volume, or unsupported topology, shapes or model export interfaces.
     * \throws std::length_error If a topology or composition index range is exceeded.
     * \note The model type must meet the export_model() requirements even when
     * individual nodes have no model properties. No Kokkos runtime is required.
     */
    EnvironmentSnapshot(node_type const& root,
                        CoordinateSystemPtr const& coordinateSystem);

    topology_type const& getTopology() const { return topology_; }
    std::vector<EnvironmentShape> const& getShapes() const { return shapes_; }
    std::vector<EnvironmentModel> const& getModels() const { return models_; }
    std::vector<EnvironmentComponent> const& getComposition() const {
      return composition_;
    }
    CoordinateSystemPtr const& getCoordinateSystem() const { return coordinateSystem_; }

  private:
    topology_type topology_;
    CoordinateSystemPtr coordinateSystem_;
    std::vector<EnvironmentShape> shapes_;
    std::vector<EnvironmentModel> models_;
    std::vector<EnvironmentComponent> composition_;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentSnapshot.inl>
