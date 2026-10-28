/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/ModelConversion.hpp>
#include <corsika/kokkos/environment/ShapeConversion.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace corsika::kokkos {

  template <typename TModelProperties>
  EnvironmentSnapshot<TModelProperties>::EnvironmentSnapshot(
      node_type const& root, CoordinateSystemPtr const& coordinateSystem)
      : topology_(root)
      , coordinateSystem_(coordinateSystem) {
    if (!coordinateSystem_) {
      throw std::invalid_argument("Environment snapshot requires a coordinate system");
    }
    auto const& nodes = topology_.getNodes();
    shapes_.reserve(nodes.size());
    for (std::size_t index = 0; index < nodes.size(); ++index) {
      auto const& native = *topology_.getNativeNode(static_cast<std::int32_t>(index));
      if (!native.hasVolume()) {
        throw std::invalid_argument(
            "Environment snapshot requires a volume on every node");
      }
      shapes_.push_back(export_shape(native.getVolume(), coordinateSystem_));
      auto const modelIndex = nodes[index].modelIndex_;
      // Topology assigns consecutive model indices in the same preorder. Export
      // only the first occurrence, using pointer identity rather than values.
      if (modelIndex >= 0 && static_cast<std::size_t>(modelIndex) == models_.size()) {
        models_.push_back(export_model(*topology_.getNativeModel(modelIndex),
                                       coordinateSystem_, composition_));
      }
    }
  }

} // namespace corsika::kokkos
