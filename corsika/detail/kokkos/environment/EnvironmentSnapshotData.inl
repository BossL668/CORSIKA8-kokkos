/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TModelProperties>
  EnvironmentDataView bind_environment_data(
      EnvironmentSnapshot<TModelProperties> const& snapshot) {
    auto const& topology = snapshot.getTopology();
    return {topology.getNodes().data(),
            snapshot.getShapes().data(),
            topology.getChildren().data(),
            topology.getExclusions().data(),
            snapshot.getModels().data(),
            snapshot.getComposition().data(),
            static_cast<std::uint32_t>(topology.getNodes().size()),
            static_cast<std::uint32_t>(topology.getChildren().size()),
            static_cast<std::uint32_t>(topology.getExclusions().size()),
            static_cast<std::uint32_t>(snapshot.getModels().size()),
            static_cast<std::uint32_t>(snapshot.getComposition().size())};
  }

} // namespace corsika::kokkos
