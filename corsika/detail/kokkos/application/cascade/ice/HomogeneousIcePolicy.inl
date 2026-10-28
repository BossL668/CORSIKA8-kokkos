/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application::ice {
  template <typename TSnapshot>
  double require_homogeneous_ice(TSnapshot const& snapshot) {
    if (snapshot.getModels().size() != 1) {
      throw std::invalid_argument("homogeneous ice requires one material model");
    }
    auto const& model = snapshot.getModels().front();
    if (model.mediumId_ != static_cast<int>(Medium::WaterIce) ||
        model.densityKind_ != EnvironmentModel::Density::Homogeneous ||
        model.refractivityKind_ != EnvironmentModel::Refractivity::Uniform) {
      throw std::invalid_argument("ice application requires homogeneous water ice");
    }
    return model.refractivityValue_;
  }
} // namespace corsika::kokkos::application::ice
