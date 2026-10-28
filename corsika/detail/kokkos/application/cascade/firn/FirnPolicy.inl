/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application::firn {
  template <typename TSnapshot>
  void require_firn(TSnapshot const& snapshot) {
    if (snapshot.getModels().size() != 1) {
      throw std::invalid_argument("firn application requires one water-ice model");
    }
    auto const& model = snapshot.getModels().front();
    if (model.mediumId_ != static_cast<int>(Medium::WaterIce) ||
        model.densityKind_ != EnvironmentModel::Density::ExponentialFirn ||
        model.refractivityKind_ != EnvironmentModel::Refractivity::ExponentialFirn) {
      throw std::invalid_argument("firn requires exported density and refractive index");
    }
  }
} // namespace corsika::kokkos::application::firn
