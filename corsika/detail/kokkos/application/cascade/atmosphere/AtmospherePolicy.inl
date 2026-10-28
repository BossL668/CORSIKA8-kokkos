/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::application::atmosphere {
  template <typename TSnapshot>
  void require_air_materials(TSnapshot const& snapshot) {
    for (auto const& model : snapshot.getModels()) {
      if (model.mediumId_ != static_cast<int>(Medium::AirDry1Atm)) {
        throw std::invalid_argument(
            "full air-shower snapshot contains a medium not covered by this PROPOSAL "
            "session");
      }
    }
  }

  inline ObservationPlane horizontal_observation_plane(double x, double y,
                                                       double height) {
    return {{x, y, height}, {0., 0., 1.}};
  }
} // namespace corsika::kokkos::application::atmosphere
