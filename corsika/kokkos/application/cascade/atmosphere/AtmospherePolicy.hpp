/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/MediumProperties.hpp>
#include <corsika/kokkos/environment/ObservationPlane.hpp>
#include <stdexcept>

namespace corsika::kokkos::application::atmosphere {
  /** Retain the air application's validated material policy, not a generic limit. */
  template <typename TSnapshot>
  void require_air_materials(TSnapshot const&);

  /** Compatibility for air callers which supply a horizontal observation height. */
  ObservationPlane horizontal_observation_plane(double x, double y, double height);
} // namespace corsika::kokkos::application::atmosphere

#include <corsika/detail/kokkos/application/cascade/atmosphere/AtmospherePolicy.inl>
