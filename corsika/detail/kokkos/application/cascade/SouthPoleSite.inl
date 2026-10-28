/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/framework/core/PhysicalConstants.hpp>
#include <cmath>
#include <stdexcept>

namespace corsika::kokkos::application {
  inline CoordinateSystemPtr SouthPoleSite::coordinates(
      CoordinateSystemPtr const& earthCoordinates) const {
    return make_rotation(earthCoordinates, QuantityVector<dimensionless_d>{1., 0., 0.},
                         std::acos(-1.));
  }

  inline Point SouthPoleSite::surface(CoordinateSystemPtr const& polarCoordinates) const {
    if (!std::isfinite(surfaceAltitudeM_)) {
      throw std::invalid_argument("South Pole surface altitude must be finite");
    }
    return Point(polarCoordinates, 0_m, 0_m,
                 constants::EarthRadius::Mean + surfaceAltitudeM_ * 1_m);
  }
} // namespace corsika::kokkos::application
