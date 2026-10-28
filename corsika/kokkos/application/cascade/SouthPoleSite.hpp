/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/framework/geometry/Point.hpp>

namespace corsika::kokkos::application {
  /** Shared ice/firn reference site on C8's spherical Earth.
   * At the geographic pole, longitude zero fixes the horizontal-axis convention:
   * north along global +x, west along -y and up along -z. Heights are measured
   * above C8's mean Earth radius; the default surface elevation is 2835 m.
   * This does not prescribe an ice profile or a geomagnetic model.
   */
  struct SouthPoleSite {
    static constexpr double latitudeDeg = -90.;
    static constexpr double longitudeDeg = 0.;
    double surfaceAltitudeM_ = 2835.;

    /// Earth-centred NWU axes at the pole; both positions and vectors use this frame.
    CoordinateSystemPtr coordinates(CoordinateSystemPtr const& earthCoordinates) const;
    /// Surface point in the polar axes, at the configured elevation.
    Point surface(CoordinateSystemPtr const& polarCoordinates) const;
  };
} // namespace corsika::kokkos::application

#include <corsika/detail/kokkos/application/cascade/SouthPoleSite.inl>
