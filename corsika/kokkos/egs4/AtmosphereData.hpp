/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos::egs4 {
  enum class DensityModel { Exponential, Homogeneous };
  enum class EnvironmentGeometry { SphericalLayers };

  /** Coefficients consumed by the C7 EGS4 atmospheric stepping equations.
   * Exponential density is a * exp((r - b) / c), in g/cm^3 and metres.
   * This is a restricted physics adapter, not the generic C8 geometry snapshot.
   */
  struct AtmosphereLayer {
    double innerRadiusM_{};
    double outerRadiusM_{};
    double densityParameterA_{};
    double densityParameterB_{};
    double densityParameterC_{};
    std::int32_t mediumId_{};
    DensityModel densityModel_{DensityModel::Exponential};
  };

  /** Five-layer AIR-NTP input for the validated C7 EGS4 atmosphere convention.
   * The application adapter must validate and export the actual C8 nodes.
   * Other geometries and media must not be silently treated as AIR-NTP.
   */
  struct AtmosphereData {
    double earthCenterM_[3]{};
    AtmosphereLayer atmosphereLayers_[5]{};
    std::uint32_t numberOfLayers_{};
    double magneticFieldT_[3]{};
    double observationPlanePointM_[3]{};
    double observationPlaneNormal_[3]{};
    double maximumMagneticDeflectionRad_{0.2};
    EnvironmentGeometry geometry_{EnvironmentGeometry::SphericalLayers};
  };
} // namespace corsika::kokkos::egs4
