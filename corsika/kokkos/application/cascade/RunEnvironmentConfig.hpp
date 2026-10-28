/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <string>
#include <corsika/kokkos/application/cascade/CascadeScene.hpp>

namespace corsika::kokkos::application {

  /** Device-neutral run metadata supplied by the calling application. */
  struct RunEnvironmentConfig {
    std::string geomagneticModel_;
    double geomagneticYear_{};
    double latitudeDeg_{};
    double longitudeDeg_{};
    double altitudeM_{};
    std::array<double, 3> magneticFieldT_{};
    double maximumMagneticDeflectionRad_{};
    std::array<double, 3> observationPlanePointM_{};
    std::string antennaFile_;
    CascadeScene scene_{CascadeScene::Atmosphere};
  };

  // Legacy configuration alias, using the receiver alias naming convention.
  using cuda_run_environment_config_type = RunEnvironmentConfig;

} // namespace corsika::kokkos::application
