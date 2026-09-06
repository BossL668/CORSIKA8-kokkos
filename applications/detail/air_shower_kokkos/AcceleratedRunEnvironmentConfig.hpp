/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <string>

namespace corsika::applications::air_shower {

  /** Device-neutral run metadata supplied by the air-shower application. */
  struct AcceleratedRunEnvironmentConfig {
    std::string geomagnetic_model;
    double geomagnetic_year{};
    double latitude_deg{};
    double longitude_deg{};
    double altitude_m{};
    std::array<double, 3> magnetic_field_T{};
    double maximum_magnetic_deflection_rad{};
    std::array<double, 3> observation_plane_point_m{};
    std::string antenna_file;
  };

  // Source compatibility for the already published native-CUDA application
  // wrapper.  New portable code uses the device-neutral spelling above.
  using CudaRunEnvironmentConfig = AcceleratedRunEnvironmentConfig;

} // namespace corsika::applications::air_shower
