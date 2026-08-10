/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/gpu/em/ObservationPlane.hpp>
#include <corsika/gpu/em/SphericalAtmosphere.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>

namespace corsika::gpu::em {

  /**
   * Build the exact device schema corresponding to create_5layer_atmosphere().
   *
   * The first four CORSIKA 8 layers use SlidingPlanarExponential with
   * rho0=offset/scaleHeight, lambda=-scaleHeight and the mean Earth radius as
   * reference radius. The fifth layer is HomogeneousMedium, despite its
   * historical CORSIKA-7 "linear" name.
   */
  inline EnvironmentSnapshot makeCorsika7AtmosphereSnapshot(
      AtmosphereId atmosphere_id,
      std::array<double, 3> const& earth_center_m = {0., 0., 0.},
      std::int32_t medium_id = 0,
      double observation_radius_m =
          constants::EarthRadius::Mean / 1_m,
      std::array<double, 3> const& magnetic_field_T = {0., 0., 0.},
      double maximum_magnetic_deflection_rad = 0.2) {
    auto const id = static_cast<std::uint8_t>(atmosphere_id);
    if (id >= static_cast<std::uint8_t>(
                  AtmosphereId::LastAtmosphere)) {
      throw std::invalid_argument("invalid CORSIKA-7 atmosphere id");
    }

    EnvironmentSnapshot snapshot{};
    snapshot.number_of_layers = MaxAtmosphereLayers;
    snapshot.observation_radius_m = observation_radius_m;
    snapshot.maximum_magnetic_deflection_rad =
        maximum_magnetic_deflection_rad;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      snapshot.earth_center_m[axis] = earth_center_m[axis];
      snapshot.magnetic_field_T[axis] = magnetic_field_T[axis];
      snapshot.observation_plane_point_m[axis] =
          earth_center_m[axis];
      snapshot.observation_plane_normal[axis] =
          axis == 2 ? 1. : 0.;
    }
    snapshot.observation_plane_point_m[2] +=
        observation_radius_m;

    auto const earth_radius_m =
        constants::EarthRadius::Mean / 1_m;
    auto const density_unit = 1_g / cube(1_cm);
    auto const& parameters = atmosphereParameterList[id];
    auto inner_radius_m = earth_radius_m;
    for (std::size_t index = 0; index < MaxAtmosphereLayers;
         ++index) {
      auto& layer = snapshot.atmosphere_layers[index];
      auto const& parameter = parameters[index];
      layer.inner_radius_m = inner_radius_m;
      layer.outer_radius_m =
          earth_radius_m + parameter.altitude / 1_m;
      layer.density_parameter_a =
          (parameter.offset / parameter.scaleHeight) /
          density_unit;
      layer.medium_id = medium_id;
      if (index + 1 < MaxAtmosphereLayers) {
        layer.density_model = DensityModel::Exponential;
        layer.density_parameter_b = earth_radius_m;
        layer.density_parameter_c =
            -parameter.scaleHeight / 1_m;
      } else {
        layer.density_model = DensityModel::Homogeneous;
      }
      inner_radius_m = layer.outer_radius_m;
    }

    if (!atmosphere_detail::validEnvironment(snapshot)) {
      throw std::invalid_argument(
          "CORSIKA-7 parameters produced an invalid GPU environment snapshot");
    }
    return snapshot;
  }

  inline void setObservationPlane(
      EnvironmentSnapshot& snapshot,
      std::array<double, 3> const& point_m,
      std::array<double, 3> const& normal) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      snapshot.observation_plane_point_m[axis] = point_m[axis];
      snapshot.observation_plane_normal[axis] = normal[axis];
    }
    if (!observation_plane_detail::validObservationPlane(snapshot)) {
      throw std::invalid_argument(
          "invalid GPU observation plane point or normal");
    }
  }

} // namespace corsika::gpu::em
