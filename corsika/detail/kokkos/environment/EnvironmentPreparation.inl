/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <typename TEnvironment>
  auto prepare_environment_snapshot(TEnvironment const& environment,
                                    CoordinateSystemPtr const& coordinates) {
    using model_type = typename TEnvironment::BaseNodeType::IModelProperties;
    return std::make_shared<EnvironmentSnapshot<model_type>>(*environment.getUniverse(),
                                                             coordinates);
  }

  inline ObservationPlane export_observation_plane(
      Plane const& plane, CoordinateSystemPtr const& coordinates) {
    auto const point = plane.getCenter().getCoordinates(coordinates);
    auto const normal = plane.getNormal().getComponents(coordinates);
    ObservationPlane result;
    for (int axis = 0; axis < 3; ++axis) {
      result.pointM_[axis] = point[axis] / 1_m;
      result.normal_[axis] = normal[axis].magnitude();
    }
    if (!observation_plane_detail::valid_observation_plane(result)) {
      throw std::invalid_argument("invalid exported observation plane");
    }
    return result;
  }
} // namespace corsika::kokkos
