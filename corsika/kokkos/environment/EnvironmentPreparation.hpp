/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/Plane.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshot.hpp>
#include <corsika/kokkos/environment/ObservationPlane.hpp>
#include <memory>

namespace corsika::kokkos {
  /** Export the caller's tree, models and materials without choosing a scene.
   * Unsupported exact model/shape exports throw before a snapshot is published.
   * Material-table compatibility remains a separate session-level requirement.
   */
  template <typename TEnvironment>
  auto prepare_environment_snapshot(TEnvironment const&, CoordinateSystemPtr const&);

  /** Convert the actual observation plane to the snapshot's common coordinates.
   * The plane is independent of the environment tree. No horizontal-plane or
   * Earth-radius assumption is made, and the supplied normal is not normalized.
   */
  ObservationPlane export_observation_plane(Plane const&, CoordinateSystemPtr const&);
} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/EnvironmentPreparation.inl>
