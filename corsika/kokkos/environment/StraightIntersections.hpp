/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentShape.hpp>
#include <corsika/kokkos/environment/GeometryIntersections.hpp>

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  /// Preserve the straight-query result name for existing callers.
  using straight_intersections_type = GeometryIntersections;

  /**
   * Intersect an exported shape with a straight parametrized trajectory.
   *
   * \param shape A supported shape exported in the common coordinate system.
   * \param positionM Starting position in metres in that same system.
   * \param velocity Numerical trajectory tangent. Using metres per second gives
   *   time endpoints; using a unit direction gives distance endpoints in metres.
   * \return The production straight-intersection result for Universe, Sphere,
   *   SeparationPlane or Box. Tangencies and parallel planes retain the original
   *   conventions; no new tolerances, padding or root solver are introduced.
   *
   * A valid exported shape is required. Physical inputs are finite; exceptional
   * IEEE behavior retains production arithmetic, not an extra repair policy.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline straight_intersections_type straight_intersect(
      EnvironmentShape const& shape, double const (&positionM)[3],
      double const (&velocity)[3]);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/StraightIntersections.inl>
