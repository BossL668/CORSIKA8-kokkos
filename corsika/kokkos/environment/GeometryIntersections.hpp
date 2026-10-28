/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  /**
   * Numerical entry and exit parameters from a production shape intersection.
   *
   * A default result has no intersections and zero-valued endpoints, matching
   * the production numerical record. Check hasIntersections() before using them.
   * Values have no embedded units; their parameter follows the calling
   * trajectory calculation. No dynamic storage is owned, and no endpoints are
   * reordered, clamped or validated here.
   */
  class GeometryIntersections {
  public:
    /// Construct the production no-intersection result with zero endpoints.
    CORSIKA_HOST_DEVICE inline GeometryIntersections();

    /// Record a valid pair of endpoints, without reordering or clamping.
    CORSIKA_HOST_DEVICE inline GeometryIntersections(double entry, double exit);

    /// Whether the shape calculation produced an intersection pair.
    [[nodiscard]] CORSIKA_HOST_DEVICE inline bool hasIntersections() const;

    /// Entry parameter, including negative or infinite production results.
    [[nodiscard]] CORSIKA_HOST_DEVICE inline double getEntry() const;

    /// Exit parameter; a plane uses positive infinity regardless of orientation.
    [[nodiscard]] CORSIKA_HOST_DEVICE inline double getExit() const;

  private:
    bool valid_{};
    double entry_{};
    double exit_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/GeometryIntersections.inl>
