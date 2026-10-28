/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /**
   * Independent observation plane in the exported geometry's common system.
   *
   * pointM_ is in metres and normal_ is dimensionless. The normal is expected
   * to be unit length and is checked, not normalized, by the query. This is not
   * an environment node and does not carry or inherit material properties.
   */
  struct ObservationPlane {
    double pointM_[3]{};
    double normal_[3]{};
  };

  enum class ObservationPlaneStatus : std::uint32_t {
    Success = 0,
    NoForwardIntersection = 1,
    InvalidInput = 2,
    NonFiniteResult = 3
  };

  /// Production straight-plane result; residualM_ remains zero on this path.
  struct ObservationPlaneIntersection {
    ObservationPlaneStatus status_{ObservationPlaneStatus::InvalidInput};
    std::uint32_t reserved_{};
    double distanceM_{};
    double residualM_{};
  };

  static_assert(std::is_standard_layout_v<ObservationPlaneIntersection>);
  static_assert(std::is_trivially_copyable_v<ObservationPlaneIntersection>);

  namespace observation_plane_detail {
    CORSIKA_HOST_DEVICE inline double infinity();
  }

  /**
   * Intersect a straight trajectory with the independent observation plane.
   *
   * \param plane Plane in the same common coordinate system as the trajectory.
   * \param positionM Starting position in metres.
   * \param direction Finite dimensionless direction. Unit length makes the
   * returned parameter a physical distance; no normalization check is added.
   * \param maximumDistanceM Accepted upper limit, finite or positive infinity.
   * \return Original status and parameter: a strictly positive intersection
   * at the upper limit is included, while parallel or backward rays have none.
   *
   * This is the exported node-tree observation path. Legacy non-tree options
   * that suppress observation are not represented here and must be handled
   * explicitly by their caller. The function does not compare environment
   * boundaries, advance a particle, absorb it, or change its logical node.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline ObservationPlaneIntersection
  intersect_observation_plane_straight(
      ObservationPlane const& plane, double const (&positionM)[3],
      double const (&direction)[3],
      double maximumDistanceM = observation_plane_detail::infinity());

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/environment/ObservationPlane.inl>
