/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace corsika::kokkos {

  namespace observation_plane_detail {

    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double absolute(double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    CORSIKA_HOST_DEVICE inline double dot(double const left[3], double const right[3]) {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    CORSIKA_HOST_DEVICE inline bool valid_observation_plane(
        ObservationPlane const& plane) {
      auto normalSquared = 0.;
      for (int axis = 0; axis < 3; ++axis) {
        if (!finite(plane.pointM_[axis]) || !finite(plane.normal_[axis])) {
          return false;
        }
        normalSquared += plane.normal_[axis] * plane.normal_[axis];
      }
      return finite(normalSquared) && absolute(normalSquared - 1.) <= 1.e-12;
    }

  } // namespace observation_plane_detail

  CORSIKA_HOST_DEVICE inline ObservationPlaneIntersection
  intersect_observation_plane_straight(ObservationPlane const& plane,
                                       double const (&positionM)[3],
                                       double const (&direction)[3],
                                       double maximumDistanceM) {
    using namespace observation_plane_detail;
    if (!valid_observation_plane(plane) ||
        (!finite(maximumDistanceM) && maximumDistanceM != infinity())) {
      return {};
    }
    double displacement[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(positionM[axis]) || !finite(direction[axis])) { return {}; }
      displacement[axis] = plane.pointM_[axis] - positionM[axis];
    }
    auto const denominator = dot(plane.normal_, direction);
    if (denominator == 0.) {
      return {ObservationPlaneStatus::NoForwardIntersection, 0, infinity(), 0.};
    }
    auto const distance = dot(plane.normal_, displacement) / denominator;
    if (!finite(distance)) {
      return {ObservationPlaneStatus::NonFiniteResult, 0, 0., 0.};
    }
    if (!(distance > 0.) || (finite(maximumDistanceM) && distance > maximumDistanceM)) {
      return {ObservationPlaneStatus::NoForwardIntersection, 0, infinity(), 0.};
    }
    return {ObservationPlaneStatus::Success, 0, distance, 0.};
  }

} // namespace corsika::kokkos
