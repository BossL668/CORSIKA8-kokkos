/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/common/Types.hpp>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif
#define CORSIKA_GPU_OBSERVATION_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  enum class ObservationPlaneStatus : std::uint32_t {
    Success = 0,
    NoForwardIntersection = 1,
    InvalidInput = 2,
    NonFiniteResult = 3,
  };

  struct ObservationPlaneIntersection {
    ObservationPlaneStatus status{
        ObservationPlaneStatus::InvalidInput};
    std::uint32_t reserved{};
    double distance_m{};
    double residual_m{};
  };

  static_assert(std::is_standard_layout_v<ObservationPlaneIntersection>);
  static_assert(std::is_trivially_copyable_v<ObservationPlaneIntersection>);

  namespace observation_plane_detail {

    CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline double absolute(double value) {
#if defined(__CUDA_ARCH__)
      return ::fabs(value);
#else
      return std::abs(value);
#endif
    }

    CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline double squareRoot(double value) {
#if defined(__CUDA_ARCH__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }

    CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline double dot(
        double const left[3], double const right[3]) {
      return left[0] * right[0] + left[1] * right[1] +
             left[2] * right[2];
    }

    CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline bool validObservationPlane(
        EnvironmentSnapshot const& environment) {
      auto normal_squared = 0.;
      for (int axis = 0; axis < 3; ++axis) {
        if (!finite(environment.observation_plane_point_m[axis]) ||
            !finite(environment.observation_plane_normal[axis])) {
          return false;
        }
        normal_squared +=
            environment.observation_plane_normal[axis] *
            environment.observation_plane_normal[axis];
      }
      return finite(normal_squared) &&
             absolute(normal_squared - 1.) <= 1.e-12;
    }

  } // namespace observation_plane_detail

  /**
   * Return the same forward straight-line intersection used by
   * TrackingStraight::intersect(particle, Plane).  The distance is the spatial
   * trajectory parameter; photons have beta=1 and charged transport converts
   * it to time only after the winning limit has been selected.
   */
  CORSIKA_GPU_OBSERVATION_HOST_DEVICE inline ObservationPlaneIntersection
  intersectObservationPlaneStraight(
      EnvironmentSnapshot const& environment, double const position_m[3],
      double const direction[3], double maximum_distance_m =
          observation_plane_detail::infinity()) {
    using namespace observation_plane_detail;
    if (!validObservationPlane(environment) ||
        (!finite(maximum_distance_m) &&
         maximum_distance_m != infinity())) {
      return {};
    }
    if (environment.geometry ==
        EnvironmentGeometry::HomogeneousConvexPolyhedron) {
      return {ObservationPlaneStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }
    double displacement[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!finite(position_m[axis]) || !finite(direction[axis])) {
        return {};
      }
      displacement[axis] =
          environment.observation_plane_point_m[axis] - position_m[axis];
    }
    auto const denominator =
        dot(environment.observation_plane_normal, direction);
    if (denominator == 0.) {
      return {ObservationPlaneStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }
    auto const distance =
        dot(environment.observation_plane_normal, displacement) /
        denominator;
    if (!finite(distance)) {
      return {ObservationPlaneStatus::NonFiniteResult, 0, 0., 0.};
    }
    if (!(distance > 0.) ||
        (finite(maximum_distance_m) &&
         distance > maximum_distance_m)) {
      return {ObservationPlaneStatus::NoForwardIntersection, 0,
              infinity(), 0.};
    }
    return {ObservationPlaneStatus::Success, 0, distance, 0.};
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_OBSERVATION_HOST_DEVICE
