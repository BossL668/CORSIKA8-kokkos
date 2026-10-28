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

  namespace straight_intersection_detail {

    CORSIKA_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

    // Preserve local CUDA rounding, without changing global physics flags.
    CORSIKA_HOST_DEVICE inline double product(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dmul_rn(first, second);
#else
      return first * second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double sum(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dadd_rn(first, second);
#else
      return first + second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double difference(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dsub_rn(first, second);
#else
      return first - second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double dot(double const (&first)[3],
                                          double const (&second)[3]) {
      return sum(sum(product(first[0], second[0]), product(first[1], second[1])),
                 product(first[2], second[2]));
    }

  } // namespace straight_intersection_detail

  CORSIKA_HOST_DEVICE inline straight_intersections_type straight_intersect(
      EnvironmentShape const& shape, double const (&positionM)[3],
      double const (&velocity)[3]) {
    using namespace straight_intersection_detail;
    double const inf = infinity();
    if (shape.kind_ == EnvironmentShape::Kind::Universe) { return {-inf, inf}; }
    double delta[3]{positionM[0] - shape.centerM_[0], positionM[1] - shape.centerM_[1],
                    positionM[2] - shape.centerM_[2]};
    if (shape.kind_ == EnvironmentShape::Kind::Sphere) {
      auto const velocitySquared = dot(velocity, velocity);
      auto const velocityDelta = dot(velocity, delta);
      auto const discriminant = difference(
          product(velocityDelta, velocityDelta),
          product(velocitySquared, difference(dot(delta, delta),
                                              product(shape.radiusM_, shape.radiusM_))));
      if (!(discriminant > 0.)) { return {}; }
      double const root = ::sqrt(discriminant);
      return {(-velocityDelta - root) / velocitySquared,
              (-velocityDelta + root) / velocitySquared};
    }
    if (shape.kind_ == EnvironmentShape::Kind::SeparationPlane) {
      double const normalVelocity = dot(velocity, shape.normal_);
      if (normalVelocity == 0.) { return {}; }
      return {-dot(delta, shape.normal_) / normalVelocity, inf};
    }
    double entry = -inf;
    double exit = inf;
    for (int axis = 0; axis < 3; ++axis) {
      auto const position = dot(delta, shape.axes_[axis]);
      auto const speed = dot(velocity, shape.axes_[axis]);
      double const first = (shape.halfExtentM_[axis] - position) / speed;
      double const second = (-shape.halfExtentM_[axis] - position) / speed;
      // The operand order also preserves production NaN behavior at slab edges.
      auto const lower = first > second ? second : first;
      auto const upper = first > second ? first : second;
      if (axis == 0) {
        entry = lower;
        exit = upper;
      } else {
        entry = entry < lower ? lower : entry;
        exit = upper < exit ? upper : exit;
      }
    }
    return exit > entry ? straight_intersections_type{entry, exit}
                        : straight_intersections_type{};
  }

} // namespace corsika::kokkos
