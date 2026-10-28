/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentNode.hpp>
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

#include "../../framework/geometry/NextEnvironmentBoundaryReference.hpp"

#include <corsika/framework/core/HostDevice.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace straight_boundary_reference {

  using shape_type = corsika::kokkos::EnvironmentShape;
  using node_type = corsika::kokkos::EnvironmentNode;
  using interval_type = boundary_reference::Interval<double>;

  CORSIKA_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
    return CUDART_INF;
#else
    return std::numeric_limits<double>::infinity();
#endif
  }

  // Independent expressions from EnvironmentTreeSnapshot.hpp at fixed production
  // 41cdc543. No new shape, arithmetic or boundary helpers enter this oracle.
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
  CORSIKA_HOST_DEVICE inline double dot(double const* first, double const* second) {
    return sum(sum(product(first[0], second[0]), product(first[1], second[1])),
               product(first[2], second[2]));
  }
  CORSIKA_HOST_DEVICE inline interval_type intersect(shape_type const& shape,
                                                     double const* position,
                                                     double const* velocity) {
    double const inf = infinity();
    if (shape.kind_ == shape_type::Kind::Universe) { return {true, -inf, inf}; }
    double delta[3]{position[0] - shape.centerM_[0], position[1] - shape.centerM_[1],
                    position[2] - shape.centerM_[2]};
    if (shape.kind_ == shape_type::Kind::Sphere) {
      auto const velocitySquared = dot(velocity, velocity);
      auto const velocityDelta = dot(velocity, delta);
      auto const discriminant = difference(
          product(velocityDelta, velocityDelta),
          product(velocitySquared, difference(dot(delta, delta),
                                              product(shape.radiusM_, shape.radiusM_))));
      if (!(discriminant > 0.)) { return {}; }
      double const root = ::sqrt(discriminant);
      return {true, (-velocityDelta - root) / velocitySquared,
              (-velocityDelta + root) / velocitySquared};
    }
    if (shape.kind_ == shape_type::Kind::SeparationPlane) {
      auto const normalVelocity = dot(velocity, shape.normal_);
      if (normalVelocity == 0.) { return {}; }
      return {true, -dot(delta, shape.normal_) / normalVelocity, inf};
    }
    double entry = -inf;
    double exit = inf;
    for (int axis = 0; axis < 3; ++axis) {
      auto const offset = dot(delta, shape.axes_[axis]);
      auto const speed = dot(velocity, shape.axes_[axis]);
      double const first = (shape.halfExtentM_[axis] - offset) / speed;
      double const second = (-shape.halfExtentM_[axis] - offset) / speed;
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
    return exit > entry ? interval_type{true, entry, exit} : interval_type{};
  }

  struct Query {
    shape_type shape_;
    double position_[3]{};
    double velocity_[3]{};
  };

  inline shape_type box(double extent) {
    shape_type result;
    result.kind_ = shape_type::Kind::Box;
    for (int axis = 0; axis < 3; ++axis) {
      result.axes_[axis][axis] = 1.;
      result.halfExtentM_[axis] = extent;
    }
    return result;
  }

  inline std::vector<Query> queries() {
    std::vector<Query> result;
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    std::array<std::array<double, 3>, 12> const positions{{{0., 0., 0.},
                                                           {3., 0., 0.},
                                                           {-3., 0., 0.},
                                                           {1., 0., 0.},
                                                           {0., 1., 0.},
                                                           {0., 0., -1.},
                                                           {.13, -.27, .41},
                                                           {1., 1., 1.},
                                                           {nan, 0., 0.},
                                                           {0., infinity, 0.},
                                                           {1.e150, 0., 0.},
                                                           {1.e-150, 0., 0.}}};
    std::array<std::array<double, 3>, 10> const velocities{{{1., 0., 0.},
                                                            {-1., 0., 0.},
                                                            {0., 1., 0.},
                                                            {0., 0., 1.},
                                                            {.17, -.31, .79},
                                                            {0., 0., 0.},
                                                            {1.e-150, 0., 0.},
                                                            {1.e150, 0., 0.},
                                                            {nan, 0., 0.},
                                                            {0., infinity, 0.}}};
    for (auto kind : {shape_type::Kind::Universe, shape_type::Kind::Sphere,
                      shape_type::Kind::Box, shape_type::Kind::SeparationPlane}) {
      auto shape = box(1.);
      shape.kind_ = kind;
      shape.radiusM_ = 1.;
      shape.normal_[2] = 1.;
      for (auto const& position : positions) {
        for (auto const& velocity : velocities) {
          Query query{shape};
          for (int axis = 0; axis < 3; ++axis) {
            query.position_[axis] = position[axis];
            query.velocity_[axis] = velocity[axis];
          }
          result.push_back(query);
        }
      }
    }
    // Ordinary oblique cases test CUDA dot rounding away from discriminant zero.
    for (int index = 0; index < 40; ++index) {
      auto shape = box(2.);
      shape.centerM_[0] = .13;
      shape.centerM_[1] = -.27;
      shape.centerM_[2] = .41;
      shape.axes_[0][0] = shape.axes_[1][1] = .6;
      shape.axes_[0][1] = .8;
      shape.axes_[1][0] = -.8;
      shape.normal_[0] = .2;
      shape.normal_[1] = .3;
      shape.normal_[2] = .4;
      shape.kind_ = index % 2 ? shape_type::Kind::Box : shape_type::Kind::SeparationPlane;
      result.push_back({shape, {-.317 + .011 * index, .173, -.691}, {.17, -.31, .79}});
    }
    return result;
  }

  struct Tree {
    node_type const* nodes_{};
    shape_type const* shapes_{};
    std::int32_t const* children_{};
    std::int32_t const* exclusions_{};
    double position_[3]{};
    double velocity_[3]{};

    CORSIKA_HOST_DEVICE inline interval_type intersect(std::int32_t node) const {
      return straight_boundary_reference::intersect(shapes_[node], position_, velocity_);
    }
    CORSIKA_HOST_DEVICE inline std::int32_t getParent(std::int32_t node) const {
      return nodes_[node].parentIndex_;
    }
    CORSIKA_HOST_DEVICE inline unsigned getChildCount(std::int32_t node) const {
      return nodes_[node].childCount_;
    }
    CORSIKA_HOST_DEVICE inline std::int32_t getChild(std::int32_t node,
                                                     unsigned index) const {
      return children_[nodes_[node].childOffset_ + index];
    }
    CORSIKA_HOST_DEVICE inline unsigned getExcludedCount(std::int32_t node) const {
      return nodes_[node].excludedCount_;
    }
    CORSIKA_HOST_DEVICE inline std::int32_t getExcluded(std::int32_t node,
                                                        unsigned index) const {
      return exclusions_[nodes_[node].excludedOffset_ + index];
    }
  };

  struct TreeFixture {
    std::vector<node_type> nodes_;
    std::vector<shape_type> shapes_;
    std::vector<std::int32_t> children_;
    std::vector<std::int32_t> exclusions_;
  };

  inline TreeFixture tree_fixture() {
    TreeFixture result;
    result.nodes_.resize(6);
    result.shapes_.resize(6, box(3.));
    result.shapes_[0].kind_ = shape_type::Kind::Universe;
    result.shapes_[1] = box(20.);
    for (int index = 2; index < 6; ++index) {
      result.shapes_[index].kind_ = shape_type::Kind::Sphere;
      result.shapes_[index].radiusM_ = 3.;
      result.shapes_[index].centerM_[0] = 8.;
    }
    result.nodes_[0].childCount_ = 3;
    result.nodes_[1].parentIndex_ = 0;
    result.nodes_[1].childOffset_ = 3;
    result.nodes_[1].childCount_ = 2;
    result.nodes_[1].excludedCount_ = 3;
    result.nodes_[2].parentIndex_ = result.nodes_[3].parentIndex_ = 1;
    result.nodes_[4].parentIndex_ = result.nodes_[5].parentIndex_ = 0;
    result.children_ = {1, 4, 5, 2, 3};
    result.exclusions_ = {4, 5, 4};
    return result;
  }

  struct Ray {
    double position_[3]{};
    double velocity_[3]{};
  };

  inline std::vector<Ray> rays() {
    return {{{0., 0., 0.}, {1., 0., 0.}},
            {{8., 0., 0.}, {1., 0., 0.}},
            {{20., 0., 0.}, {-1., 0., 0.}},
            {{-25., 0., 0.}, {1., 0., 0.}},
            {{.13, -.27, .41}, {.17, -.31, .79}}};
  }

} // namespace straight_boundary_reference
