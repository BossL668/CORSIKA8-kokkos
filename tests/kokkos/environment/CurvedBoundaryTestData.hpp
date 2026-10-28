/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "CurvedBoundaryReference.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace curved_boundary_test {

  using shape_type = corsika::kokkos::EnvironmentShape;
  using tree_fixture_type = straight_boundary_reference::TreeFixture;

  struct Ray {
    double positionM_[3]{};
    double velocityMPerS_[3]{1., 0., 0.};
    double direction_[3]{1., 0., 0.};
    double fieldT_[3]{0., 0., 1.};
    double speedMPerS_{1.};
    double momentumKgMPerS_{1.};
    double chargeC_{1.};
    double gyroradiusM_{1.};
    bool linear_{};
  };

  enum class Category {
    Linear,
    Universe,
    Plane,
    Parallel,
    Threshold,
    EmptyRoots,
    Unsupported,
    IncompleteRoots,
    NonfiniteRoots,
    Guard,
    Earth
  };

  struct Query {
    shape_type shape_;
    Ray ray_;
    Category category_{Category::Plane};
    int valid_{-1};
    bool unsupported_{};
    bool incomplete_{};
    bool checkFlags_{true};
    bool knownEntry_{};
    double entry_{};
  };

  struct Outcome {
    double entry_{};
    double exit_{};
    bool valid_{};
    bool unsupported_{};
    bool incomplete_{};
  };

  struct SpherePolynomial {
    double quadratic_{};
    double linear_{};
    double constant_{};
  };

  // Independent coefficient evidence for the targeted root-status fixtures.
  // These are the pinned production expressions, not migrated helper calls.
  KOKKOS_INLINE_FUNCTION SpherePolynomial sphere_polynomial(Query const& query) {
    using namespace curved_boundary_reference;
    auto const& ray = query.ray_;
    double const projected[3]{difference(product(ray.direction_[1], ray.fieldT_[2]),
                                         product(ray.direction_[2], ray.fieldT_[1])),
                              difference(product(ray.direction_[2], ray.fieldT_[0]),
                                         product(ray.direction_[0], ray.fieldT_[2])),
                              difference(product(ray.direction_[0], ray.fieldT_[1]),
                                         product(ray.direction_[1], ray.fieldT_[0]))};
    double const delta[3]{ray.positionM_[0] - query.shape_.centerM_[0],
                          ray.positionM_[1] - query.shape_.centerM_[1],
                          ray.positionM_[2] - query.shape_.centerM_[2]};
    double const deltaNorm = ::sqrt(dot(delta, delta));
    double const k = ray.chargeC_ / ray.momentumKgMPerS_;
    double const denominator = 4. / (dot(projected, projected) * k * k);
    return {product(sum(product(dot(projected, delta), k), 1), denominator),
            dot(ray.direction_, delta) * 2 * denominator,
            (deltaNorm + query.shape_.radiusM_) * (deltaNorm - query.shape_.radiusM_) *
                denominator};
  }

  KOKKOS_INLINE_FUNCTION curved_boundary_reference::Access reference(Ray const& ray) {
    curved_boundary_reference::Access result;
    for (int axis = 0; axis < 3; ++axis) {
      result.position_[axis] = ray.positionM_[axis];
      result.velocity_[axis] = ray.velocityMPerS_[axis];
      result.direction_[axis] = ray.direction_[axis];
      result.field_[axis] = ray.fieldT_[axis];
    }
    result.speed_ = ray.speedMPerS_;
    result.momentumSI_ = ray.momentumKgMPerS_;
    result.chargeSI_ = ray.chargeC_;
    result.gyroradiusM_ = ray.gyroradiusM_;
    result.linear_ = ray.linear_;
    return result;
  }

  KOKKOS_INLINE_FUNCTION Outcome reference_shape(Query const& query) {
    auto access = reference(query.ray_);
    auto const interval = access.intersectShape(query.shape_);
    return {interval.entry_, interval.exit_, interval.valid_, access.unsupportedShape_,
            access.incompleteRoots_};
  }

  inline shape_type sphere(double radius) {
    shape_type result;
    result.kind_ = shape_type::Kind::Sphere;
    result.radiusM_ = radius;
    return result;
  }

  inline shape_type universe() {
    shape_type result;
    result.kind_ = shape_type::Kind::Universe;
    return result;
  }

  inline std::vector<Query> queries() {
    std::vector<Query> result;
    // Reuse the independently checked straight inputs, including IEEE cases.
    // These deliberately probe only the linear short circuit, not the finite
    // physical-trajectory contract of the curved polynomial path.
    for (auto const& straight : straight_boundary_reference::queries()) {
      Query query;
      query.shape_ = straight.shape_;
      query.category_ = Category::Linear;
      query.ray_.linear_ = true;
      query.ray_.gyroradiusM_ = std::numeric_limits<double>::infinity();
      for (int axis = 0; axis < 3; ++axis) {
        query.ray_.positionM_[axis] = straight.position_[axis];
        query.ray_.velocityMPerS_[axis] = straight.velocity_[axis];
      }
      result.push_back(query);
    }
    Query query;
    query.shape_ = universe();
    query.category_ = Category::Universe;
    query.valid_ = 0;
    result.push_back(query);
    query.shape_ = straight_boundary_reference::box(1.);
    query.category_ = Category::Unsupported;
    query.unsupported_ = true;
    result.push_back(query);

    query = {};
    query.shape_.kind_ = shape_type::Kind::SeparationPlane;
    query.shape_.normal_[1] = 1.;
    query.category_ = Category::Plane;
    query.ray_.positionM_[1] = .5;
    query.valid_ = 1;
    query.knownEntry_ = true;
    query.entry_ = 1.;
    result.push_back(query); // -L^2+1=0: choose the nonnegative root.
    query.ray_.positionM_[1] = -.5;
    query.entry_ = std::numeric_limits<double>::infinity();
    result.push_back(query); // No real root is a valid (+infinity,+infinity) interval.
    query.ray_.positionM_[1] = 0.;
    query.entry_ = 0.;
    result.push_back(query);
    query.shape_.normal_[1] = 0.;
    query.shape_.normal_[0] = 1.;
    query.ray_.positionM_[0] = -2.;
    query.entry_ = 2.;
    result.push_back(query); // Quadratic coefficient is exactly zero.
    query.ray_.chargeC_ = 0.;
    result.push_back(query); // Neutral plane uses the straight branch.
    query.ray_.chargeC_ = 1.;
    query.ray_.positionM_[0] = 2.;
    query.entry_ = std::numeric_limits<double>::infinity();
    result.push_back(query); // Only a negative root.
    query.shape_.normal_[0] = 0.;
    query.shape_.normal_[1] = 1.;
    query.ray_.positionM_[0] = 0.;
    query.ray_.positionM_[1] = -.5;
    query.ray_.direction_[0] = query.ray_.velocityMPerS_[0] = .6;
    query.ray_.direction_[1] = query.ray_.velocityMPerS_[1] = .8;
    query.entry_ = 1.;
    result.push_back(query); // Two positive roots, select the smaller one.

    query = {};
    query.shape_ = sphere(1.);
    query.category_ = Category::Parallel;
    query.ray_.fieldT_[0] = 1.;
    query.ray_.fieldT_[2] = 0.;
    query.valid_ = 1;
    query.knownEntry_ = true;
    query.entry_ = -1.;
    result.push_back(query);

    for (double radius :
         {std::nextafter(1., 0.), 1., std::nextafter(1., 2.), std::nextafter(1000., 0.),
          1000., std::nextafter(1000., 2000.)}) {
      query = {};
      query.shape_ = sphere(1.);
      query.category_ = Category::Threshold;
      query.ray_.positionM_[1] = 2.;
      query.ray_.direction_[0] = query.ray_.velocityMPerS_[0] = 0.;
      query.ray_.direction_[1] = query.ray_.velocityMPerS_[1] = -1.;
      query.ray_.gyroradiusM_ = query.ray_.momentumKgMPerS_ = radius;
      query.checkFlags_ = false;
      result.push_back(query);
    }

    query = {};
    query.shape_ = sphere(1.);
    query.category_ = Category::EmptyRoots;
    query.ray_.positionM_[1] = 1.1;
    query.valid_ = 0;
    result.push_back(query); // L^4-.4 L^2+.84 has no real roots.

    query = {};
    query.shape_ = sphere(1.);
    query.category_ = Category::IncompleteRoots;
    query.ray_.positionM_[1] = 1.25;
    query.ray_.momentumKgMPerS_ = query.ray_.gyroradiusM_ = .5;
    query.valid_ = 0;
    query.incomplete_ = true;
    result.push_back(query); // (L^2-.75)^2: one distinct forward root, no invented exit.

    query = {};
    query.shape_ = sphere(1.e75);
    query.category_ = Category::NonfiniteRoots;
    query.ray_.positionM_[0] = 1.e-260;
    query.ray_.positionM_[1] = 5.e74;
    query.ray_.momentumKgMPerS_ = query.ray_.gyroradiusM_ = 1.e75;
    query.valid_ = 0;
    query.incomplete_ = true;
    // Extreme but finite and mutually consistent inputs; binary64 coefficient
    // normalization loses the nonzero linear term before extended arithmetic.
    result.push_back(query);

    for (double length : {.99e-4, 1.e-4, 1.01e-4}) {
      query = {};
      query.shape_ =
          sphere(std::sqrt(length * length + 2500 * length * length * length * length));
      query.category_ = Category::Guard;
      query.ray_.gyroradiusM_ = query.ray_.momentumKgMPerS_ = .01;
      query.checkFlags_ = length != 1.e-4;
      query.incomplete_ = length < 1.e-4;
      query.valid_ = length == 1.e-4 ? -1 : length > 1.e-4;
      result.push_back(query);
    }

    for (double gap : {-1000., -100., 100., 1000.}) {
      for (double directionX : {0., .6}) {
        for (double chargeSign : {-1., 1.}) {
          query = {};
          query.shape_ = sphere(6371000.);
          query.category_ = Category::Earth;
          query.ray_.positionM_[2] = 6371000. + gap;
          query.ray_.direction_[0] = directionX;
          query.ray_.direction_[2] = directionX == 0. ? -1. : -.8;
          query.ray_.speedMPerS_ = 299792458.;
          for (int axis = 0; axis < 3; ++axis) {
            query.ray_.velocityMPerS_[axis] =
                query.ray_.speedMPerS_ * query.ray_.direction_[axis];
          }
          query.ray_.fieldT_[1] = 5.e-5;
          query.ray_.fieldT_[2] = 0.;
          query.ray_.momentumKgMPerS_ = 1.602176634e-19;
          query.ray_.chargeC_ = chargeSign * 1.602176634e-19;
          query.ray_.gyroradiusM_ = 20000.;
          query.valid_ = 1;
          result.push_back(query);
        }
      }
    }
    return result;
  }

  inline tree_fixture_type ordered_tree() {
    auto result = straight_boundary_reference::tree_fixture();
    result.shapes_[1] = sphere(20.);
    return result;
  }

  inline tree_fixture_type failing_tree() {
    tree_fixture_type result;
    result.nodes_.resize(3);
    result.shapes_ = {universe(), sphere(1.), straight_boundary_reference::box(1.)};
    result.nodes_[0].childCount_ = 1;
    result.nodes_[1].parentIndex_ = 0;
    result.nodes_[1].childOffset_ = 1;
    result.nodes_[1].childCount_ = 1;
    result.nodes_[2].parentIndex_ = 1;
    result.children_ = {1, 2};
    return result;
  }

  struct SearchQuery {
    Ray ray_;
    std::int32_t firstNode_{};
    std::int32_t secondNode_{};
    double firstLimit_{};
    double secondLimit_{};
  };

  inline std::vector<SearchQuery> search_queries(bool failureTree) {
    std::vector<SearchQuery> result;
    if (failureTree) {
      for (bool linear : {false, true}) {
        for (int repeat = 0; repeat < 32; ++repeat) {
          SearchQuery query;
          query.ray_.linear_ = linear;
          query.ray_.positionM_[1] = 1.25;
          query.ray_.momentumKgMPerS_ = query.ray_.gyroradiusM_ = .5;
          query.firstNode_ = 1;
          query.secondNode_ = 0;
          query.firstLimit_ = 2.;
          query.secondLimit_ = .5;
          result.push_back(query);
        }
      }
      return result;
    }
    for (auto const& input : straight_boundary_reference::rays()) {
      for (bool linear : {false, true}) {
        for (std::int32_t current = 0; current < 6; ++current) {
          for (double limit :
               {0., .5, 5., 100., std::numeric_limits<double>::infinity()}) {
            SearchQuery query;
            double const norm = std::sqrt(
                curved_boundary_reference::dot(input.velocity_, input.velocity_));
            for (int axis = 0; axis < 3; ++axis) {
              query.ray_.positionM_[axis] = input.position_[axis];
              query.ray_.direction_[axis] = input.velocity_[axis] / norm;
              query.ray_.velocityMPerS_[axis] = query.ray_.direction_[axis];
            }
            query.ray_.momentumKgMPerS_ = query.ray_.gyroradiusM_ = 20.;
            query.ray_.linear_ = linear;
            query.firstNode_ = current;
            query.secondNode_ = (current + 1) % 6;
            query.firstLimit_ = limit;
            query.secondLimit_ = .25;
            result.push_back(query);
          }
        }
      }
    }
    return result;
  }

} // namespace curved_boundary_test
