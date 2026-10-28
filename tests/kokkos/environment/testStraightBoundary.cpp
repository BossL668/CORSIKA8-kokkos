/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "HostEnvironmentTree.hpp"
#include <corsika/kokkos/environment/StraightBoundaryAccess.hpp>
#include <corsika/kokkos/environment/EnvironmentTopology.hpp>
#include <corsika/kokkos/environment/ShapeConversion.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <corsika/framework/core/PhysicalConstants.hpp>
#include <corsika/framework/core/PhysicalGeometry.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/Universe.hpp>
#include <corsika/modules/tracking/TrackingStraight.hpp>

// This TU compares native units and host node interfaces. The device test TU
// compiles the same independent oracle for numerical device accessors instead.
#define CORSIKA_BOUNDARY_REFERENCE_HOST_ONLY
#include "StraightBoundaryReference.hpp"
#undef CORSIKA_BOUNDARY_REFERENCE_HOST_ONLY

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace {

  using namespace corsika;
  using tree_type = environment_test::HostEnvironmentTree;
  using environment_type = Environment<IMediumModel>;
  using node_type = environment_type::BaseNodeType;
  using topology_type = kokkos::EnvironmentTopology<IMediumModel>;

  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_native(double actual, double expected) {
    if (!std::isfinite(expected)) {
      require_exact(actual, expected);
      return;
    }
    // Native Sphere uses multiplication by an inverse denominator; numerical
    // production uses division. Coordinate transforms add ordinary roundoff.
    // These regular finite fixtures are deliberately away from tangent roots.
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::abs(actual - expected) <= 5.e-12 * std::max(std::abs(expected), 1.e-9));
  }

  tree_type make_view(straight_boundary_reference::TreeFixture const& input) {
    return tree_type(input.nodes_, input.shapes_, input.children_, input.exclusions_);
  }

  void check_production() {
    INFO("Fixed production shape algebra, including IEEE edge behavior");
    for (auto const& query : straight_boundary_reference::queries()) {
      CAPTURE(query.shape_.kind_, query.position_[0], query.position_[1],
              query.position_[2], query.velocity_[0], query.velocity_[1],
              query.velocity_[2]);
      auto const actual =
          kokkos::straight_intersect(query.shape_, query.position_, query.velocity_);
      auto const expected = straight_boundary_reference::intersect(
          query.shape_, query.position_, query.velocity_);
      REQUIRE(actual.hasIntersections() == expected.hasIntersections());
      require_exact(actual.getEntry(), expected.getEntry());
      require_exact(actual.getExit(), expected.getExit());
    }
    auto const fixture = straight_boundary_reference::tree_fixture();
    auto const view = make_view(fixture);
    for (auto const& ray : straight_boundary_reference::rays()) {
      kokkos::StraightBoundaryAccess<environment_test::HostEnvironmentTree> const access(
          view, ray.position_, ray.velocity_);
      straight_boundary_reference::Tree reference{
          fixture.nodes_.data(), fixture.shapes_.data(), fixture.children_.data(),
          fixture.exclusions_.data()};
      for (int axis = 0; axis < 3; ++axis) {
        reference.position_[axis] = ray.position_[axis];
        reference.velocity_[axis] = ray.velocity_[axis];
      }
      for (std::int32_t current = 0; current < 6; ++current) {
        for (double limit : {0., .5, 5., 100., std::numeric_limits<double>::infinity()}) {
          auto const actual = kokkos::next_environment_boundary(access, current, limit);
          auto const expected = boundary_reference::next(reference, current, limit);
          REQUIRE(actual.getNode() == expected.node_);
          require_exact(actual.getParameter(), expected.parameter_);
        }
      }
    }
  }

  struct Particle {
    Point position_;
    MomentumVector momentum_;
    node_type const* node_{};
    Point const& getPosition() const { return position_; }
    MomentumVector const& getMomentum() const { return momentum_; }
    HEPEnergyType getEnergy() const { return 1_GeV; }
    node_type const* getNode() const { return node_; }
  };

  struct NativeAccess {
    Particle const& particle_;
    auto intersect(node_type const* node) const {
      return tracking_line::Tracking::intersect(particle_, *node);
    }
    auto getParent(node_type const* node) const { return node->getParent(); }
    unsigned getChildCount(node_type const* node) const {
      return node->getChildNodes().size();
    }
    auto getChild(node_type const* node, unsigned index) const {
      return node->getChildNodes()[index].get();
    }
    unsigned getExcludedCount(node_type const* node) const {
      return node->getExcludedNodes().size();
    }
    auto getExcluded(node_type const* node, unsigned index) const {
      return node->getExcludedNodes()[index];
    }
  };

  struct BoundaryProbe : Intersect<BoundaryProbe> {
    using Intersect<BoundaryProbe>::nextIntersect;
    static auto intersect(Particle const& particle, node_type const& node) {
      return tracking_line::Tracking::intersect(particle, node);
    }
  };

  void coordinates(Particle const& particle, CoordinateSystemPtr const& frame,
                   double (&position)[3], double (&velocity)[3]) {
    auto const values = particle.getPosition().getCoordinates(frame);
    VelocityVector const tangent =
        particle.getMomentum() / particle.getEnergy() * constants::c;
    auto const components = tangent.getComponents(frame);
    for (int axis = 0; axis < 3; ++axis) {
      position[axis] = values[axis] / 1_m;
      velocity[axis] = components[axis] / (1_m / 1_s);
    }
  }

  void check_native_shapes(CoordinateSystemPtr const& shapeFrame,
                           CoordinateSystemPtr const& commonFrame) {
    INFO(
        "Native TrackingStraight primitives with unitful velocities and frame "
        "transforms");
    std::vector<std::unique_ptr<IVolume>> shapes;
    shapes.push_back(std::make_unique<Universe>(shapeFrame));
    shapes.push_back(std::make_unique<Sphere>(Point(shapeFrame, 0_m, 0_m, 0_m), 3_m));
    shapes.push_back(std::make_unique<Box>(shapeFrame, 2_m, 3_m, 4_m));
    shapes.push_back(std::make_unique<SeparationPlane>(
        Plane(Point(shapeFrame, .17_m, -.23_m, .31_m),
              DirectionVector(shapeFrame, {1., 2., 3.}))));
    for (auto& shape : shapes) {
      node_type node(std::move(shape));
      auto const exported = kokkos::export_shape(node.getVolume(), commonFrame);
      for (double x : {-5.13, -.37, .27, 5.41}) {
        for (double direction : {-.3, .3}) {
          Particle const particle{
              Point(shapeFrame, x * 1_m, .31_m, -.27_m),
              MomentumVector(shapeFrame, direction * 1_GeV, .07_GeV, .11_GeV), &node};
          double position[3], velocity[3];
          coordinates(particle, commonFrame, position, velocity);
          auto const actual = kokkos::straight_intersect(exported, position, velocity);
          auto const expected = tracking_line::Tracking::intersect(particle, node);
          REQUIRE(actual.hasIntersections() == expected.hasIntersections());
          if (expected.hasIntersections()) {
            require_native(actual.getEntry(), expected.getEntry() / 1_s);
            require_native(actual.getExit(), expected.getExit() / 1_s);
          }
        }
      }
    }
  }

  void check_native_boundary(CoordinateSystemPtr const& shapeFrame,
                             CoordinateSystemPtr const& commonFrame) {
    INFO("Native CRTP adapter and numerical geometry use the same ordered tree search");
    environment_type environment;
    auto outer = environment_type::createNode<Box>(shapeFrame, 20_m, 20_m, 20_m);
    auto sphere = [&]() {
      return environment_type::createNode<Sphere>(Point(shapeFrame, 8_m, 0_m, 0_m), 3_m);
    };
    auto first = sphere();
    auto second = sphere();
    auto excludedFirst = sphere();
    auto excludedSecond = sphere();
    outer->excludeOverlapWith(excludedFirst);
    outer->excludeOverlapWith(excludedSecond);
    outer->excludeOverlapWith(excludedFirst);
    outer->addChild(std::move(first));
    outer->addChild(std::move(second));
    environment.getUniverse()->addChild(std::move(outer));
    environment.getUniverse()->addChild(std::move(excludedFirst));
    environment.getUniverse()->addChild(std::move(excludedSecond));
    topology_type const topology(*environment.getUniverse());
    straight_boundary_reference::TreeFixture fixture;
    fixture.nodes_ = topology.getNodes();
    fixture.children_ = topology.getChildren();
    fixture.exclusions_ = topology.getExclusions();
    for (std::int32_t node = 0; node < 6; ++node) {
      fixture.shapes_.push_back(kokkos::export_shape(
          topology.getNativeNode(node).get_ptr()->getVolume(), commonFrame));
    }
    auto const view = make_view(fixture);
    for (double x : {-4.17, 4.13, 12.11, 21.37}) {
      for (std::int32_t current : {0, 1, 2, 4}) {
        auto const* node = topology.getNativeNode(current).get_ptr();
        Particle const particle{Point(shapeFrame, x * 1_m, .31_m, -.27_m),
                                MomentumVector(shapeFrame, .3_GeV, .07_GeV, .05_GeV),
                                node};
        double position[3], velocity[3];
        coordinates(particle, commonFrame, position, velocity);
        kokkos::StraightBoundaryAccess<environment_test::HostEnvironmentTree> const
            access(view, position, velocity);
        for (double limit : {1.e-8, 1.e-6, std::numeric_limits<double>::infinity()}) {
          CAPTURE(x, current, limit);
          auto const actual = kokkos::next_environment_boundary(access, current, limit);
          auto const expected =
              boundary_reference::next(NativeAccess{particle}, node, limit * 1_s);
          auto const native = BoundaryProbe{}.nextIntersect(particle, limit * 1_s);
          REQUIRE(std::get<1>(native) == expected.node_);
          require_exact(std::get<0>(native) / 1_s, expected.parameter_ / 1_s);
          REQUIRE(actual.getNode() ==
                  topology.getNodeIndex((
                      expected.node_ ? topology_type::node_reference_type(*expected.node_)
                                     : boost::none)));
          require_native(actual.getParameter(), expected.parameter_ / 1_s);
        }
      }
    }
  }

} // namespace

TEST_CASE("Straight boundary access preserves production and native finite geometry",
          "[straight_boundary]") {
  REQUIRE_FALSE(environment_test::is_initialized());
  {
    kokkos::Runtime runtime;
    check_production();
    auto const root = get_root_CoordinateSystem();
    auto const translated = make_translation(root, {1_km, -2_km, 3_km});
    auto const rotated =
        make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, .7);
    for (auto const& frames : std::array<std::array<CoordinateSystemPtr, 2>, 3>{
             {{root, root}, {rotated, root}, {root, rotated}}}) {
      check_native_shapes(frames[0], frames[1]);
      check_native_boundary(frames[0], frames[1]);
    }
  }
  REQUIRE_FALSE(environment_test::is_initialized());
  REQUIRE(environment_test::is_finalized());
}
