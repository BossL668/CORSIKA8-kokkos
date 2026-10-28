/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ShapeConversion.hpp>
#include <corsika/framework/geometry/Box.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/media/Universe.hpp>

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

  using namespace corsika;
  using Catch::Approx;
  using shape_type = kokkos::EnvironmentShape;

  void require_center(shape_type const& shape, Point const& point,
                      CoordinateSystemPtr const& frame) {
    auto const coordinates = point.getCoordinates(frame);
    for (int component = 0; component < 3; ++component) {
      CAPTURE(component);
      REQUIRE(shape.centerM_[component] == coordinates[component] / 1_m);
    }
  }

  void require_box_axes(shape_type const& shape, CoordinateSystemPtr const& boxFrame,
                        CoordinateSystemPtr const& frame) {
    std::array<DirectionVector, 3> const axes{DirectionVector(boxFrame, {1., 0., 0.}),
                                              DirectionVector(boxFrame, {0., 1., 0.}),
                                              DirectionVector(boxFrame, {0., 0., 1.})};
    for (int axis = 0; axis < 3; ++axis) {
      auto const components = axes[axis].getComponents(frame);
      for (int component = 0; component < 3; ++component) {
        CAPTURE(axis, component);
        REQUIRE(shape.axes_[axis][component] == components[component].magnitude());
      }
    }
  }

  void require_unused_vectors_zero(shape_type const& shape) {
    for (int axis = 0; axis < 3; ++axis) {
      REQUIRE(shape.halfExtentM_[axis] == 0.);
      REQUIRE(shape.normal_[axis] == 0.);
      for (int component = 0; component < 3; ++component) {
        REQUIRE(shape.axes_[axis][component] == 0.);
      }
    }
  }

  class UnsupportedVolume : public IVolume {
  public:
    bool contains(Point const&) const override { return false; }
  };

  template <typename TShape>
  class OverriddenShape : public TShape {
  public:
    using TShape::TShape;
    bool contains(Point const&) const override { return false; }
  };

} // namespace

TEST_CASE("Universe shape has an explicit tag rather than an infinite sphere",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  auto const shifted = make_translation(root, {1_km, -2_km, 3_km});
  Universe const universe(root);
  REQUIRE(std::isinf(universe.getRadius() / 1_m));
  for (auto const& frame : {root, shifted}) {
    auto const shape = kokkos::export_shape(universe, frame);
    REQUIRE(shape.kind_ == shape_type::Kind::Universe);
    REQUIRE(shape.radiusM_ == 0.);
    for (auto const center : shape.centerM_) { REQUIRE(center == 0.); }
    require_unused_vectors_zero(shape);
  }
}

TEST_CASE("Sphere shape export converts units and is independent of later mutation",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  Sphere sphere(Point(root, 125_cm, -250_cm, 3_km), 45_cm);
  auto const original = kokkos::export_shape(sphere, root);
  REQUIRE(original.kind_ == shape_type::Kind::Sphere);
  REQUIRE(original.centerM_[0] == Approx(1.25));
  REQUIRE(original.centerM_[1] == Approx(-2.5));
  REQUIRE(original.centerM_[2] == Approx(3000.));
  REQUIRE(original.radiusM_ == Approx(0.45));
  require_unused_vectors_zero(original);

  sphere.setCenter(Point(root, -4_m, 5_m, -6_m));
  sphere.setRadius(2_km);
  auto const updated = kokkos::export_shape(sphere, root);
  require_center(updated, sphere.getCenter(), root);
  REQUIRE(updated.radiusM_ == Approx(2000.));
  REQUIRE(original.centerM_[0] == Approx(1.25));
  REQUIRE(original.radiusM_ == Approx(0.45));
}

TEST_CASE("Sphere shapes transform into a common frame without rebasing the source",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {1_m, -2_m, 3_m});
  auto const rotated = make_rotation(root, QuantityVector<dimensionless_d>{0., 0., 1.},
                                     std::acos(-1.) / 2.);
  auto const nested = make_translation(rotated, {4_m, 5_m, -6_m});
  std::vector<CoordinateSystemPtr> const frames{root, translated, rotated, nested};
  for (auto const& sourceFrame : frames) {
    Sphere const sphere(Point(sourceFrame, 4_m, -5_m, 6_m), 7_m);
    for (auto const& frame : frames) {
      auto const shape = kokkos::export_shape(sphere, frame);
      require_center(shape, sphere.getCenter(), frame);
      REQUIRE(shape.radiusM_ == 7.);
      REQUIRE(sphere.getCoordinateSystem() == sourceFrame);
      REQUIRE(sphere.getCenter().getCoordinates()[0] / 1_m == 4.);
    }
  }
}

TEST_CASE("Box shape export retains native half extents and local basis vectors",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {100_m, -200_m, 300_m});
  auto const rotated =
      make_rotation(translated, QuantityVector<dimensionless_d>{1., 2., 3.}, 0.7);
  auto const nested = make_translation(rotated, {-4_m, 5_m, 6_m});
  for (auto const& boxFrame : {root, translated, rotated, nested}) {
    Box const box(boxFrame, 125_cm, 250_cm, 3_km);
    for (auto const& frame : {root, rotated, nested}) {
      auto const shape = kokkos::export_shape(box, frame);
      REQUIRE(shape.kind_ == shape_type::Kind::Box);
      require_center(shape, Point(boxFrame, 0_m, 0_m, 0_m), frame);
      require_box_axes(shape, boxFrame, frame);
      REQUIRE(shape.halfExtentM_[0] == Approx(1.25));
      REQUIRE(shape.halfExtentM_[1] == Approx(2.5));
      REQUIRE(shape.halfExtentM_[2] == Approx(3000.));
      REQUIRE(shape.radiusM_ == 0.);
      for (auto const normal : shape.normal_) { REQUIRE(normal == 0.); }
      REQUIRE(box.getCoordinateSystem() == boxFrame);
    }
  }

  Box const cube(root, 8_m);
  auto const cubeShape = kokkos::export_shape(cube, root);
  for (auto const extent : cubeShape.halfExtentM_) { REQUIRE(extent == 4.); }
  // The export stores native parameters; it does not introduce a new contains test.
  REQUIRE(cube.contains(Point(root, 3_m, 0_m, 0_m)));
  REQUIRE_FALSE(cube.contains(Point(root, 4_m, 0_m, 0_m)));
}

TEST_CASE("Box shape re-export uses its current coordinate system after rotation",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {10_m, 20_m, 30_m});
  Box box(translated, 1_m, 2_m, 3_m);
  auto const before = kokkos::export_shape(box, root);
  box.rotate(QuantityVector<dimensionless_d>{0., 0., 1.}, std::acos(-1.) / 2.);
  auto const rotated = box.getCoordinateSystem();
  REQUIRE(rotated != translated);
  // C8 keeps the cached center in the old frame when Box::rotate changes cs_.
  REQUIRE(box.getCenter().getCoordinateSystem() == translated);
  auto const after = kokkos::export_shape(box, root);
  require_center(after, Point(rotated, 0_m, 0_m, 0_m), root);
  require_box_axes(after, rotated, root);
  REQUIRE(std::abs(after.axes_[0][1]) == Approx(1.));
  REQUIRE(std::abs(after.axes_[1][0]) == Approx(1.));
  REQUIRE(before.axes_[0][0] == 1.);
  REQUIRE(before.axes_[0][1] == 0.);
  for (int axis = 0; axis < 3; ++axis) {
    REQUIRE(after.halfExtentM_[axis] == before.halfExtentM_[axis]);
  }
}

TEST_CASE("Separation planes retain the native outward normal in the common frame",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  auto const translated = make_translation(root, {3_m, -4_m, 5_m});
  auto const rotated = make_rotation(root, QuantityVector<dimensionless_d>{1., 0., 0.},
                                     std::acos(-1.) / 3.);
  Point const center(translated, 1_m, 2_m, 3_m);
  DirectionVector const normal(rotated, {0., 0., 7.});
  Plane const plane(center, normal);
  SeparationPlane const volume(plane);
  for (auto const& frame : {root, translated, rotated}) {
    auto const shape = kokkos::export_shape(volume, frame);
    REQUIRE(shape.kind_ == shape_type::Kind::SeparationPlane);
    require_center(shape, plane.getCenter(), frame);
    auto const normalComponents = plane.getNormal().getComponents(frame);
    for (int component = 0; component < 3; ++component) {
      REQUIRE(shape.normal_[component] == normalComponents[component].magnitude());
      REQUIRE(shape.halfExtentM_[component] == 0.);
      for (int axis = 0; axis < 3; ++axis) {
        REQUIRE(shape.axes_[axis][component] == 0.);
      }
    }
    REQUIRE(shape.radiusM_ == 0.);
    REQUIRE(plane.getCenter().getCoordinateSystem() == translated);
    REQUIRE(plane.getNormal().getCoordinateSystem() == rotated);
  }
  REQUIRE(volume.contains(center));
  REQUIRE(volume.contains(center - plane.getNormal() * 1_m));
  REQUIRE_FALSE(volume.contains(center + plane.getNormal() * 1_m));
}

TEST_CASE("Finite degenerate shape dimensions are exported without repair",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  for (auto const dimension : {0., -0., -3., 4.}) {
    CAPTURE(dimension);
    Sphere const sphere(Point(root, 0_m, 0_m, 0_m), dimension * 1_m);
    auto const sphereShape = kokkos::export_shape(sphere, root);
    REQUIRE(sphereShape.radiusM_ == dimension);
    REQUIRE(std::signbit(sphereShape.radiusM_) == std::signbit(dimension));
    Box const box(root, dimension * 1_m, 2_m, 3_m);
    auto const boxShape = kokkos::export_shape(box, root);
    REQUIRE(boxShape.halfExtentM_[0] == dimension);
    REQUIRE(std::signbit(boxShape.halfExtentM_[0]) == std::signbit(dimension));
  }
}

TEST_CASE("Unsupported shapes and overriding subclasses are rejected explicitly",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  UnsupportedVolume const unsupported;
  OverriddenShape<Sphere> const sphere(Point(root, 0_m, 0_m, 0_m), 1_m);
  OverriddenShape<Box> const box(root, 1_m);
  OverriddenShape<SeparationPlane> const plane(
      Plane(Point(root, 0_m, 0_m, 0_m), DirectionVector(root, {0., 0., 1.})));
  OverriddenShape<Universe> const universe(root);
  std::array<IVolume const*, 5> const volumes{&unsupported, &sphere, &box, &plane,
                                              &universe};
  for (auto const* volume : volumes) {
    REQUIRE_THROWS_AS(kokkos::export_shape(*volume, root), std::invalid_argument);
  }
}

TEST_CASE("Shape export rejects missing coordinate systems before conversion",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  CoordinateSystemPtr const missing;
  Sphere const sphere(Point(root, 0_m, 0_m, 0_m), 1_m);
  Box const box(root, 1_m);
  SeparationPlane const plane(
      Plane(Point(root, 0_m, 0_m, 0_m), DirectionVector(root, {0., 0., 1.})));
  Universe const universe(root);
  std::array<IVolume const*, 4> const volumes{&sphere, &box, &plane, &universe};
  for (auto const* volume : volumes) {
    REQUIRE_THROWS_AS(kokkos::export_shape(*volume, missing), std::invalid_argument);
  }

  Sphere const missingSphere(Point(missing, 0_m, 0_m, 0_m), 1_m);
  Box const missingBox(missing, 1_m);
  SeparationPlane const missingCenter(
      Plane(Point(missing, 0_m, 0_m, 0_m), DirectionVector(root, {0., 0., 1.})));
  SeparationPlane const missingNormal(
      Plane(Point(root, 0_m, 0_m, 0_m), DirectionVector(missing, {0., 0., 1.})));
  Universe const missingUniverse(missing);
  std::array<IVolume const*, 5> const missingSources{
      &missingSphere, &missingBox, &missingCenter, &missingNormal, &missingUniverse};
  for (auto const* volume : missingSources) {
    REQUIRE_THROWS_AS(kokkos::export_shape(*volume, root), std::invalid_argument);
  }
}

TEST_CASE("Nonfinite shape parameters and transformed coordinates fail explicitly",
          "[kokkos_shape]") {
  auto const root = get_root_CoordinateSystem();
  for (auto const value :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    CAPTURE(value);
    Sphere const radius(Point(root, 0_m, 0_m, 0_m), value * 1_m);
    Sphere const center(Point(root, value * 1_m, 0_m, 0_m), 1_m);
    Box const extent(root, 1_m, value * 1_m, 1_m);
    SeparationPlane const planeCenter(
        Plane(Point(root, 0_m, value * 1_m, 0_m), DirectionVector(root, {0., 0., 1.})));
    SeparationPlane const planeNormal(
        Plane(Point(root, 0_m, 0_m, 0_m), DirectionVector(root, {value, 0., 1.})));
    std::array<IVolume const*, 5> const volumes{&radius, &center, &extent, &planeCenter,
                                                &planeNormal};
    for (auto const* volume : volumes) {
      REQUIRE_THROWS_AS(kokkos::export_shape(*volume, root), std::invalid_argument);
    }

    auto const badFrame = make_translation(root, {value * 1_m, 0_m, 0_m});
    Sphere const transformedSphere(Point(badFrame, 0_m, 0_m, 0_m), 1_m);
    Box const transformedBox(badFrame, 1_m);
    REQUIRE_THROWS_AS(kokkos::export_shape(transformedSphere, root),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(kokkos::export_shape(transformedBox, root), std::invalid_argument);
    Sphere const finiteSphere(Point(root, 0_m, 0_m, 0_m), 1_m);
    REQUIRE_THROWS_AS(kokkos::export_shape(finiteSphere, badFrame),
                      std::invalid_argument);
  }
}
