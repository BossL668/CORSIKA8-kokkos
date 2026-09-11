/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#include <catch2/catch_all.hpp>

#include <corsika/framework/geometry/ConvexPolyhedron.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/modules/tracking/TrackingConvexPolyhedron.hpp>

#include <cmath>
#include <limits>
#include <random>

using namespace corsika;
using Catch::Approx;

namespace {
  ConvexPolyhedron makeTetrahedron() {
    auto const cs = get_root_CoordinateSystem();
    return ConvexPolyhedron::tetrahedron(
        {Point(cs, 0_m, 0_m, 0_m), Point(cs, 2_m, 0_m, 0_m),
         Point(cs, 0_m, 2_m, 0_m), Point(cs, 0_m, 0_m, 2_m)});
  }

  struct GeometryProbeParticle {
    Point position;
    MomentumVector momentum;
    HEPEnergyType getEnergy() const { return 10_GeV; }
    Point const& getPosition() const { return position; }
    MomentumVector const& getMomentum() const { return momentum; }
  };
} // namespace

TEST_CASE("Convex polyhedron tetrahedron units and boundaries", "[mountain][geometry]") {
  auto const cs = get_root_CoordinateSystem();
  auto const volume = makeTetrahedron();
  CHECK(volume.volumeCubicMeters() == Approx(8. / 6.));
  CHECK(volume.exportPlanes().size() == 4);
  CHECK(volume.contains(Point(cs, .2_m, .2_m, .2_m)));
  CHECK(volume.contains(Point(cs, 0_m, 0_m, 0_m)));
  CHECK_FALSE(volume.contains(Point(cs, 1_m, 1_m, 1_m)));
  CHECK_FALSE(volume.contains(Point(cs, -.001_m, .2_m, .2_m)));
  auto const chord = volume.intersectRay(Point(cs, -1_m, .2_m, .2_m),
                                         DirectionVector(cs, {17., 0., 0.}));
  REQUIRE(chord.intersects);
  CHECK(chord.entry_m == Approx(1.));
  CHECK(chord.exit_m == Approx(2.6));
  CHECK(chord.entry_face < 4);
  CHECK(chord.exit_face < 4);
  auto const inside = volume.intersectRay(Point(cs, .2_m, .2_m, .2_m),
                                          DirectionVector(cs, {1., 0., 0.}));
  CHECK(inside.entry_m == 0.);
  CHECK(inside.exit_m == Approx(1.4));
  auto const miss = volume.intersectRay(Point(cs, -1_m, 3_m, 3_m),
                                        DirectionVector(cs, {1., 0., 0.}));
  CHECK_FALSE(miss.intersects);
  auto const outward = volume.intersectRay(Point(cs, 0_m, .2_m, .2_m),
                                           DirectionVector(cs, {-1., 0., 0.}));
  REQUIRE(outward.intersects);
  CHECK(outward.entry_m == 0.);
  CHECK(outward.exit_m == 0.);
  auto const inward = volume.intersectRay(Point(cs, 0_m, .2_m, .2_m),
                                          DirectionVector(cs, {1., 0., 0.}));
  REQUIRE(inward.intersects);
  CHECK(inward.exit_m == Approx(1.6));
  CHECK_THROWS(volume.intersectRay(Point(cs, 0_m, 0_m, 0_m),
                                    DirectionVector(cs, {0., 0., 0.})));

  GeometryProbeParticle particle{Point(cs, .2_m, .2_m, .2_m),
                                  MomentumVector(cs, {6_GeV, 0_GeV, 0_GeV})};
  auto const intersection = tracking_line::ConvexPolyhedronTracking::intersect(
      particle, volume);
  REQUIRE(intersection.hasIntersections());
  CHECK(intersection.getExit() / (1_m / constants::c) == Approx(1.4 / .6));
}

TEST_CASE("Convex polyhedron shares exact host device clipping", "[mountain][geometry]") {
  auto const cs = get_root_CoordinateSystem();
  auto const volume = ConvexPolyhedron::box(cs, {3., -2., 7.}, {2., 3., 4.});
  CHECK(volume.volumeCubicMeters() == Approx(8. * 2. * 3. * 4.));
  std::mt19937_64 generator(903811);
  std::uniform_real_distribution<double> sample(-12., 12.);
  for (int i = 0; i < 10000; ++i) {
    double const x = sample(generator), y = sample(generator), z = sample(generator);
    bool const expected = std::abs(x - 3.) <= 2. && std::abs(y + 2.) <= 3. &&
                          std::abs(z - 7.) <= 4.;
    CHECK(volume.contains(Point(cs, x * 1_m, y * 1_m, z * 1_m)) == expected);
    auto const& planes = volume.exportPlanes();
    CHECK(geometry_detail::containsConvex(planes.data(), planes.size(), x, y, z) ==
          expected);
    auto const interval = geometry_detail::clipConvexRay(
        planes.data(), planes.size(), x, y, z, 1., 0., 0.);
    bool const rayExpected = x <= 5. && std::abs(y + 2.) <= 3. && std::abs(z - 7.) <= 4.;
    CHECK(interval.intersects == rayExpected);
    if (rayExpected) {
      CHECK(interval.entry_m == Approx(std::max(0., 1. - x)));
      CHECK(interval.exit_m == Approx(5. - x));
    }
    // Independent analytic slab oracle for non-axis-aligned directions.
    std::array<double, 3> direction{sample(generator), sample(generator),
                                    sample(generator)};
    double const norm = std::hypot(direction[0], direction[1], direction[2]);
    for (auto& coordinate : direction) coordinate /= norm;
    std::array<double, 3> const point{x, y, z};
    std::array<double, 3> const low{1., -5., 3.};
    std::array<double, 3> const high{5., 1., 11.};
    double expectedEntry = 0.;
    double expectedExit = std::numeric_limits<double>::infinity();
    for (std::size_t axis = 0; axis < 3; ++axis) {
      double a = (low[axis] - point[axis]) / direction[axis];
      double b = (high[axis] - point[axis]) / direction[axis];
      if (a > b) std::swap(a, b);
      expectedEntry = std::max(expectedEntry, a);
      expectedExit = std::min(expectedExit, b);
    }
    auto const slanted = geometry_detail::clipConvexRay(
        planes.data(), planes.size(), x, y, z, direction[0], direction[1], direction[2]);
    bool const slantedExpected = expectedEntry <= expectedExit;
    CHECK(slanted.intersects == slantedExpected);
    if (slantedExpected) {
      CHECK(slanted.entry_m == Approx(expectedEntry).epsilon(1e-12));
      CHECK(slanted.exit_m == Approx(expectedExit).epsilon(1e-12));
    }
  }
  auto const translated = make_translation(cs, {10_m, 20_m, 30_m});
  auto const translatedBox =
      ConvexPolyhedron::box(translated, {0., 0., 0.}, {1., 1., 1.});
  CHECK(translatedBox.contains(Point(cs, 10_m, 20_m, 30_m)));
  CHECK_FALSE(translatedBox.contains(Point(cs, 0_m, 0_m, 0_m)));
  auto const interval = translatedBox.intersectRay(Point(cs, 8_m, 20_m, 30_m),
                                                   DirectionVector(cs, {1., 0., 0.}));
  REQUIRE(interval.intersects);
  CHECK(interval.entry_m == Approx(1.));
  CHECK(interval.exit_m == Approx(3.));
}

TEST_CASE("Convex polyhedron rejects malformed and concave meshes",
          "[mountain][geometry]") {
  auto const cs = get_root_CoordinateSystem();
  auto const tetrahedron = makeTetrahedron();
  auto vertices = tetrahedron.verticesMeters();
  auto faces = tetrahedron.triangles();
  SECTION("missing triangle") {
    faces.pop_back();
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces));
  }
  SECTION("bad index") {
    faces[0][0] = vertices.size();
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces));
  }
  SECTION("non finite vertex") {
    vertices[0][0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces));
  }
  SECTION("degenerate triangle") {
    vertices[0] = vertices[1];
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces));
  }
  SECTION("duplicate triangle") {
    faces.push_back(faces.front());
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces));
  }
  SECTION("reversed winding is corrected") {
    std::swap(faces[0][1], faces[0][2]);
    ConvexPolyhedron corrected(cs, vertices, faces);
    CHECK(corrected.contains(Point(cs, .2_m, .2_m, .2_m)));
  }
  SECTION("flat tetrahedron") {
    vertices[3] = {.2, .2, 0.};
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces));
  }
  SECTION("concave closed mesh is not silently made convex") {
    auto const cube = ConvexPolyhedron::box(cs, {0., 0., 0.}, {1., 1., 1.});
    auto dented = cube.verticesMeters();
    dented[7] = {0., 0., 0.};
    CHECK_THROWS(ConvexPolyhedron(cs, dented, cube.triangles()));
  }
  SECTION("invalid tolerance, size, path and query") {
    CHECK_THROWS(ConvexPolyhedron(cs, vertices, faces, -1_m));
    CHECK_THROWS(ConvexPolyhedron::box(cs, {0., 0., 0.}, {0., 1., 1.}));
    CHECK_THROWS(ConvexPolyhedron::fromOBJ("/nonexistent/corsika-convex-test.obj", cs));
    CHECK_THROWS(tetrahedron.contains(
        Point(cs, std::numeric_limits<double>::infinity() * 1_m, 0_m, 0_m)));
  }
}

TEST_CASE("Convex polyhedron OBJ import preserves physical scale",
          "[mountain][geometry]") {
  auto const cs = get_root_CoordinateSystem();
  auto const fixtures = std::filesystem::path(__FILE__).parent_path() / "fixtures";
  auto const volume = ConvexPolyhedron::fromOBJ(fixtures / "convex_tetrahedron.obj", cs,
                                               1_km);
  CHECK(volume.volumeCubicMeters() == Approx(8.e9 / 6.));
  CHECK(volume.contains(Point(cs, 200_m, 200_m, 200_m)));
  auto const chord = volume.intersectRay(Point(cs, -1000_m, 200_m, 200_m),
                                         DirectionVector(cs, {1., 0., 0.}));
  REQUIRE(chord.intersects);
  CHECK(chord.entry_m == Approx(1000.));
  CHECK(chord.exit_m == Approx(2600.));
  CHECK_THROWS(ConvexPolyhedron::fromOBJ(fixtures / "convex_open.obj", cs));
}
