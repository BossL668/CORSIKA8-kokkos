/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentShape.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <array>
#include <type_traits>

namespace {
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using shape_type = corsika::kokkos::EnvironmentShape;
  using kind_type = shape_type::Kind;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
} // namespace

TEST_CASE("Shape records retain every field in the configured execution space",
          "[environment_transfer]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    int constexpr count = 257;
    std::array<kind_type, 5> constexpr kinds{kind_type::Invalid, kind_type::Universe,
                                             kind_type::Sphere, kind_type::Box,
                                             kind_type::SeparationPlane};
    std::array<double, 5> constexpr kindChecks{-1., 11., 22., 33., 44.};
    Kokkos::View<shape_type*, Kokkos::HostSpace> hostShapes("host_shapes", count);
    // Fill even inactive fields with distinct exact binary fractions. These are
    // transfer fixtures, not physical volumes or a containment/search test.
    for (int index = 0; index < count; ++index) {
      auto& shape = hostShapes(index);
      shape.kind_ = kinds[index % kinds.size()];
      shape.radiusM_ = (index + 1) * 0.5;
      for (int axis = 0; axis < 3; ++axis) {
        shape.centerM_[axis] = (index * 3 + axis - 384) * 0.25;
        shape.halfExtentM_[axis] = (index * 3 + axis + 1) * 0.125;
        shape.normal_[axis] = (index * 3 + axis - 192) * 0.0625;
        for (int component = 0; component < 3; ++component) {
          shape.axes_[axis][component] =
              (index * 9 + axis * 3 + component - 768) * 0.03125;
        }
      }
    }
    Kokkos::View<shape_type*, memory_space_type> shapes("shapes", count);
    Kokkos::View<shape_type*, memory_space_type> output("shape_output", count);
    Kokkos::View<shape_type*, memory_space_type> defaults("shape_defaults", count);
    // Explicit matching layouts allow deep_copy between HostSpace and CUDA.
    Kokkos::View<double * [6], Kokkos::LayoutRight, memory_space_type> checks(
        "shape_checks", count);
    Kokkos::deep_copy(shapes, hostShapes);
    Kokkos::deep_copy(checks, -999.);
    Kokkos::parallel_for(
        "shape_fields", Kokkos::RangePolicy<execution_space_type>(0, count),
        KOKKOS_LAMBDA(int const index) {
          auto const source = shapes(index);
          output(count - 1 - index) = source;
          defaults(index) = shape_type{};
          switch (source.kind_) {
            case kind_type::Invalid:
              checks(index, 0) = -1.;
              break;
            case kind_type::Universe:
              checks(index, 0) = 11.;
              break;
            case kind_type::Sphere:
              checks(index, 0) = 22.;
              break;
            case kind_type::Box:
              checks(index, 0) = 33.;
              break;
            case kind_type::SeparationPlane:
              checks(index, 0) = 44.;
              break;
          }
          double center = 0.;
          double halfExtent = 0.;
          double normal = 0.;
          double axes = 0.;
          for (int axis = 0; axis < 3; ++axis) {
            center += (axis + 1) * source.centerM_[axis];
            halfExtent += (axis + 1) * source.halfExtentM_[axis];
            normal += (axis + 1) * source.normal_[axis];
            for (int component = 0; component < 3; ++component) {
              axes += (axis * 3 + component + 1) * source.axes_[axis][component];
            }
          }
          checks(index, 1) = center;
          checks(index, 2) = source.radiusM_;
          checks(index, 3) = halfExtent;
          checks(index, 4) = axes;
          checks(index, 5) = normal;
        });
    Kokkos::fence("shape transfer complete");
    Kokkos::View<shape_type*, Kokkos::HostSpace> hostOutput("host_shape_output", count);
    Kokkos::View<shape_type*, Kokkos::HostSpace> hostDefaults("host_shape_defaults",
                                                              count);
    Kokkos::View<double * [6], Kokkos::LayoutRight, Kokkos::HostSpace> hostChecks(
        "host_shape_checks", count);
    Kokkos::deep_copy(hostOutput, output);
    Kokkos::deep_copy(hostDefaults, defaults);
    Kokkos::deep_copy(hostChecks, checks);
    for (int index = 0; index < count; ++index) {
      CAPTURE(index);
      auto const& actual = hostOutput(count - 1 - index);
      auto const& expected = hostShapes(index);
      auto const& empty = hostDefaults(index);
      REQUIRE(actual.kind_ == expected.kind_);
      REQUIRE(actual.radiusM_ == expected.radiusM_);
      REQUIRE(empty.kind_ == kind_type::Invalid);
      REQUIRE(empty.radiusM_ == 0.);
      double center = 0.;
      double halfExtent = 0.;
      double normal = 0.;
      double axes = 0.;
      for (int axis = 0; axis < 3; ++axis) {
        CAPTURE(axis);
        REQUIRE(actual.centerM_[axis] == expected.centerM_[axis]);
        REQUIRE(actual.halfExtentM_[axis] == expected.halfExtentM_[axis]);
        REQUIRE(actual.normal_[axis] == expected.normal_[axis]);
        REQUIRE(empty.centerM_[axis] == 0.);
        REQUIRE(empty.halfExtentM_[axis] == 0.);
        REQUIRE(empty.normal_[axis] == 0.);
        center += (axis + 1) * expected.centerM_[axis];
        halfExtent += (axis + 1) * expected.halfExtentM_[axis];
        normal += (axis + 1) * expected.normal_[axis];
        for (int component = 0; component < 3; ++component) {
          CAPTURE(component);
          REQUIRE(actual.axes_[axis][component] == expected.axes_[axis][component]);
          REQUIRE(empty.axes_[axis][component] == 0.);
          axes += (axis * 3 + component + 1) * expected.axes_[axis][component];
        }
      }
      REQUIRE(hostChecks(index, 0) == kindChecks[index % kindChecks.size()]);
      REQUIRE(hostChecks(index, 1) == center);
      REQUIRE(hostChecks(index, 2) == expected.radiusM_);
      REQUIRE(hostChecks(index, 3) == halfExtent);
      REQUIRE(hostChecks(index, 4) == axes);
      REQUIRE(hostChecks(index, 5) == normal);
    }
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
