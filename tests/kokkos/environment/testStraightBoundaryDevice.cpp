/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/StraightBoundaryAccess.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "StraightBoundaryReference.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using tree_type = corsika::kokkos::EnvironmentTreeView<memory_space_type>;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif

  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_cross_device(double actual, double expected) {
    if (!std::isfinite(expected) || expected == 0.) {
      require_exact(actual, expected);
      return;
    }
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::abs(actual - expected) <=
            128 * std::numeric_limits<double>::epsilon() * std::abs(expected) +
                32 * std::numeric_limits<double>::denorm_min());
  }

  template <typename TValue>
  auto copy_to_device(std::vector<TValue> const& values) {
    Kokkos::View<TValue*, Kokkos::HostSpace> host("boundary_host_input", values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
      host(index) = values[index];
    }
    Kokkos::View<TValue*, memory_space_type> device("boundary_device_input",
                                                    values.size());
    Kokkos::deep_copy(device, host);
    return device;
  }

  void check_shapes() {
    auto const inputs = straight_boundary_reference::queries();
    auto const queries = copy_to_device(inputs);
    int const count = static_cast<int>(inputs.size());
    Kokkos::View<double * [6], Kokkos::LayoutRight, memory_space_type> results(
        "shape_intersections", count);
    Kokkos::parallel_for(
        "straight_shape_intersections",
        Kokkos::RangePolicy<execution_space_type>(0, count), KOKKOS_LAMBDA(int index) {
          auto const query = queries(index);
          auto const actual = corsika::kokkos::straight_intersect(
              query.shape_, query.position_, query.velocity_);
          auto const expected = straight_boundary_reference::intersect(
              query.shape_, query.position_, query.velocity_);
          results(index, 0) = actual.hasIntersections();
          results(index, 1) = actual.getEntry();
          results(index, 2) = actual.getExit();
          results(index, 3) = expected.hasIntersections();
          results(index, 4) = expected.getEntry();
          results(index, 5) = expected.getExit();
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (int index = 0; index < count; ++index) {
      auto const& query = inputs[index];
      CAPTURE(index, query.shape_.kind_, query.position_[0], query.position_[1],
              query.position_[2], query.velocity_[0], query.velocity_[1],
              query.velocity_[2]);
      require_exact(host(index, 0), host(index, 3));
      require_exact(host(index, 1), host(index, 4));
      require_exact(host(index, 2), host(index, 5));
      auto const expected = straight_boundary_reference::intersect(
          query.shape_, query.position_, query.velocity_);
      REQUIRE(host(index, 0) == expected.hasIntersections());
      require_cross_device(host(index, 1), expected.getEntry());
      require_cross_device(host(index, 2), expected.getExit());
    }
  }

  void check_ordering() {
    auto const inputs = boundary_reference::ordered_cases();
    auto const queries = copy_to_device(inputs);
    int const count = static_cast<int>(inputs.size());
    Kokkos::View<double * [2], Kokkos::LayoutRight, memory_space_type> parameters(
        "ordered_parameters", count);
    Kokkos::View<std::int32_t * [2], Kokkos::LayoutRight, memory_space_type> nodes(
        "ordered_nodes", count);
    Kokkos::parallel_for(
        "ordered_boundary_cases", Kokkos::RangePolicy<execution_space_type>(0, count),
        KOKKOS_LAMBDA(int index) {
          auto const input = queries(index);
          auto const actual = corsika::kokkos::next_environment_boundary(
              input, std::int32_t{0}, input.limit_);
          auto const expected =
              boundary_reference::next(input, std::int32_t{0}, input.limit_);
          parameters(index, 0) = actual.getParameter();
          parameters(index, 1) = expected.parameter_;
          nodes(index, 0) = actual.getNode();
          nodes(index, 1) = expected.node_;
        });
    auto const hostParameters =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, parameters);
    auto const hostNodes =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nodes);
    for (int index = 0; index < count; ++index) {
      CAPTURE(inputs[index].name_);
      require_exact(hostParameters(index, 0), hostParameters(index, 1));
      require_exact(hostParameters(index, 0), inputs[index].expectedParameter_);
      REQUIRE(hostNodes(index, 0) == hostNodes(index, 1));
      REQUIRE(hostNodes(index, 0) == inputs[index].expectedNode_);
    }
  }

  void check_tree() {
    auto const input = straight_boundary_reference::tree_fixture();
    auto const nodes = copy_to_device(input.nodes_);
    auto const shapes = copy_to_device(input.shapes_);
    auto const children = copy_to_device(input.children_);
    auto const exclusions = copy_to_device(input.exclusions_);
    tree_type const tree(nodes, shapes, children, exclusions);
    std::vector<double> const limits{0., .5, 5., 100.,
                                     std::numeric_limits<double>::infinity()};
    auto const deviceLimits = copy_to_device(limits);
    for (auto const& ray : straight_boundary_reference::rays()) {
      // The adapter is host-constructed, then safely copied into the kernel.
      corsika::kokkos::StraightBoundaryAccess<
          corsika::kokkos::EnvironmentTreeView<memory_space_type>> const
          access(tree, ray.position_, ray.velocity_);
      straight_boundary_reference::Tree reference{nodes.data(), shapes.data(),
                                                  children.data(), exclusions.data()};
      straight_boundary_reference::Tree hostReference{
          input.nodes_.data(), input.shapes_.data(), input.children_.data(),
          input.exclusions_.data()};
      for (int axis = 0; axis < 3; ++axis) {
        reference.position_[axis] = hostReference.position_[axis] = ray.position_[axis];
        reference.velocity_[axis] = hostReference.velocity_[axis] = ray.velocity_[axis];
      }
      Kokkos::View<double * [2], Kokkos::LayoutRight, memory_space_type> parameters(
          "tree_parameters", 30);
      Kokkos::View<std::int32_t * [2], Kokkos::LayoutRight, memory_space_type> selected(
          "tree_selection", 30);
      Kokkos::parallel_for(
          "straight_tree_boundaries", Kokkos::RangePolicy<execution_space_type>(0, 30),
          KOKKOS_LAMBDA(int index) {
            std::int32_t const current = index / 5;
            double const limit = deviceLimits(index % 5);
            auto const actual =
                corsika::kokkos::next_environment_boundary(access, current, limit);
            auto const expected = boundary_reference::next(reference, current, limit);
            parameters(index, 0) = actual.getParameter();
            parameters(index, 1) = expected.parameter_;
            selected(index, 0) = actual.getNode();
            selected(index, 1) = expected.node_;
          });
      auto const hostParameters =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, parameters);
      auto const hostNodes =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, selected);
      for (int index = 0; index < 30; ++index) {
        CAPTURE(index, ray.position_[0], ray.position_[1], ray.position_[2]);
        auto const expected = boundary_reference::next(
            hostReference, std::int32_t{index / 5}, limits[index % 5]);
        require_exact(hostParameters(index, 0), hostParameters(index, 1));
        require_cross_device(hostParameters(index, 0), expected.parameter_);
        REQUIRE(hostNodes(index, 0) == hostNodes(index, 1));
        REQUIRE(hostNodes(index, 0) == expected.node_);
      }
    }
  }

} // namespace

TEST_CASE("Device straight boundaries preserve ordered production geometry",
          "[straight_boundary_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    check_shapes();
    check_ordering();
    check_tree();
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
