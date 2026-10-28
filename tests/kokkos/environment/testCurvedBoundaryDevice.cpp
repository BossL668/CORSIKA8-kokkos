/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/CurvedBoundaryAccess.hpp>
#include <corsika/kokkos/environment/StraightBoundaryAccess.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include "CurvedBoundaryTestData.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

  namespace fixture = curved_boundary_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using tree_type = corsika::kokkos::EnvironmentTreeView<memory_space_type>;
  using access_type = corsika::kokkos::CurvedBoundaryAccess<
      corsika::kokkos::EnvironmentTreeView<memory_space_type>>;
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
    REQUIRE(actual != 0.);
    REQUIRE(std::signbit(actual) == std::signbit(expected));
    REQUIRE(std::abs(actual - expected) <=
            512 * std::numeric_limits<double>::epsilon() * std::abs(expected) +
                32 * std::numeric_limits<double>::denorm_min());
  }

  template <typename TValue>
  auto copy_to_device(std::vector<TValue> const& input) {
    Kokkos::View<TValue*, Kokkos::HostSpace> host("curved_host_input", input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      host(index) = input[index];
    }
    Kokkos::View<TValue*, memory_space_type> device("curved_device_input", input.size());
    Kokkos::deep_copy(device, host);
    return device;
  }

  void check_shapes() {
    auto const inputs = fixture::queries();
    auto const queries = copy_to_device(inputs);
    tree_type const empty;
    Kokkos::View<fixture::Outcome * [4], Kokkos::LayoutRight, memory_space_type> results(
        "curved_shape_results", inputs.size());
    Kokkos::View<unsigned int * [3], Kokkos::LayoutRight, memory_space_type> rootWitness(
        "curved_failure_root_witness", inputs.size());
    Kokkos::parallel_for(
        "per_particle_curved_shapes",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(int index) {
          auto const query = queries(index);
          auto const& ray = query.ray_;
          // Each lane constructs its own stateful adapter inside the kernel.
          access_type access(empty, ray.positionM_, ray.velocityMPerS_, ray.direction_,
                             ray.fieldT_, ray.speedMPerS_, ray.momentumKgMPerS_,
                             ray.chargeC_, ray.gyroradiusM_, ray.linear_);
          auto reference = fixture::reference(ray);
          auto const actual = access.intersectShape(query.shape_);
          auto const expected = reference.intersectShape(query.shape_);
          results(index, 0) = {actual.getEntry(), actual.getExit(),
                               actual.hasIntersections(), access.hasUnsupportedShape(),
                               access.hasIncompleteRoots()};
          results(index, 1) = {expected.getEntry(), expected.getExit(),
                               expected.hasIntersections(), reference.unsupportedShape_,
                               reference.incompleteRoots_};
          fixture::shape_type universe;
          universe.kind_ = fixture::shape_type::Kind::Universe;
          auto const later = access.intersectShape(universe);
          results(index, 2) = {later.getEntry(), later.getExit(),
                               later.hasIntersections(), access.hasUnsupportedShape(),
                               access.hasIncompleteRoots()};
          access_type const fresh(empty, ray.positionM_, ray.velocityMPerS_,
                                  ray.direction_, ray.fieldT_, ray.speedMPerS_,
                                  ray.momentumKgMPerS_, ray.chargeC_, ray.gyroradiusM_,
                                  ray.linear_);
          results(index, 3) = {0., 0., false, fresh.hasUnsupportedShape(),
                               fresh.hasIncompleteRoots()};
          if (query.category_ == fixture::Category::EmptyRoots ||
              query.category_ == fixture::Category::IncompleteRoots ||
              query.category_ == fixture::Category::NonfiniteRoots) {
            auto const coefficients = fixture::sphere_polynomial(query);
            auto const roots = polynomial_roots_reference::stableQuartic(
                coefficients.quadratic_, coefficients.linear_, coefficients.constant_);
            rootWitness(index, 0) = roots.size;
            unsigned int nonfinite = 0;
            unsigned int positive = 0;
            for (unsigned int root = 0; root < roots.size; ++root) {
              if (!polynomial_roots_reference::extendedFinite(roots.values[root])) {
                ++nonfinite;
              }
              if (roots.values[root] > 0.) { ++positive; }
            }
            rootWitness(index, 1) = nonfinite;
            rootWitness(index, 2) = positive;
          }
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    auto const hostWitness =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rootWitness);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto const& query = inputs[index];
      CAPTURE(index, query.category_, query.shape_.kind_);
      auto const& actual = host(index, 0);
      auto const& expected = host(index, 1);
      {
        INFO("Same-device fixed-production comparison");
        REQUIRE(actual.valid_ == expected.valid_);
        REQUIRE(actual.unsupported_ == expected.unsupported_);
        REQUIRE(actual.incomplete_ == expected.incomplete_);
        require_exact(actual.entry_, expected.entry_);
        require_exact(actual.exit_, expected.exit_);
      }
      {
        INFO("Cross-device fixed-production comparison");
        auto const hostExpected = fixture::reference_shape(query);
        REQUIRE(actual.valid_ == hostExpected.valid_);
        REQUIRE(actual.unsupported_ == hostExpected.unsupported_);
        REQUIRE(actual.incomplete_ == hostExpected.incomplete_);
        require_cross_device(actual.entry_, hostExpected.entry_);
        require_cross_device(actual.exit_, hostExpected.exit_);
      }
      if (query.valid_ >= 0) { REQUIRE(actual.valid_ == (query.valid_ != 0)); }
      if (query.checkFlags_) {
        REQUIRE(actual.unsupported_ == query.unsupported_);
        REQUIRE(actual.incomplete_ == query.incomplete_);
      }
      REQUIRE(host(index, 2).unsupported_ == actual.unsupported_);
      REQUIRE(host(index, 2).incomplete_ == actual.incomplete_);
      REQUIRE_FALSE(host(index, 3).unsupported_);
      REQUIRE_FALSE(host(index, 3).incomplete_);
      if (query.category_ == fixture::Category::EmptyRoots) {
        REQUIRE(hostWitness(index, 0) == 0);
        REQUIRE(hostWitness(index, 1) == 0);
      } else if (query.category_ == fixture::Category::IncompleteRoots) {
        REQUIRE(hostWitness(index, 0) == 2);
        REQUIRE(hostWitness(index, 1) == 0);
        REQUIRE(hostWitness(index, 2) == 1);
      } else if (query.category_ == fixture::Category::NonfiniteRoots) {
        REQUIRE(hostWitness(index, 0) == 1);
        REQUIRE(hostWitness(index, 1) == 1);
        REQUIRE(hostWitness(index, 2) == 1);
      }
    }
    REQUIRE(inputs.size() == 558);
  }

  struct Selection {
    double parameter_{};
    std::int32_t node_{};
    bool unsupported_{};
    bool incomplete_{};
  };

  void check_tree(bool failures) {
    auto const data = failures ? fixture::failing_tree() : fixture::ordered_tree();
    auto const nodes = copy_to_device(data.nodes_);
    auto const shapes = copy_to_device(data.shapes_);
    auto const children = copy_to_device(data.children_);
    auto const exclusions = copy_to_device(data.exclusions_);
    tree_type const tree(nodes, shapes, children, exclusions);
    auto const inputs = fixture::search_queries(failures);
    auto const queries = copy_to_device(inputs);
    Kokkos::View<Selection * [9], Kokkos::LayoutRight, memory_space_type> selections(
        "per_particle_boundaries", inputs.size());
    Kokkos::parallel_for(
        "per_particle_repeated_boundary_search",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(int index) {
          auto const query = queries(index);
          auto const& ray = query.ray_;
          access_type access(tree, ray.positionM_, ray.velocityMPerS_, ray.direction_,
                             ray.fieldT_, ray.speedMPerS_, ray.momentumKgMPerS_,
                             ray.chargeC_, ray.gyroradiusM_, ray.linear_);
          auto reference = fixture::reference(ray);
          reference.nodes_ = nodes.data();
          reference.shapes_ = shapes.data();
          reference.children_ = children.data();
          reference.exclusions_ = exclusions.data();
          corsika::kokkos::StraightBoundaryAccess<
              corsika::kokkos::EnvironmentTreeView<memory_space_type>> const
              straight(tree, ray.positionM_, ray.velocityMPerS_);
          auto const straightReference =
              static_cast<straight_boundary_reference::Tree>(reference);
          for (int search = 0; search < 2; ++search) {
            auto const current = search ? query.secondNode_ : query.firstNode_;
            auto const limit = search ? query.secondLimit_ : query.firstLimit_;
            auto const actual =
                corsika::kokkos::next_environment_boundary(access, current, limit);
            auto const expected = boundary_reference::next(reference, current, limit);
            selections(index, 2 * search) = {actual.getParameter(), actual.getNode(),
                                             access.hasUnsupportedShape(),
                                             access.hasIncompleteRoots()};
            selections(index, 2 * search + 1) = {expected.parameter_, expected.node_,
                                                 reference.unsupportedShape_,
                                                 reference.incompleteRoots_};
            auto const straightResult =
                corsika::kokkos::next_environment_boundary(straight, current, limit);
            auto const straightExpected =
                boundary_reference::next(straightReference, current, limit);
            selections(index, 4 + 2 * search) = {straightResult.getParameter(),
                                                 straightResult.getNode(), false, false};
            selections(index, 5 + 2 * search) = {straightExpected.parameter_,
                                                 straightExpected.node_, false, false};
          }
          access_type const fresh(tree, ray.positionM_, ray.velocityMPerS_,
                                  ray.direction_, ray.fieldT_, ray.speedMPerS_,
                                  ray.momentumKgMPerS_, ray.chargeC_, ray.gyroradiusM_,
                                  ray.linear_);
          selections(index, 8) = {0., 0, fresh.hasUnsupportedShape(),
                                  fresh.hasIncompleteRoots()};
        });
    auto const host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, selections);
    unsigned int stickyFailures = 0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto const& query = inputs[index];
      CAPTURE(failures, index, query.ray_.linear_, query.firstNode_, query.secondNode_);
      auto reference = fixture::reference(query.ray_);
      reference.nodes_ = data.nodes_.data();
      reference.shapes_ = data.shapes_.data();
      reference.children_ = data.children_.data();
      reference.exclusions_ = data.exclusions_.data();
      for (int search = 0; search < 2; ++search) {
        auto const& actual = host(index, 2 * search);
        auto const& expected = host(index, 2 * search + 1);
        REQUIRE(actual.node_ == expected.node_);
        REQUIRE(actual.unsupported_ == expected.unsupported_);
        REQUIRE(actual.incomplete_ == expected.incomplete_);
        require_exact(actual.parameter_, expected.parameter_);
        auto const current = search ? query.secondNode_ : query.firstNode_;
        auto const limit = search ? query.secondLimit_ : query.firstLimit_;
        auto const hostExpected = boundary_reference::next(reference, current, limit);
        REQUIRE(actual.node_ == hostExpected.node_);
        REQUIRE(actual.unsupported_ == reference.unsupportedShape_);
        REQUIRE(actual.incomplete_ == reference.incompleteRoots_);
        require_cross_device(actual.parameter_, hostExpected.parameter_);
        auto const& straight = host(index, 4 + 2 * search);
        auto const& straightExpected = host(index, 5 + 2 * search);
        REQUIRE(straight.node_ == straightExpected.node_);
        require_exact(straight.parameter_, straightExpected.parameter_);
        if (failures && !query.ray_.linear_) {
          REQUIRE(actual.unsupported_);
          REQUIRE(actual.incomplete_);
          ++stickyFailures;
        }
      }
      REQUIRE_FALSE(host(index, 8).unsupported_);
      REQUIRE_FALSE(host(index, 8).incomplete_);
    }
    REQUIRE(inputs.size() == (failures ? 64 : 300));
    REQUIRE(stickyFailures == (failures ? 64 : 0));
  }

} // namespace

TEST_CASE("Device-local curved and straight access preserve independent particle queries",
          "[curved_boundary_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    check_shapes();
    check_tree(false);
    check_tree(true);
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
