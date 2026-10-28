/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/Runtime.hpp>
#include <corsika/kokkos/utility/StableQuartic.hpp>

#include "PolynomialRootsTestData.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
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

  template <typename TValue>
  auto copy_to_device(std::vector<TValue> const& input) {
    Kokkos::View<TValue*, Kokkos::HostSpace> host("root_test_host_input", input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      host(index) = input[index];
    }
    Kokkos::View<TValue*, memory_space_type> device("root_test_device_input",
                                                    input.size());
    Kokkos::deep_copy(device, host);
    return device;
  }

  void check_roots() {
    namespace fixture = polynomial_roots_test;
    double const epsilon = std::numeric_limits<double>::epsilon();
    auto const inputs = fixture::queries();
    auto const deviceInputs = copy_to_device(inputs);
    int const count = static_cast<int>(inputs.size());
    Kokkos::View<fixture::Result * [2], Kokkos::LayoutRight, memory_space_type> results(
        "root_results", count);
    Kokkos::View<fixture::Branch*, memory_space_type> branches("resolvent_branches",
                                                               count);
    Kokkos::parallel_for(
        "root_production_equivalence",
        Kokkos::RangePolicy<execution_space_type>(0, count), KOKKOS_LAMBDA(int index) {
          auto const query = deviceInputs(index);
          results(index, 0) = fixture::evaluate(query);
          results(index, 1) = fixture::reference(query);
          branches(index) = fixture::reference_branch(query);
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    auto const hostBranches =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, branches);
    std::array<double, 5> maximumScaledDifference{};
    std::array<double, 5> maximumRelativeDifference{};
    std::array<unsigned int, 5> branchCounts{};
    std::array<unsigned int, 5> quarticBranchCounts{};
    unsigned int sphereCases = 0;
    unsigned int relativeCases = 0;
    REQUIRE(count == 120);
    for (int index = 0; index < count; ++index) {
      auto const& query = inputs[index];
      CAPTURE(index, query.operation_, query.a_, query.b_, query.c_, query.scaleExponent_,
              query.sphere_);
      auto const& actual = host(index, 0);
      auto const& expected = host(index, 1);
      auto const hostExpected = fixture::reference(query);
      REQUIRE(actual.count_ <= 4);
      // Multiplicity/order/failure status are never weakened for cross-backend
      // comparisons. A mismatch is a failed acceptance test, not a large error bar.
      REQUIRE(actual.count_ == expected.count_);
      REQUIRE(actual.count_ == hostExpected.count_);
      REQUIRE(hostBranches(index) == fixture::reference_branch(query));
      if (!query.sphere_) { REQUIRE(hostBranches(index) == query.branch_); }
      if (query.operation_ == fixture::Operation::Resolvent) {
        ++branchCounts[static_cast<unsigned int>(hostBranches(index))];
      }
      if (query.operation_ == fixture::Operation::Quartic) {
        ++quarticBranchCounts[static_cast<unsigned int>(hostBranches(index))];
      }
      unsigned int const operation = static_cast<unsigned int>(query.operation_);
      for (unsigned int root = 0; root < actual.count_; ++root) {
        CAPTURE(root);
        double const value = actual.roots_[root];
        double const comparison = hostExpected.roots_[root];
        {
          INFO("Comparison: device implementation versus device frozen reference");
          require_exact(value, expected.roots_[root]);
        }
        INFO("Comparison: device implementation versus host frozen reference");
        if (!std::isfinite(comparison)) {
          require_exact(value, comparison);
        } else {
          REQUIRE(std::isfinite(value));
          if (comparison == 0.) {
            require_exact(value, comparison);
          } else {
            REQUIRE(value != 0.);
            REQUIRE(std::signbit(value) == std::signbit(comparison));
            double const coefficientScale = std::ldexp(1., query.scaleExponent_);
            double const scale = std::max(std::abs(comparison), coefficientScale);
            double const difference = std::abs(value - comparison);
            double const scaledDifference = difference / scale;
            double const relativeDifference = difference / std::abs(comparison);
            maximumScaledDifference[operation] =
                std::max(maximumScaledDifference[operation], scaledDifference);
            maximumRelativeDifference[operation] =
                std::max(maximumRelativeDifference[operation], relativeDifference);
            REQUIRE(scaledDifference <= 256 * epsilon);
            if (query.relativeRoots_) { REQUIRE(relativeDifference <= 64 * epsilon); }
          }
        }
        if (query.operation_ == fixture::Operation::Quartic && !query.failure_) {
          REQUIRE(std::isfinite(value));
          REQUIRE(fixture::scaled_residual(query, value) <= 512 * epsilon);
          if (root) { REQUIRE(actual.roots_[root - 1] <= value); }
        }
        if (query.knownCount_ >= 0) {
          double const scale = std::max(std::ldexp(1., query.scaleExponent_),
                                        std::abs(query.knownRoots_[root]));
          REQUIRE(std::abs(value - query.knownRoots_[root]) <= 256 * epsilon * scale);
          if (query.relativeRoots_) {
            REQUIRE(std::abs((value - query.knownRoots_[root]) /
                             query.knownRoots_[root]) <= 64 * epsilon);
          }
        }
      }
      if (query.knownCount_ >= 0) {
        REQUIRE(actual.count_ == static_cast<unsigned int>(query.knownCount_));
      }
      if (query.failure_) {
        REQUIRE(actual.count_ == 1);
        REQUIRE(std::isinf(actual.roots_[0]));
        REQUIRE(actual.roots_[0] > 0.);
      }
      if (query.sphere_) {
        ++sphereCases;
        REQUIRE(actual.count_ >= 2);
      }
      if (query.relativeRoots_) { ++relativeCases; }
    }
    REQUIRE(sphereCases == 16);
    REQUIRE(relativeCases == 2);
    for (unsigned int branch = 1; branch < branchCounts.size(); ++branch) {
      CAPTURE(branch);
      INFO("Complete quartic queries in this resolvent branch: "
           << quarticBranchCounts[branch]);
      REQUIRE(branchCounts[branch] == 2);
      REQUIRE(quarticBranchCounts[branch] > 0);
    }
    // Record both normalizations; do not present coefficient-scale errors as
    // relative errors of a tiny root. The host oracle uses long double; CUDA
    // uses the frozen two-double implementation and double transcendental calls.
    for (unsigned int operation = 0; operation < maximumScaledDifference.size();
         ++operation) {
      CAPTURE(operation);
      INFO("Host versus device maximum scaled difference: "
           << maximumScaledDifference[operation]
           << "; maximum relative difference: " << maximumRelativeDifference[operation]);
      REQUIRE(maximumScaledDifference[operation] <= 256 * epsilon);
    }
  }

  void check_distinct() {
    namespace fixture = polynomial_roots_test;
    auto const inputs = fixture::distinct_queries();
    auto const deviceInputs = copy_to_device(inputs);
    int const count = static_cast<int>(inputs.size());
    Kokkos::View<fixture::Result * [2], Kokkos::LayoutRight, memory_space_type> results(
        "distinct_root_results", count);
    Kokkos::parallel_for(
        "root_deduplication_boundary",
        Kokkos::RangePolicy<execution_space_type>(0, count), KOKKOS_LAMBDA(int index) {
          auto const query = deviceInputs(index);
          results(index, 0) = fixture::evaluate(query);
          results(index, 1) = fixture::reference(query);
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    REQUIRE(count == 20);
    for (int index = 0; index < count; ++index) {
      CAPTURE(index);
      auto const& actual = host(index, 0);
      auto const& expected = host(index, 1);
      REQUIRE(actual.count_ <= 4);
      REQUIRE(actual.count_ == expected.count_);
      REQUIRE(actual.count_ == inputs[index].expected_.count_);
      for (unsigned int root = 0; root < actual.count_; ++root) {
        CAPTURE(root);
        require_exact(actual.roots_[root], expected.roots_[root]);
        require_exact(actual.roots_[root], inputs[index].expected_.roots_[root]);
      }
    }
  }

  void check_extended() {
    namespace fixture = polynomial_roots_test;
    auto const inputs = fixture::extended_queries();
    auto const deviceInputs = copy_to_device(inputs);
    int const count = static_cast<int>(inputs.size());
    Kokkos::View<fixture::ExtendedResult * [2], Kokkos::LayoutRight, memory_space_type>
        results("extended_results", count);
    Kokkos::parallel_for(
        "extended_production_equivalence",
        Kokkos::RangePolicy<execution_space_type>(0, count), KOKKOS_LAMBDA(int index) {
          auto const query = deviceInputs(index);
          results(index, 0) = fixture::evaluate(query);
          results(index, 1) = fixture::reference(query);
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    REQUIRE(count == 11);
    for (int index = 0; index < count; ++index) {
      CAPTURE(index);
      auto const& actual = host(index, 0);
      auto const& expected = host(index, 1);
      for (int operation = 0; operation < 10; ++operation) {
        CAPTURE(operation);
        require_exact(actual.high_[operation], expected.high_[operation]);
        require_exact(actual.low_[operation], expected.low_[operation]);
        require_exact(actual.rounded_[operation], expected.rounded_[operation]);
      }
      REQUIRE(actual.less_ == expected.less_);
      REQUIRE(actual.greater_ == expected.greater_);
      REQUIRE(actual.equal_ == expected.equal_);
      REQUIRE(actual.finite_ == expected.finite_);
    }
  }

} // namespace

TEST_CASE("Device polynomial roots preserve the fixed production arithmetic",
          "[polynomial_roots_device]") {
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    corsika::kokkos::Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    check_roots();
    check_distinct();
    check_extended();
  }
  REQUIRE_FALSE(Kokkos::is_initialized());
  REQUIRE(Kokkos::is_finalized());
}
