/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/utility/StableQuartic.hpp>

#include "PolynomialRootsTestData.hpp"

#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {

  static_assert(std::is_trivially_copyable_v<corsika::kokkos::ExtendedPrecision>);
  static_assert(std::is_trivially_copyable_v<corsika::kokkos::PolynomialRoots>);

  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  void require_exact(polynomial_roots_test::Result const& actual,
                     polynomial_roots_test::Result const& expected) {
    REQUIRE(actual.count_ == expected.count_);
    REQUIRE(actual.count_ <= 4);
    for (unsigned int index = 0; index < actual.count_; ++index) {
      CAPTURE(index);
      require_exact(actual.roots_[index], expected.roots_[index]);
    }
  }

} // namespace

TEST_CASE("Polynomial roots preserve the fixed production arithmetic and signals",
          "[polynomial_roots]") {
  namespace fixture = polynomial_roots_test;
  double const epsilon = std::numeric_limits<double>::epsilon();
  auto const inputs = fixture::queries();
  std::array<unsigned int, 5> branches{};
  std::array<unsigned int, 5> quarticBranches{};
  unsigned int physicalCases = 0;
  unsigned int knownCases = 0;
  unsigned int failureCases = 0;
  unsigned int relativeCases = 0;
  REQUIRE(inputs.size() == 120);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    auto const& input = inputs[index];
    CAPTURE(index, input.operation_, input.a_, input.b_, input.c_, input.epsilon_,
            input.scaleExponent_, input.sphere_);
    auto const actual = fixture::evaluate(input);
    auto const expected = fixture::reference(input);
    require_exact(actual, expected);
    auto const branch = fixture::reference_branch(input);
    if (!input.sphere_) { REQUIRE(branch == input.branch_); }
    if (input.operation_ == fixture::Operation::Resolvent) {
      ++branches[static_cast<unsigned int>(branch)];
    }
    if (input.operation_ == fixture::Operation::Quartic) {
      ++quarticBranches[static_cast<unsigned int>(branch)];
    }

    if (input.knownCount_ >= 0) {
      ++knownCases;
      REQUIRE(actual.count_ == static_cast<unsigned int>(input.knownCount_));
      for (unsigned int root = 0; root < actual.count_; ++root) {
        double const unit = std::ldexp(1., input.scaleExponent_);
        double const scale = std::max(unit, std::abs(input.knownRoots_[root]));
        REQUIRE(std::abs(actual.roots_[root] - input.knownRoots_[root]) <=
                256 * epsilon * scale);
        if (input.relativeRoots_) {
          REQUIRE(actual.roots_[root] != 0.);
          REQUIRE(std::signbit(actual.roots_[root]) ==
                  std::signbit(input.knownRoots_[root]));
          REQUIRE(std::abs((actual.roots_[root] - input.knownRoots_[root]) /
                           input.knownRoots_[root]) <= 64 * epsilon);
        }
      }
    }
    if (input.relativeRoots_) { ++relativeCases; }
    if (input.failure_) {
      ++failureCases;
      REQUIRE(actual.count_ == 1);
      REQUIRE(std::isinf(actual.roots_[0]));
      REQUIRE(actual.roots_[0] > 0.);
    } else if (input.operation_ == fixture::Operation::Quartic) {
      for (unsigned int root = 0; root < actual.count_; ++root) {
        REQUIRE(std::isfinite(actual.roots_[root]));
        REQUIRE(fixture::scaled_residual(input, actual.roots_[root]) <= 512 * epsilon);
        if (root) { REQUIRE(actual.roots_[root - 1] <= actual.roots_[root]); }
      }
    }
    if (input.sphere_) {
      ++physicalCases;
      REQUIRE(actual.count_ >= 2);
    }
  }
  REQUIRE(knownCases == 65);
  REQUIRE(relativeCases == 2);
  REQUIRE(physicalCases == 16);
  REQUIRE(failureCases == 1);
  for (unsigned int branch = 1; branch < branches.size(); ++branch) {
    CAPTURE(branch);
    INFO(
        "Complete quartic queries in this resolvent branch: " << quarticBranches[branch]);
    REQUIRE(branches[branch] == 2);
    REQUIRE(quarticBranches[branch] > 0);
  }

  auto const distinctInputs = fixture::distinct_queries();
  REQUIRE(distinctInputs.size() == 20);
  for (std::size_t index = 0; index < distinctInputs.size(); ++index) {
    CAPTURE(index);
    auto const actual = fixture::evaluate(distinctInputs[index]);
    require_exact(actual, fixture::reference(distinctInputs[index]));
    require_exact(actual, distinctInputs[index].expected_);
  }

  auto const extendedInputs = fixture::extended_queries();
  REQUIRE(extendedInputs.size() == 11);
  for (std::size_t index = 0; index < extendedInputs.size(); ++index) {
    auto const& input = extendedInputs[index];
    CAPTURE(index, input.firstHigh_, input.firstLow_, input.secondHigh_,
            input.secondLow_);
    auto const actual = fixture::evaluate(input);
    auto const expected = fixture::reference(input);
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

  // Independent cancellation check: binary64 rounds both high words to one,
  // whereas the two-double subtraction retains the low-word difference.
  corsika::kokkos::ExtendedPrecision const first(1., std::ldexp(1., -54));
  corsika::kokkos::ExtendedPrecision const second(1., -std::ldexp(1., -54));
  require_exact(static_cast<double>(first - second), std::ldexp(1., -53));
  corsika::kokkos::ExtendedPrecision const zero;
  require_exact(zero.getHigh(), 0.);
  require_exact(zero.getLow(), 0.);
  corsika::kokkos::PolynomialRoots const empty;
  REQUIRE(empty.getSize() == 0);
  for (unsigned int index = 0; index < 4; ++index) { require_exact(empty[index], 0.); }
}
