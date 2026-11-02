/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/Runtime.hpp>
#include <corsika/kokkos/utility/ElementaryFunctions.hpp>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

TEST_CASE("Elementary functions preserve signed zeros and the host math contract",
          "[elementary_functions_device]") {
  corsika::kokkos::Runtime runtime;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  INFO("Actual execution space: " << execution_space_type::name());
  double const infinity = std::numeric_limits<double>::infinity();
  std::array<double, 15> const inputs{
      -0.,       0.,       -1.e-16,
      1.e-16,    -1.e-8,   1.e-8,
      -.5,       .5,       std::nextafter(-1., 0.),
      -1.,       -2.,      1000.,
      -infinity, infinity, std::numeric_limits<double>::quiet_NaN()};
  Kokkos::View<double*, memory_space_type> values("elementary_inputs", inputs.size());
  auto hostInputs = Kokkos::create_mirror_view(values);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    hostInputs(index) = inputs[index];
  }
  Kokkos::deep_copy(values, hostInputs);
  Kokkos::View<double* [2], memory_space_type> results("elementary_results",
                                                       inputs.size());
  Kokkos::parallel_for(
      "elementary_functions", Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
      KOKKOS_LAMBDA(int const index) {
        results(index, 0) = corsika::kokkos::logarithm_one_plus(values(index));
        results(index, 1) = corsika::kokkos::exponential_minus_one(values(index));
      });
  auto const hostResults =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    std::array<double, 2> const expected{std::log1p(inputs[index]),
                                         std::expm1(inputs[index])};
    for (int operation = 0; operation < 2; ++operation) {
      double const actual = hostResults(index, operation);
      double const reference = expected[operation];
      CAPTURE(index, operation, inputs[index], actual, reference);
      if (std::isnan(reference)) {
        REQUIRE(std::isnan(actual));
      } else if (reference == 0. || std::isinf(reference)) {
        REQUIRE(actual == reference);
        REQUIRE(std::signbit(actual) == std::signbit(reference));
      } else {
        REQUIRE(std::isfinite(actual));
        REQUIRE(std::abs(actual - reference) <=
                64 * std::numeric_limits<double>::epsilon() * std::abs(reference));
      }
    }
  }
}
