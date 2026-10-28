/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentQuery.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include "EnvironmentQueryTestData.hpp"

#include <catch2/catch_all.hpp>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {
  namespace fixture = environment_query_test;
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
} // namespace

TEST_CASE(
    "Actual execution space queries NodeTree data and advances native atmosphere "
    "particles",
    "[kokkos_environment_query_device]") {
  corsika::kokkos::Runtime runtime;
  auto datasets = fixture::datasets();
  datasets.push_back(fixture::atmosphere());
  auto const state = [](auto actual, auto expected) { REQUIRE(actual == expected); };
  auto const check = [](bool condition) { REQUIRE(condition); };
  std::size_t count{}, crossCount{}, integrationCount{};
  for (auto const& data : datasets) {
    INFO(data.name_);
    fixture::Uploaded<memory_space_type> const uploaded(data);
    auto const view = uploaded.view();
    auto const reference = uploaded.referenceView();
    auto const queries = fixture::Uploaded<memory_space_type>::upload(data.queries_);
    Kokkos::View<fixture::Outcome * [2], Kokkos::LayoutRight, memory_space_type> results(
        "environment_query_results", data.queries_.size());
    Kokkos::parallel_for(
        "independent_environment_queries",
        Kokkos::RangePolicy<execution_space_type>(0, data.queries_.size()),
        KOKKOS_LAMBDA(int index) {
          auto const query = queries(index);
          results(index, 0) = fixture::actual(view, query);
          results(index, 1) = fixture::expected(reference, query);
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    auto const inputs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, queries);
    auto const hostReference = fixture::reference_view(data);
    for (std::size_t index = 0; index < data.queries_.size(); ++index) {
      CAPTURE(index);
      {
        INFO(
            "Same-device immutable production oracle: exact finite/signed-zero outputs, "
            "nonfinite classes");
        fixture::compare(host(index, 0), host(index, 1), require_exact, state);
      }
      fixture::compare(inputs(index), data.queries_[index], require_exact, state);
      fixture::check_semantics(data.queries_[index], host(index, 0), check);
      if (data.queries_[index].crossDevice_) {
        INFO(
            "Host-versus-device finite well-conditioned subset, all states remain exact");
        fixture::compare(host(index, 0),
                         fixture::expected(hostReference, data.queries_[index]),
                         require_cross_device, state);
        ++crossCount;
      }
      ++count;
    }

    if (!data.integration_.empty()) {
      REQUIRE(data.queries_.size() == data.integration_.size());
    }
    auto const integration =
        fixture::Uploaded<memory_space_type>::upload(data.integration_);
    Kokkos::View<fixture::IntegrationOutcome * [2], Kokkos::LayoutRight,
                 memory_space_type>
        chains("native_atmosphere_chains", data.integration_.size());
    Kokkos::parallel_for(
        "native_atmosphere_to_per_particle_advance",
        Kokkos::RangePolicy<execution_space_type>(0, data.integration_.size()),
        KOKKOS_LAMBDA(int index) {
          // Each lane independently consumes its exported model and particle.
          // The original caller's Linear -> zero-field choice is kept explicit.
          auto const input = integration(index);
          // Compare the same stored input, including the observation plane.
          // Keep input preparation outside the independently inlined chains:
          // diagnostics found context-dependent rounding before materialization.
          chains(index, 0) = fixture::actual_integration(view, input, queries(index));
          chains(index, 1) =
              fixture::expected_integration(reference, input, queries(index));
        });
    auto const hostChains =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, chains);
    auto const integrationInputs =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, integration);
    for (std::size_t index = 0; index < data.integration_.size(); ++index) {
      CAPTURE(index);
      fixture::compare(data.queries_[index],
                       fixture::integration_query(data.integration_[index]),
                       require_exact, state);
      {
        INFO("Same-device independent query/limit/boundary/X/L/advance chain");
        fixture::compare(hostChains(index, 0), hostChains(index, 1), require_exact,
                         state);
      }
      {
        INFO("Host-versus-device real atmosphere fixed-input chain");
        fixture::compare(
            hostChains(index, 0),
            fixture::expected_integration(hostReference, data.integration_[index],
                                          data.queries_[index]),
            require_cross_device, state);
      }
      fixture::check_integration(data.integration_[index], hostChains(index, 0), check);
      fixture::compare(integrationInputs(index), data.integration_[index], require_exact,
                       state);
      ++integrationCount;
    }
    fixture::check_storage(data, uploaded, require_exact, state);
  }
  CAPTURE(count, crossCount, integrationCount);
  REQUIRE(count == 160);
  REQUIRE(crossCount == 34);
  REQUIRE(integrationCount == 15);
  Kokkos::fence();
}
