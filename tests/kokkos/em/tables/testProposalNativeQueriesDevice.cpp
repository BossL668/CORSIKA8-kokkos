/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "NativeTableFixture.hpp"
#include "ProposalNativeQueryTestData.hpp"
#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include <catch2/catch_all.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

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
      if (expected == 0.) REQUIRE(std::signbit(actual) == std::signbit(expected));
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
            512. * std::numeric_limits<double>::epsilon() * std::abs(expected) +
                32. * std::numeric_limits<double>::denorm_min());
  }
} // namespace

TEST_CASE("Actual execution space matches the fixed native query oracle",
          "[kokkos_native_queries_device]") {
  namespace fixture = native_query_test;
  namespace current = corsika::kokkos;
  current::Runtime runtime;
  auto const table = current::test::make_native_tables();
  current::NativeTableStorage<execution_space_type> storage;
  storage.initialize(table);
  fixture::UploadedReference<memory_space_type> const oldStorage(table);
  auto const view = storage.getDeviceView();
  auto const reference = oldStorage.view();
  auto const queries = fixture::queries(table);
  auto const input = fixture::UploadedReference<memory_space_type>::upload(queries);
  Kokkos::View<fixture::Outcome * [2], Kokkos::LayoutRight, memory_space_type> output(
      "native_query_outcomes", queries.size());
  Kokkos::parallel_for(
      "native_queries_against_frozen_production",
      Kokkos::RangePolicy<execution_space_type>(0, queries.size()),
      KOKKOS_LAMBDA(std::size_t index) {
        // One materialized input is shared, not the numerical result or helpers.
        auto const query = input(index);
        output(index, 0) = fixture::evaluate(view, query);
        output(index, 1) = fixture::expected(reference, query);
      });
  Kokkos::fence("native queries completed before comparing results");
  auto const results = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
  fixture::HostReference const hostOwner(table);
  auto const hostReference = hostOwner.view(table);
  auto const state = [](auto actual, auto expected) { REQUIRE(actual == expected); };
  std::array<std::size_t, static_cast<std::size_t>(fixture::Operation::Count)> calls{};
  std::size_t crossCount{}, newton{}, bisection{};
  for (std::size_t index = 0; index < queries.size(); ++index) {
    auto const& query = queries[index];
    INFO("Query " << index << ", operation " << static_cast<unsigned>(query.operation_)
                  << ", PID " << query.pid_ << ", process " << query.process_
                  << ", energy " << query.energy_ << ", argument " << query.argument_
                  << ", split " << query.photonSplit_);
    {
      INFO(
          "Same-device independently evaluated production oracle; exact states and "
          "iterations");
      fixture::compare(results(index, 0), results(index, 1), require_exact, state);
    }
    ++calls[static_cast<std::size_t>(query.operation_)];
    newton += results(index, 0).newton_;
    bisection += results(index, 0).bisection_;
    if (query.crossDevice_) {
      INFO(
          "Cross-device well-conditioned rates only; not inverse-solver iteration "
          "identity");
      fixture::compare(results(index, 0), fixture::expected(hostReference, query),
                       require_cross_device, state, false);
      ++crossCount;
    }
  }
  for (auto count : calls) REQUIRE(count > 0);
  REQUIRE(crossCount > 0);
  REQUIRE(newton > 0);
  REQUIRE(bisection > 0);
  REQUIRE(storage.getSourceContentHash() == table.contentHash_);
  REQUIRE(current::calculate_proposal_native_hash(table) == table.contentHash_);
}
