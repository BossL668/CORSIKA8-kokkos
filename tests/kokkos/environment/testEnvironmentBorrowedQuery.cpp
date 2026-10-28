/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <tests/kokkos/environment/EnvironmentQueryTestData.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cmath>

namespace {
  namespace cp = corsika::kokkos;
  namespace fixture = environment_query_test;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;

  void ensure_runtime() { static cp::Runtime runtime; }

  void exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }

  cp::EnvironmentDataView borrow(fixture::Dataset const& data) {
    return {data.nodes_.data(),
            data.shapes_.data(),
            data.children_.data(),
            data.exclusions_.data(),
            data.models_.data(),
            data.components_.data(),
            static_cast<std::uint32_t>(data.nodes_.size()),
            static_cast<std::uint32_t>(data.children_.size()),
            static_cast<std::uint32_t>(data.exclusions_.size()),
            static_cast<std::uint32_t>(data.models_.size()),
            static_cast<std::uint32_t>(data.components_.size())};
  }
} // namespace

TEST_CASE("Borrowed and managed environment queries retain production results",
          "[kokkos][environment][borrowed]") {
  ensure_runtime();
  auto datasets = fixture::datasets();
  datasets.push_back(fixture::atmosphere());
  auto const state = [](auto actual, auto expected) { REQUIRE(actual == expected); };
  auto const check = [](bool condition) { REQUIRE(condition); };
  std::size_t queryCount{}, chainCount{};
  for (auto const& data : datasets) {
    INFO(data.name_);
    cp::EnvironmentStorage<execution_space_type> owner(borrow(data));
    auto managed = owner.getView();
    Kokkos::View<cp::EnvironmentBorrowedView, memory_space_type> context(
        "borrowed environment context");
    Kokkos::deep_copy(context, cp::EnvironmentBorrowedView(owner.getDataView()));
    fixture::Uploaded<memory_space_type> oracle(data);
    auto reference = oracle.referenceView();
    auto queries = fixture::Uploaded<memory_space_type>::upload(data.queries_);
    Kokkos::View<fixture::Outcome * [3], Kokkos::LayoutRight, memory_space_type> results(
        "borrowed environment results", data.queries_.size());
    Kokkos::parallel_for(
        "query borrowed context",
        Kokkos::RangePolicy<execution_space_type>(0, data.queries_.size()),
        KOKKOS_LAMBDA(int i) {
          results(i, 0) = fixture::actual(context(), queries(i));
          results(i, 1) = fixture::actual(managed, queries(i));
          results(i, 2) = fixture::expected(reference, queries(i));
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (std::size_t i = 0; i < data.queries_.size(); ++i) {
      CAPTURE(i);
      fixture::compare(host(i, 0), host(i, 1), exact, state);
      fixture::compare(host(i, 0), host(i, 2), exact, state);
      fixture::check_semantics(data.queries_[i], host(i, 0), check);
      ++queryCount;
    }

    if (!data.integration_.empty()) {
      REQUIRE(data.integration_.size() == data.queries_.size());
    }
    auto inputs = fixture::Uploaded<memory_space_type>::upload(data.integration_);
    Kokkos::View<fixture::IntegrationOutcome * [3], Kokkos::LayoutRight,
                 memory_space_type>
        chains("borrowed environment chains", data.integration_.size());
    Kokkos::parallel_for(
        "borrowed atmosphere to magnetic advance",
        Kokkos::RangePolicy<execution_space_type>(0, data.integration_.size()),
        KOKKOS_LAMBDA(int i) {
          auto input = inputs(i);
          chains(i, 0) = fixture::actual_integration(context(), input, queries(i));
          chains(i, 1) = fixture::actual_integration(managed, input, queries(i));
          chains(i, 2) = fixture::expected_integration(reference, input, queries(i));
        });
    auto hostChains = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, chains);
    for (std::size_t i = 0; i < data.integration_.size(); ++i) {
      CAPTURE(i);
      fixture::compare(hostChains(i, 0), hostChains(i, 1), exact, state);
      fixture::compare(hostChains(i, 0), hostChains(i, 2), exact, state);
      fixture::check_integration(data.integration_[i], hostChains(i, 0), check);
      ++chainCount;
    }
  }
  REQUIRE(queryCount == 160);
  REQUIRE(chainCount == 15);
}

TEST_CASE("Empty borrowed environment reports the existing invalid status",
          "[kokkos][environment][borrowed]") {
  ensure_runtime();
  Kokkos::View<int*, memory_space_type> result("empty borrowed query", 2);
  Kokkos::parallel_for(
      "empty borrowed query", Kokkos::RangePolicy<execution_space_type>(0, 1),
      KOKKOS_LAMBDA(int) {
        cp::EnvironmentBorrowedView const view;
        double position[3]{};
        auto query = cp::query_environment(view, cp::EnvironmentQueryConfig{}, position,
                                           nullptr, -3);
        result(0) = static_cast<int>(query.status_);
        result(1) = view.getModel(0) == nullptr && !view.hasNodeStorage() &&
                    view.hasAdjacencyStorage();
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
  REQUIRE(host(0) == static_cast<int>(cp::EnvironmentStatus::InvalidSnapshot));
  REQUIRE(host(1) == 1);
}
