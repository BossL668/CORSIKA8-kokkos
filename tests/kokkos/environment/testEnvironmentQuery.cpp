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
#include <array>
#include <cmath>
#include <limits>

namespace {
  namespace fixture = environment_query_test;
  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) { REQUIRE(std::signbit(actual) == std::signbit(expected)); }
    }
  }
} // namespace

TEST_CASE(
    "NodeTree environment queries preserve production contracts and native atmosphere "
    "inputs",
    "[kokkos_environment_query]") {
  // CTest supplies the OpenMP thread count; CUDA/Serial must not request it.
  corsika::kokkos::Runtime runtime;
  auto datasets = fixture::datasets();
  auto atmosphere = fixture::atmosphere();
  REQUIRE(atmosphere.nodes_.size() == 6);
  REQUIRE(atmosphere.models_.size() == 5);
  REQUIRE(atmosphere.integration_.size() == 15);
  datasets.push_back(atmosphere);
  std::array<unsigned int, static_cast<unsigned int>(fixture::Category::Count)>
      categories{};
  std::array<unsigned int, 7> statuses{};
  std::array<unsigned int, 4> planeStatuses{};
  auto const state = [](auto actual, auto expected) { REQUIRE(actual == expected); };
  auto const check = [](bool condition) { REQUIRE(condition); };
  std::size_t count{};
  unsigned int homogeneousNanWitness{}, negativeZeroWitness{};
  for (auto const& data : datasets) {
    INFO(data.name_);
    fixture::Uploaded<Kokkos::HostSpace> const uploaded(data);
    auto const view = uploaded.view();
    auto const reference = fixture::reference_view(data);
    REQUIRE(view.getModelCount() == data.models_.size());
    REQUIRE(view.getComponentCount() == data.components_.size());
    REQUIRE(view.hasModelStorage() == (uploaded.models_.data() != nullptr));
    for (std::size_t index = 0; index < data.queries_.size(); ++index) {
      CAPTURE(index);
      auto query = data.queries_[index];
      auto const actual = fixture::actual(view, query);
      auto const expected = fixture::expected(reference, query);
      fixture::compare(actual, expected, require_exact, state);
      fixture::compare(query, data.queries_[index], require_exact, state);
      fixture::check_semantics(query, actual, check);
      ++categories[static_cast<unsigned int>(query.category_)];
      ++statuses[static_cast<unsigned int>(actual.layer_.status_)];
      ++statuses[static_cast<unsigned int>(actual.boundary_.status_)];
      ++statuses[static_cast<unsigned int>(actual.grammage_.status_)];
      ++planeStatuses[static_cast<unsigned int>(actual.observation_.status_)];
      if (data.name_ == "density-1" &&
          query.category_ == fixture::Category::NonFinitePosition &&
          query.lengthM_ == 5.) {
        REQUIRE(actual.layer_.status_ == fixture::reference::Status::NonFiniteInput);
        REQUIRE(actual.grammage_.status_ == fixture::reference::Status::Success);
        REQUIRE(actual.arclength_.status_ == fixture::reference::Status::Success);
        ++homogeneousNanWitness;
      }
      if (data.name_ == "density-1" && query.grammageGPerCm2_ == 0. &&
          std::signbit(query.grammageGPerCm2_)) {
        REQUIRE(actual.arclength_.status_ == fixture::reference::Status::Success);
        REQUIRE(std::signbit(actual.arclength_.value_));
        ++negativeZeroWitness;
      }
      ++count;
    }
    if (!data.integration_.empty()) {
      REQUIRE(data.queries_.size() == data.integration_.size());
    }
    for (std::size_t index = 0; index < data.integration_.size(); ++index) {
      INFO("Actual native snapshot, fixed short step, not a shower or winner loop");
      CAPTURE(index);
      auto input = data.integration_[index];
      auto const& query = data.queries_[index];
      fixture::compare(query, fixture::integration_query(input), require_exact, state);
      auto const actual = fixture::actual_integration(view, input, query);
      auto const expected = fixture::expected_integration(reference, input, query);
      fixture::compare(actual, expected, require_exact, state);
      fixture::check_integration(input, actual, check);
      fixture::compare(input, data.integration_[index], require_exact, state);
    }
    fixture::check_storage(data, uploaded, require_exact, state);
  }
  CAPTURE(count);
  REQUIRE(count == 160);
  for (auto value : categories) { REQUIRE(value > 0); }
  for (unsigned int status : {0u, 1u, 2u, 3u, 5u, 6u}) { REQUIRE(statuses[status] > 0); }
  REQUIRE(statuses[4] == 0);
  for (auto value : planeStatuses) { REQUIRE(value > 0); }
  REQUIRE(homogeneousNanWitness == 3);
  REQUIRE(negativeZeroWitness == 1);
  Kokkos::fence();
}
