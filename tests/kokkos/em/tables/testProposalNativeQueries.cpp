/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "NativeTableFixture.hpp"
#include "ProposalNativeQueryTestData.hpp"

#include <catch2/catch_all.hpp>
#include <array>
#include <cmath>
#include <set>

namespace {
  void require_exact(double actual, double expected) {
    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == expected);
      if (expected == 0.) REQUIRE(std::signbit(actual) == std::signbit(expected));
    }
  }
} // namespace

TEST_CASE("Real native tables preserve every production query on the host",
          "[kokkos_native_queries]") {
  namespace fixture = native_query_test;
  namespace current = corsika::kokkos;
  auto const table = current::test::make_native_tables();
  REQUIRE(table.formatVersion_ == 7);
  REQUIRE_FALSE(table.dndxColumns_.empty());
  REQUIRE_FALSE(table.bicubicValues_.empty());
  REQUIRE_FALSE(table.cubicValues_.empty());
  std::set<std::int32_t> particles;
  std::set<current::NativeRateModel> rateModels;
  for (auto const& column : table.dndxColumns_) {
    particles.insert(column.pdgId_);
    rateModels.insert(column.rateModel_);
  }
  REQUIRE(particles == std::set<std::int32_t>{-13, -11, 11, 13, 22});
  REQUIRE(rateModels.size() == 4);
  auto const hashBefore = current::calculate_proposal_native_hash(table);
  fixture::HostReference const referenceOwner(table);
  auto const reference = referenceOwner.view(table);
  auto const view = current::make_proposal_native_host_view(table);
  auto const queries = fixture::queries(table);
  std::array<std::size_t, static_cast<std::size_t>(fixture::Operation::Count)> calls{};
  std::array<std::size_t, static_cast<std::size_t>(fixture::Operation::Count)>
      successes{};
  std::array<std::size_t, 9> statuses{};
  std::size_t newton{}, bisection{};
  auto const state = [](auto actual, auto expected) { REQUIRE(actual == expected); };
  for (std::size_t index = 0; index < queries.size(); ++index) {
    auto const& query = queries[index];
    INFO("Same-host production oracle; query "
         << index << ", operation " << static_cast<unsigned>(query.operation_) << ", PID "
         << query.pid_ << ", process " << query.process_ << ", energy " << query.energy_
         << ", argument " << query.argument_ << ", split " << query.photonSplit_);
    auto const actual = fixture::evaluate(view, query);
    auto const expected = fixture::expected(reference, query);
    fixture::compare(actual, expected, require_exact, state);
    auto const operation = static_cast<std::size_t>(query.operation_);
    ++calls[operation];
    REQUIRE(actual.status_ < statuses.size());
    ++statuses[actual.status_];
    if (actual.status_ == 0) ++successes[operation];
    newton += actual.newton_;
    bisection += actual.bisection_;
  }
  for (std::size_t i = 0; i < calls.size(); ++i) {
    INFO("Public operation " << i);
    REQUIRE(calls[i] > 0);
    REQUIRE(successes[i] > 0);
  }
  REQUIRE(newton > 0);
  REQUIRE(bisection > 0);
  // These are deliberately supplied invalid requests, not malformed coefficient pointers.
  for (auto status : {current::NativeQueryStatus::InvalidView,
                      current::NativeQueryStatus::NonFiniteInput,
                      current::NativeQueryStatus::ParticleNotFound,
                      current::NativeQueryStatus::ColumnNotFound,
                      current::NativeQueryStatus::EnergyOutOfRange,
                      current::NativeQueryStatus::QuantileOutOfRange,
                      current::NativeQueryStatus::UnsupportedKinematics})
    REQUIRE(statuses[static_cast<std::size_t>(status)] > 0);
  REQUIRE(current::calculate_proposal_native_hash(table) == hashBefore);
}
