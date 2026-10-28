/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "AuxiliaryModelAssertions.hpp"
#include <array>

TEST_CASE("Host auxiliary queries match fixed production and live PROPOSAL helpers",
          "[kokkos][auxiliary][queries]") {
  namespace fixture = corsika::kokkos::test;
  auto const data = fixture::make_auxiliary_dataset();
  REQUIRE(data.componentCount_ > 0);
  REQUIRE(data.queries_.size() > 500);
  std::vector<fixture::AuxiliaryReferenceSnapshots> references;
  for (auto const& snapshot : data.snapshots_)
    references.push_back(fixture::to_reference(snapshot));
  std::array<std::size_t, 5> calls{};
  std::array<std::uint32_t, 3> branches{};
  std::size_t nativeCount{};
  fixture::AuxiliaryErrorReport report;
  std::vector<fixture::AuxiliaryOutcome> actualValues, expectedValues;
  for (auto const& query : data.queries_) {
    auto const actual =
        fixture::evaluate_auxiliary(data.snapshots_.at(query.snapshotIndex_), query);
    auto const expected =
        fixture::reference_auxiliary(references.at(query.snapshotIndex_), query);
    actualValues.push_back(actual);
    expectedValues.push_back(expected);
    report.observe(query, actual, expected);
  }
  report.write("host");
  for (std::size_t index = 0; index < data.queries_.size(); ++index) {
    auto const& query = data.queries_[index];
    INFO("query " << index << " operation " << static_cast<unsigned>(query.operation_)
                  << " snapshot " << query.snapshotIndex_ << " energy "
                  << query.energyMeV_ << " fraction " << query.fraction_ << " density "
                  << query.densityGPerCm3_);
    REQUIRE(query.snapshotIndex_ < data.snapshots_.size());
    auto const actual = actualValues[index];
    auto const expected = expectedValues[index];
    fixture::require_auxiliary_exact(actual, expected);
    auto const operation = static_cast<std::size_t>(query.operation_);
    ++calls.at(operation);
    if (operation < 3) branches[operation] |= query.branchMask_;
    if (query.nativeComparable_) {
      fixture::require_auxiliary_native(actual, query);
      ++nativeCount;
    }
  }
  for (auto count : calls) REQUIRE(count > 0);
  for (auto mask : branches) REQUIRE(mask == 255u);
  REQUIRE(nativeCount > 500);
  REQUIRE(report.sameDeviceDifferences_ == 0);
}
