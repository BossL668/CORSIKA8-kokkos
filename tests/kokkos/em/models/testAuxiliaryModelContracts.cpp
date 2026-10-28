/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "AuxiliaryModelQueryTestData.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <type_traits>

TEST_CASE(
    "Auxiliary numerical records retain defaults layout and invalid-state contracts",
    "[kokkos][auxiliary][contracts]") {
  namespace current = corsika::kokkos;
  namespace fixture = current::test;
  current::PhotonPairLpmSnapshot const photon;
  current::BremsLpmSnapshot const brems;
  current::MoliereSnapshot const moliere;
  REQUIRE(photon.componentCount_ == 0);
  REQUIRE(brems.componentCount_ == 0);
  REQUIRE(moliere.componentCount_ == 0);
  REQUIRE(brems.muonELpmMeV_ == 0.);
  REQUIRE_FALSE(current::valid_moliere_snapshot(moliere));
  REQUIRE(current::PhotonPairLpmResult{}.status_ ==
          current::PhotonPairLpmStatus::InvalidSnapshot);
  REQUIRE(current::BremsLpmResult{}.status_ == current::BremsLpmStatus::InvalidSnapshot);
  REQUIRE(std::is_trivially_copyable_v<fixture::AuxiliaryOutcome>);
  REQUIRE(std::is_trivially_copyable_v<fixture::AuxiliarySnapshots>);
  auto const reference = fixture::to_reference(fixture::AuxiliarySnapshots{});
  for (std::uint32_t operation = 0; operation < 5; ++operation) {
    fixture::AuxiliaryQuery query;
    query.operation_ = static_cast<fixture::AuxiliaryOperation>(operation);
    auto const actual = fixture::evaluate_auxiliary({}, query);
    auto const expected = fixture::reference_auxiliary(reference, query);
    REQUIRE(actual.status_ == expected.status_);
    REQUIRE(actual.reserved_ == expected.reserved_);
    REQUIRE(actual.value_ == expected.value_);
  }
}
