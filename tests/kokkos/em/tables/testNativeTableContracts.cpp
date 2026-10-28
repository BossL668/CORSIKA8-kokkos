/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>

#include <catch2/catch_all.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

TEST_CASE("Canonical digest matches standard SHA-256 vectors", "[native_table_hash]") {
  using namespace corsika::kokkos;
  std::vector<std::uint8_t> const empty;
  REQUIRE(to_hex(sha256(empty)) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  std::vector<std::uint8_t> const abc{'a', 'b', 'c'};
  REQUIRE(to_hex(sha256(abc)) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::string const text = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  std::vector<std::uint8_t> const multiBlock{text.begin(), text.end()};
  REQUIRE(to_hex(sha256(multiBlock)) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("Empty native export is not a usable physics table",
          "[native_table_contract]") {
  using namespace corsika::kokkos;
  ProposalNativeTableSet table;
  REQUIRE(table.formatVersion_ == 7);
  REQUIRE(proposal_native_table_bytes(table) == 0);
  REQUIRE_THROWS_AS(validate_proposal_native_table(table), std::invalid_argument);
  auto const originalHash = calculate_proposal_native_hash(table);
  table.proposalCacheAllHit_ = true;
  table.proposalCacheHitCount_ = 5;
  table.proposalCacheTableCount_ = 5;
  REQUIRE(calculate_proposal_native_hash(table) == originalHash);
  table.stochasticCutMeV_ = 0.4;
  REQUIRE(calculate_proposal_native_hash(table) != originalHash);
}
