/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "NativeTableFixture.hpp"

#include <corsika/kokkos/em/proposal/NativeTableExporter.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace corsika::kokkos;

TEST_CASE("Native table export preserves all five official calculator families",
          "[kokkos][proposal][native_tables]") {
  auto const directory = corsika::kokkos::test::native_table_cache_directory();
  auto const audit = corsika::kokkos::test::audit_native_tables(directory);
  auto const& table = audit.table_;

  REQUIRE(audit.processCount_ == 22);
  REQUIRE(audit.projectileIds_ == std::vector<std::int32_t>{22, 11, -11, 13, -13});
  REQUIRE(audit.defaultViewsEmpty_);
  REQUIRE(audit.emptyViewsRejected_);
  REQUIRE(audit.ordinaryStateUnchanged_);
  REQUIRE(audit.allRecordsEqual_);
  REQUIRE(table.contentHash_ == audit.referenceHash_);
  REQUIRE(calculate_proposal_native_hash(table) == audit.referenceHash_);
  REQUIRE_NOTHROW(validate_proposal_native_table(table));
  REQUIRE(table.formatVersion_ == ProposalNativeTableFormatVersion);
  REQUIRE(table.stochasticCutMeV_ == .4);
  REQUIRE(table.interactionIdentities_.size() == 5);
  REQUIRE(table.totalRateColumns_.size() == 5);
  REQUIRE(table.utilityColumns_.size() == 4);
  REQUIRE(table.proposalCacheTableCount_ > 0);
  REQUIRE(table.proposalCacheHitCount_ <= table.proposalCacheTableCount_);
  REQUIRE(proposal_native_table_bytes(table) > 0);

  std::set<std::int32_t> particles;
  std::set<std::pair<std::int32_t, std::int32_t>> processes;
  for (auto const& column : table.dndxColumns_) {
    particles.insert(column.pdgId_);
    processes.emplace(column.pdgId_, column.processId_);
  }
  REQUIRE(particles == std::set<std::int32_t>{-13, -11, 11, 13, 22});
  REQUIRE(processes.size() == 22);
  REQUIRE(table.selectionColumnIndices_.size() == table.dndxColumns_.size());
  REQUIRE(table.dedxAccumulationIndices_.size() == table.dedxColumns_.size());

  // A second construction uses the same dedicated cache, independently of whether
  // the first construction in this executable was cold or already warm.
  auto const loaded = corsika::kokkos::test::make_native_tables(directory);
  REQUIRE(loaded.contentHash_ == table.contentHash_);
  REQUIRE(loaded.proposalCacheTableCount_ == table.proposalCacheTableCount_);
  REQUIRE(loaded.proposalCacheAllHit_);
  REQUIRE(loaded.proposalCacheHitCount_ == loaded.proposalCacheTableCount_);
  REQUIRE(calculate_proposal_native_hash(loaded) == table.contentHash_);

  SECTION("Cache diagnostics and the stored digest are not physics hash inputs") {
    auto copy = table;
    ++copy.proposalCacheTableCount_;
    copy.proposalCacheHitCount_ = 0;
    copy.proposalCacheAllHit_ = false;
    copy.contentHash_.fill(0);
    REQUIRE(calculate_proposal_native_hash(copy) == table.contentHash_);
    REQUIRE(table.contentHash_ == audit.referenceHash_);
  }

  SECTION("Existing validation rejects corrupt export records") {
    auto copy = table;
    ++copy.formatVersion_;
    REQUIRE_THROWS_AS(validate_proposal_native_table(copy), std::invalid_argument);
    copy = table;
    REQUIRE_FALSE(copy.bicubicValues_.empty());
    copy.bicubicValues_.front() = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(validate_proposal_native_table(copy), std::invalid_argument);
    copy = table;
    copy.bicubicValues_.pop_back();
    REQUIRE_THROWS_AS(validate_proposal_native_table(copy), std::invalid_argument);
    copy = table;
    REQUIRE(copy.selectionColumnIndices_.size() > 1);
    copy.selectionColumnIndices_[1] = copy.selectionColumnIndices_[0];
    REQUIRE_THROWS_AS(validate_proposal_native_table(copy), std::invalid_argument);
    REQUIRE_THROWS_AS(classify_proposal_native_kinematic_model(
                          table.dndxColumns_.front().processId_,
                          "unknown-parametrization", table.dndxColumns_.front().pdgId_),
                      std::invalid_argument);
  }
}

TEST_CASE("Native content hashing retains standard SHA-256 byte encoding",
          "[kokkos][native_tables][sha256]") {
  REQUIRE(to_hex(sha256({})) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  std::vector<std::uint8_t> const abc{'a', 'b', 'c'};
  REQUIRE(to_hex(sha256(abc)) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::string const boundary = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  REQUIRE(to_hex(sha256(std::vector<std::uint8_t>(boundary.begin(), boundary.end()))) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}
