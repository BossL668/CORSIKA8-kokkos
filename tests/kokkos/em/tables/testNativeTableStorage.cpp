/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "NativeTableFixture.hpp"

#include <corsika/kokkos/em/execution/NativeTableStorage.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeQueries.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>

#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = typename execution_space_type::memory_space;
  using storage_type = corsika::kokkos::NativeTableStorage<execution_space_type>;
#if defined(CORSIKA_EXPECT_CUDA)
  static_assert(std::is_same_v<execution_space_type, Kokkos::Cuda>);
#elif defined(CORSIKA_EXPECT_OPENMP)
  static_assert(std::is_same_v<execution_space_type, Kokkos::OpenMP>);
#endif
} // namespace

TEST_CASE("Native table ownership preserves upload and failure contracts",
          "[native_table_storage]") {
  using namespace corsika::kokkos;
  auto const* cache = std::getenv("CORSIKA_NATIVE_TABLE_CACHE");
  REQUIRE(cache != nullptr);
  REQUIRE_FALSE(std::string{cache}.empty());
  auto const table = test::make_native_tables(cache);
  REQUIRE_FALSE(Kokkos::is_initialized());
  {
    Runtime runtime;
    INFO("Actual execution space: " << execution_space_type::name());
    storage_type storage;
    REQUIRE_FALSE(storage.isInitialized());
    REQUIRE(storage.getDeviceBytes() == 0);
    REQUIRE_THROWS_AS(storage.getDeviceView(), std::logic_error);
    REQUIRE_THROWS_AS(storage.getSourceContentHash(), std::logic_error);
    auto const bytes = storage_type::projectedDeviceBytes(table);
    REQUIRE(bytes > 0);
    REQUIRE(bytes == proposal_native_table_bytes(table));

    REQUIRE_THROWS_AS(storage.initialize(table, bytes - 1), std::runtime_error);
    REQUIRE_FALSE(storage.isInitialized());
    REQUIRE(storage.getDeviceBytes() == 0);
    {
      auto invalid = table;
      invalid.formatVersion_ = 0;
      REQUIRE_THROWS_AS(storage.initialize(invalid), std::invalid_argument);
      REQUIRE_FALSE(storage.isInitialized());
      invalid = table;
      invalid.contentHash_[0] ^= 1;
      REQUIRE_THROWS_AS(storage.initialize(invalid), std::invalid_argument);
      REQUIRE_FALSE(storage.isInitialized());
      invalid = table;
      REQUIRE_FALSE(invalid.cubicValues_.empty());
      invalid.cubicValues_[0] += 1.;
      REQUIRE_THROWS_AS(storage.initialize(invalid), std::invalid_argument);
      REQUIRE_FALSE(storage.isInitialized());
    }

    storage.initialize(table, bytes);
    REQUIRE(storage.isInitialized());
    REQUIRE(storage.getDeviceBytes() == bytes);
    REQUIRE(storage.getSourceContentHash() == table.contentHash_);
    REQUIRE_THROWS_AS(storage.initialize(table), std::logic_error);

    // Production copies share immutable Kokkos allocations. A borrower remains
    // valid after one owner resets, provided another owner keeps the arrays alive.
    auto shared = storage;
    auto const borrowed = storage.getDeviceView();
    storage.reset();
    storage.reset();
    REQUIRE_FALSE(storage.isInitialized());
    REQUIRE(storage.getDeviceBytes() == 0);
    REQUIRE_THROWS_AS(storage.getDeviceView(), std::logic_error);
    REQUIRE(shared.isInitialized());
    Kokkos::View<NativeQueryResult*, memory_space_type> results("storage_query", 1);
    Kokkos::parallel_for(
        "native_table_shared_lifetime", Kokkos::RangePolicy<execution_space_type>(0, 1),
        KOKKOS_LAMBDA(int) {
          results(0) = query_proposal_native_total_rate(borrowed, 22, 1.e4);
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    REQUIRE(host(0).status_ == NativeQueryStatus::Success);
    REQUIRE(host(0).value_ > 0.);
    execution_space_type{}.fence("finish borrowed native table queries");
    shared.reset();
    REQUIRE_FALSE(shared.isInitialized());
    storage.initialize(table, bytes);
    REQUIRE(storage.getSourceContentHash() == table.contentHash_);
    storage.reset();
  }
  REQUIRE(Kokkos::is_finalized());
}
