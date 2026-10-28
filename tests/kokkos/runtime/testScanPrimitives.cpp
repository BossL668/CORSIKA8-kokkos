/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/runtime/CompositeExclusiveScan.hpp>
#include <corsika/kokkos/runtime/RangePolicy.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <tests/kokkos/runtime/ScanReference.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>

namespace {
  namespace current = corsika::kokkos;
  namespace reference = scan_reference;
  using execution_space_type = Kokkos::DefaultExecutionSpace;
  using memory_space_type = execution_space_type::memory_space;

  void ensure_runtime() { static current::Runtime runtime; }

  struct CountRecord {
    std::uint64_t first_{};
    std::uint32_t stride_{};
  };

  template <std::size_t NColumns>
  struct RecordLoader {
    KOKKOS_INLINE_FUNCTION current::CompositeScanValue<NColumns> operator()(
        CountRecord const& record) const {
      current::CompositeScanValue<NColumns> value{};
      for (std::size_t c = 0; c < NColumns; ++c)
        value.values_[c] = record.first_ + c * record.stride_;
      return value;
    }
  };

  template <std::size_t NColumns>
  struct ReferenceRecordLoader {
    KOKKOS_INLINE_FUNCTION reference::CompositeScanValue<NColumns> operator()(
        CountRecord const& record) const {
      reference::CompositeScanValue<NColumns> value{};
      for (std::size_t c = 0; c < NColumns; ++c)
        value.values[c] = record.first_ + c * record.stride_;
      return value;
    }
  };

  template <std::size_t NColumns>
  void check_scan() {
    ensure_runtime();
    using workspace_type =
        current::CompositeExclusiveScan<execution_space_type, NColumns>;
    using original_type =
        reference::KokkosCompositeExclusiveScanWorkspace<execution_space_type, NColumns>;
    using input_type = current::CompositeUint32Counts<NColumns>;
    using original_input_type = reference::CompositeUint32Counts<NColumns>;
    static_assert(sizeof(input_type) == sizeof(original_input_type));
    static_assert(std::is_trivially_copyable_v<input_type>);
    constexpr std::size_t maximum = 65537;
    Kokkos::View<input_type*, memory_space_type> input("input", maximum);
    Kokkos::View<original_input_type*, memory_space_type> originalInput("old input",
                                                                        maximum);
    auto host = Kokkos::create_mirror_view(input);
    auto oldHost = Kokkos::create_mirror_view(originalInput);
    for (std::size_t i = 0; i < maximum; ++i) {
      for (std::size_t c = 0; c < NColumns; ++c) {
        auto value = (i % 3 == 0)
                         ? std::numeric_limits<std::uint32_t>::max()
                         : static_cast<std::uint32_t>((i * 7919 + c * 31) % 65537);
        host(i).values_[c] = oldHost(i).values[c] = value;
      }
    }
    execution_space_type execution;
    Kokkos::deep_copy(execution, input, host);
    Kokkos::deep_copy(execution, originalInput, oldHost);
    typename workspace_type::prefix_view_type prefixes("prefixes", maximum);
    typename workspace_type::total_view_type total("total");
    typename original_type::PrefixView oldPrefixes("old prefixes", maximum);
    typename original_type::TotalView oldTotal("old total");
    workspace_type workspace;
    original_type original;
    REQUIRE(workspace.cubAvailable() == original.cubAvailable());
    for (auto mode : {current::CompositeScanImplementation::Portable,
                      current::CompositeScanImplementation::Automatic}) {
      auto oldMode = mode == current::CompositeScanImplementation::Portable
                         ? reference::CompositeScanImplementation::Portable
                         : reference::CompositeScanImplementation::Automatic;
      for (std::size_t count : {0, 1, 2, 7, 31, 32, 33, 255, 256, 257, 4096, 4097, 65535,
                                65536, 65537, 3, 0}) {
        CAPTURE(NColumns, count, static_cast<int>(mode));
        auto previousCapacity = workspace.cubCapacity();
        workspace.exclusiveScan(input, count, prefixes, total,
                                current::CompositeUint32Loader<NColumns>{}, execution,
                                mode);
        original.exclusiveScan(originalInput, count, oldPrefixes, oldTotal,
                               reference::CompositeUint32Loader<NColumns>{}, execution,
                               oldMode);
        execution.fence("scan comparison");
        auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
        auto oldValues =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, oldPrefixes);
        auto sum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, total);
        auto oldSum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, oldTotal);
        std::array<std::uint64_t, NColumns> expected{};
        bool equalReference = true, equalExpected = true;
        for (std::size_t i = 0; i < count; ++i) {
          for (std::size_t c = 0; c < NColumns; ++c) {
            equalReference &= values(i).values_[c] == oldValues(i).values[c];
            equalExpected &= values(i).values_[c] == expected[c];
            expected[c] += host(i).values_[c];
          }
        }
        REQUIRE(equalReference);
        REQUIRE(equalExpected);
        for (std::size_t c = 0; c < NColumns; ++c) {
          REQUIRE(sum().values_[c] == oldSum().values[c]);
          REQUIRE(sum().values_[c] == expected[c]);
        }
        REQUIRE(workspace.usedCubForLastScan() == original.usedCubForLastScan());
        REQUIRE(workspace.usedCubForLastScan() ==
                (workspace.cubAvailable() &&
                 mode == current::CompositeScanImplementation::Automatic && count > 0));
        REQUIRE(workspace.cubCapacity() >= previousCapacity);
        REQUIRE(workspace.cubCapacity() == original.cubCapacity());
        REQUIRE(workspace.cubTemporaryBytes() == original.cubTemporaryBytes());
      }
    }
    // Resident kernels supply their own count records rather than the packed
    // helper type. Also verify a loader with individual values above uint32.
    Kokkos::View<CountRecord*, memory_space_type> records("records", 257);
    auto hostRecords = Kokkos::create_mirror_view(records);
    for (std::size_t i = 0; i < 257; ++i)
      hostRecords(i) = {0x100000001ULL * (i + 1), static_cast<std::uint32_t>(i % 17)};
    Kokkos::deep_copy(execution, records, hostRecords);
    for (auto mode : {current::CompositeScanImplementation::Portable,
                      current::CompositeScanImplementation::Automatic}) {
      auto oldMode = mode == current::CompositeScanImplementation::Portable
                         ? reference::CompositeScanImplementation::Portable
                         : reference::CompositeScanImplementation::Automatic;
      workspace.exclusiveScan(records, 257, prefixes, total, RecordLoader<NColumns>{},
                              execution, mode);
      original.exclusiveScan(records, 257, oldPrefixes, oldTotal,
                             ReferenceRecordLoader<NColumns>{}, execution, oldMode);
      execution.fence("custom scan loader");
      auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, prefixes);
      auto oldValues =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, oldPrefixes);
      auto sum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, total);
      auto oldSum = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, oldTotal);
      std::array<std::uint64_t, NColumns> expected{};
      bool matched = true;
      for (std::size_t i = 0; i < 257; ++i) {
        for (std::size_t c = 0; c < NColumns; ++c) {
          matched &= values(i).values_[c] == oldValues(i).values[c];
          matched &= values(i).values_[c] == expected[c];
          expected[c] += hostRecords(i).first_ + c * hostRecords(i).stride_;
        }
      }
      REQUIRE(matched);
      for (std::size_t c = 0; c < NColumns; ++c) {
        REQUIRE(sum().values_[c] == expected[c]);
        REQUIRE(sum().values_[c] == oldSum().values[c]);
      }
    }
    REQUIRE_THROWS_AS(
        workspace.exclusiveScan(input, maximum + 1, prefixes, total,
                                current::CompositeUint32Loader<NColumns>{}, execution),
        std::length_error);
    typename workspace_type::prefix_view_type shortOutput("short", 1);
    REQUIRE_THROWS_AS(
        workspace.exclusiveScan(input, 2, shortOutput, total,
                                current::CompositeUint32Loader<NColumns>{}, execution),
        std::length_error);
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (workspace_type::cubAvailable()) {
      REQUIRE_THROWS_AS(current::composite_scan::cub_temporary_bytes<NColumns>(
                            static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1,
                            execution.cuda_stream()),
                        std::length_error);
    }
#endif
  }
} // namespace

TEST_CASE("Production range policy preserves OpenMP-only chunk control", "[scan]") {
  ensure_runtime();
  execution_space_type execution;
  for (std::size_t count : {0, 1, 7, 1024}) {
    for (std::size_t chunk : {0, 1, 8, 64}) {
      auto policy = current::make_range_policy(execution, count, chunk);
      auto original = reference::makeKokkosRangePolicy(execution, count, chunk);
      REQUIRE(policy.begin() == original.begin());
      REQUIRE(policy.end() == original.end());
      REQUIRE(policy.chunk_size() == original.chunk_size());
      std::uint64_t sum{};
      Kokkos::parallel_reduce(
          "range sum", policy,
          KOKKOS_LAMBDA(std::size_t i, std::uint64_t & value) { value += i; }, sum);
      REQUIRE(sum == (count == 0 ? 0 : count * (count - 1) / 2));
    }
  }
}

TEST_CASE("Production seven-column scan matches exact integer prefixes", "[scan]") {
  check_scan<7>();
}

TEST_CASE("Production eight-column scan matches exact integer prefixes", "[scan]") {
  check_scan<8>();
}
