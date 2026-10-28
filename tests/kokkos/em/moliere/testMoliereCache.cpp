/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "MoliereTestUtilities.hpp"
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <limits>
#include <sys/wait.h>

namespace {
  using namespace corsika::kokkos;
  namespace fixture = corsika::kokkos::test;
  namespace old = fixture::moliere_reference;

  // Independent v1 wire description, matching fixed-production fields exactly.
  struct CacheHeader {
    char magic_[8]{};
    std::uint32_t version_{};
    std::uint32_t snapshotBytes_{};
    std::uint64_t tableBytes_{};
    std::uint64_t checksum_{};
    old::MoliereSnapshot snapshot_{};
  };

  template <typename TValue>
  void store(std::vector<char>& bytes, std::size_t offset, TValue value) {
    REQUIRE(offset + sizeof(value) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  }

  void rehash(std::vector<char>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (auto index = sizeof(CacheHeader); index < bytes.size(); ++index) {
      hash ^= static_cast<unsigned char>(bytes[index]);
      hash *= 1099511628211ULL;
    }
    store(bytes, offsetof(CacheHeader, checksum_), hash);
  }

  void same_table(MoliereInterpolationTable const& table,
                  old::MoliereInterpolationTable const& expected) {
    auto const adapted = fixture::moliere_reference_table(table);
    REQUIRE(std::memcmp(&adapted, &expected, sizeof(expected)) == 0);
  }
} // namespace

TEST_CASE("Moliere interpolation and v1 cache preserve production contracts",
          "[kokkos][moliere][cache]") {
  auto const directory = fixture::moliere_test_directory("cache");
  auto const snapshot = fixture::small_moliere_snapshot();
  auto const reference = fixture::moliere_reference_snapshot(snapshot);
  auto const oldPath = directory / "old.cache";
  auto const newPath = directory / "new.cache";
  auto const expected = old::loadOrMakeMoliereInterpolationTable(reference, oldPath);
  auto const actual = load_or_make_moliere_interpolation_table(snapshot, newPath);
  same_table(actual, expected);
  auto const pristine = fixture::moliere_read_bytes(oldPath);
  REQUIRE(pristine.size() == sizeof(CacheHeader) + sizeof(expected));
  REQUIRE(fixture::moliere_read_bytes(newPath) == pristine);

  // A changed snapshot invalidates an otherwise complete v1 file.
  auto changed = snapshot;
  changed.particleMassMeV_ *= 2.;
  auto const changedExpected = old::loadOrMakeMoliereInterpolationTable(
      fixture::moliere_reference_snapshot(changed), directory / "changed-old.cache");
  same_table(load_or_make_moliere_interpolation_table(changed, newPath), changedExpected);
  REQUIRE(fixture::moliere_read_bytes(newPath) ==
          fixture::moliere_read_bytes(directory / "changed-old.cache"));
  REQUIRE(fixture::moliere_read_bytes(newPath) != pristine);
  same_table(load_or_make_moliere_interpolation_table(snapshot, newPath), expected);
  REQUIRE(fixture::moliere_read_bytes(newPath) == pristine);
  auto const timestamp = std::filesystem::last_write_time(newPath);
  same_table(load_or_make_moliere_interpolation_table(snapshot, newPath), expected);
  REQUIRE(std::filesystem::last_write_time(newPath) == timestamp);
  // Both directions read files written by the other implementation.
  same_table(load_or_make_moliere_interpolation_table(snapshot, oldPath), expected);
  auto const oldFromNew = old::loadOrMakeMoliereInterpolationTable(reference, newPath);
  REQUIRE(std::memcmp(&oldFromNew, &expected, sizeof(expected)) == 0);
  same_table(make_moliere_interpolation_table(snapshot), expected);
  same_table(load_or_make_moliere_interpolation_table(snapshot, {}), expected);

  for (unsigned mutation = 0; mutation < 11; ++mutation) {
    INFO("v1 cache corruption " << mutation);
    auto bytes = pristine;
    switch (mutation) {
      case 0:
        bytes.resize(sizeof(CacheHeader) - 1);
        break;
      case 1:
        bytes.pop_back();
        break;
      case 2:
        bytes.push_back('x');
        break;
      case 3:
        bytes[0] ^= 1;
        break;
      case 4:
        store(bytes, offsetof(CacheHeader, version_), std::uint32_t{2});
        break;
      case 5:
        store(bytes, offsetof(CacheHeader, snapshotBytes_), std::uint32_t{0});
        break;
      case 6:
        store(bytes, offsetof(CacheHeader, tableBytes_), std::uint64_t{0});
        break;
      case 7:
        bytes[offsetof(CacheHeader, checksum_)] ^= 1;
        break;
      case 8:
        bytes[offsetof(CacheHeader, snapshot_)] ^= 1;
        break;
      case 9:
        store(bytes, sizeof(CacheHeader), std::numeric_limits<double>::quiet_NaN());
        rehash(bytes);
        break;
      case 10:
        store(bytes,
              sizeof(CacheHeader) +
                  offsetof(old::MoliereInterpolationTable, initial_guess_delta),
              std::numeric_limits<double>::infinity());
        rehash(bytes);
        break;
    }
    auto const path = directory / ("corrupt-" + std::to_string(mutation));
    fixture::moliere_write_bytes(path, bytes);
    same_table(load_or_make_moliere_interpolation_table(snapshot, path), expected);
    REQUIRE(fixture::moliere_read_bytes(path) == pristine);
  }

  // Lock/open and atomic rename failures preserve the in-memory result.
  same_table(load_or_make_moliere_interpolation_table(
                 snapshot, directory / "missing-parent" / "file"),
             expected);
  auto const blocked = directory / "directory-not-file";
  std::filesystem::create_directory(blocked);
  same_table(load_or_make_moliere_interpolation_table(snapshot, blocked), expected);
  REQUIRE(std::filesystem::is_directory(blocked));
  for (auto const& entry : std::filesystem::directory_iterator(directory))
    REQUIRE(entry.path().filename().string().find(".tmp.") == std::string::npos);

  auto invalid = snapshot;
  invalid.componentCount_ = 0;
  REQUIRE_THROWS_AS(make_moliere_interpolation_table(invalid), std::invalid_argument);
  REQUIRE_THROWS_AS(load_or_make_moliere_interpolation_table(invalid, newPath),
                    std::invalid_argument);
  REQUIRE(fixture::moliere_read_bytes(newPath) == pristine);

  // Two cold producers exercise the shared persistent lock and atomic publication.
  auto const concurrent = directory / "concurrent.cache";
  std::array<pid_t, 2> children{};
  for (auto& child : children) {
    child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
      try {
        auto const table = load_or_make_moliere_interpolation_table(snapshot, concurrent);
        auto const adapted = fixture::moliere_reference_table(table);
        ::_exit(std::memcmp(&adapted, &expected, sizeof(expected)) == 0 ? 0 : 2);
      } catch (...) { ::_exit(3); }
    }
  }
  for (auto child : children) {
    int status{};
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
  }
  REQUIRE(fixture::moliere_read_bytes(concurrent) == pristine);
  REQUIRE(std::filesystem::exists(concurrent.string() + ".lock"));
  for (auto const& entry : std::filesystem::directory_iterator(directory))
    REQUIRE(entry.path().filename().string().find(".tmp.") == std::string::npos);
}
