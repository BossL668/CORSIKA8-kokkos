/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/runtime/TuningCache.hpp>
#include <corsika/accelerator/em/KokkosTuningCache.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cp = corsika::kokkos;
namespace old = corsika::accelerator::em;
std::size_t cache_other_translation_unit();
namespace {
  void require(bool value, char const* text) {
    if (!value) throw std::runtime_error(text);
  }
  std::string read(std::filesystem::path path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  }
  void write(std::filesystem::path path, std::string const& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    require(output.good(), "Cannot write cache fixture");
  }
  std::string replace(std::string text, std::string const& key,
                      std::string const& value) {
    auto start = text.find(key + '=');
    require(start != std::string::npos, "Fixture field missing");
    auto end = text.find('\n', start);
    text.replace(start, end - start, key + '=' + value);
    return text;
  }
  std::string remove(std::string text, std::string const& key) {
    auto start = text.find(key + '=');
    require(start != std::string::npos, "Fixture field missing");
    text.erase(start, text.find('\n', start) - start + 1);
    return text;
  }
  template <typename TResult>
  std::string encode(TResult const& result);
  template <>
  std::string encode(cp::TuningCacheResult const& r) {
    auto p = r.parameters_;
    std::ostringstream out;
    out << r.matched_ << ' ' << r.contentHash_ << ' ' << p.batchSize_ << ' '
        << p.chunkSize_ << ' ' << p.teamSize_ << ' ' << p.trackTileSize_ << ' '
        << p.observerTileSize_ << ' ' << p.deviceQueues_;
    return out.str();
  }
  template <>
  std::string encode(old::KokkosTuningCacheResult const& r) {
    auto p = r.parameters;
    std::ostringstream out;
    out << r.matched << ' ' << r.content_hash << ' ' << p.batch_size << ' '
        << p.chunk_size << ' ' << p.team_size << ' ' << p.track_tile_size << ' '
        << p.observer_tile_size << ' ' << p.device_queues;
    return out.str();
  }
  template <typename TCall>
  std::string outcome(TCall call) {
    try {
      return "ok " + encode(call());
    } catch (std::exception const& error) { return "error " + std::string(error.what()); }
  }
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "Expected cache fixture directory");
    std::filesystem::path directory = argv[1];
    std::filesystem::create_directories(directory);
    cp::TuningKey key{"cuda",     "device 0", "sm_89",      "driver", "runtime", "4.4.01",
                      "compiler", "41cdc543", "table-hash", 0,        1};
    old::KokkosTuningKey oldKey{"cuda",       "device 0", "sm_89",    "driver",
                                "runtime",    "4.4.01",   "compiler", "41cdc543",
                                "table-hash", 0,          1};
    cp::TuningParameters parameters{16384, 32, 64, 16, 8, 3};
    old::KokkosTuningParameters oldParameters{16384, 32, 64, 16, 8, 3};
    auto candidate = directory / "candidate.cache",
         production = directory / "production.cache";
    cp::write_tuning_cache(candidate, key, parameters);
    old::writeKokkosTuningCache(production, oldKey, oldParameters);
    auto const valid = read(candidate);
    require(valid == read(production),
            "Cache serialized bytes differ from pinned production");
    require(valid.find("format=c8-kokkos-tuning-v1\n") == 0,
            "Changed cache schema/version");
    require(cache_other_translation_unit() == 1, "Independent header TU default changed");
    unsigned comparisons = 0;
    auto compare = [&](std::filesystem::path const& path, bool required, bool expectError,
                       bool matched) {
      auto actual = outcome([&] { return cp::load_tuning_cache(path, key, required); });
      auto reference =
          outcome([&] { return old::loadKokkosTuningCache(path, oldKey, required); });
      require(actual == reference, "Cache load or error differs from pinned production");
      require((actual.rfind("error ", 0) == 0) == expectError,
              "Unexpected cache acceptance");
      if (!expectError) {
        require((actual.rfind("ok 1 ", 0) == 0) == matched,
                "Unexpected cache match flag");
        if (!matched)
          require(actual == "ok " + encode(cp::TuningCacheResult{}), "Defaults changed");
      }
      ++comparisons;
    };
    compare(candidate, false, false, true);
    compare(production, true, false, true);
    auto loaded = cp::load_tuning_cache(production, key, true);
    require(loaded.parameters_.batchSize_ == 16384 &&
                loaded.parameters_.deviceQueues_ == 3 && loaded.contentHash_.size() == 64,
            "Valid cache parameters/hash were not loaded");
    write(directory / "content.sha256", loaded.contentHash_ + '\n');
    for (auto path : {std::filesystem::path{}, directory / "missing.cache", directory}) {
      compare(path, false, false, false);
      compare(path, true, true, false);
    }
    auto mutation = directory / "mutation.cache";
    // Every exact-match field, including schema and each integer identity field.
    for (std::string field :
         {"format", "backend", "device_name", "architecture", "driver_version",
          "runtime_version", "kokkos_version", "compiler_version", "project_revision",
          "proposal_table_hash", "device", "threads"}) {
      for (auto text : {replace(valid, field, "different"), remove(valid, field)}) {
        write(mutation, text);
        compare(mutation, false, false, false);
        compare(mutation, true, true, false);
      }
    }
    for (std::string field : {"batch_size", "chunk_size", "team_size", "track_tile_size",
                              "observer_tile_size", "device_queues"}) {
      for (auto text : {remove(valid, field), replace(valid, field, "0"),
                        replace(valid, field, "abc"), replace(valid, field, "2x"),
                        replace(valid, field, "18446744073709551616000")}) {
        write(mutation, text);
        compare(mutation, false, true, false);
        compare(mutation, true, true, false);
      }
    }
    for (auto text :
         {valid + "bad-line\n", valid + "=value\n", valid + "backend=cuda\n"}) {
      write(mutation, text);
      compare(mutation, false, true, false);
      compare(mutation, true, true, false);
    }
    write(mutation, "# comment\n\n" + valid + "unknown_option=kept\n");
    compare(mutation, true, false, true);
    for (unsigned field = 0; field < 6; ++field) {
      auto p = parameters;
      auto q = oldParameters;
      std::array<std::size_t*, 6> a{&p.batchSize_,        &p.chunkSize_,
                                    &p.teamSize_,         &p.trackTileSize_,
                                    &p.observerTileSize_, &p.deviceQueues_};
      std::array<std::size_t*, 6> b{&q.batch_size,         &q.chunk_size,
                                    &q.team_size,          &q.track_tile_size,
                                    &q.observer_tile_size, &q.device_queues};
      *a[field] = 0;
      *b[field] = 0;
      bool currentError = false, oldError = false;
      try {
        cp::write_tuning_cache(mutation, key, p);
      } catch (std::invalid_argument const&) { currentError = true; }
      try {
        old::writeKokkosTuningCache(mutation, oldKey, q);
      } catch (std::invalid_argument const&) { oldError = true; }
      require(currentError && oldError, "Invalid zero tuning parameter was written");
    }
    bool emptyError = false;
    try {
      cp::write_tuning_cache({}, key, parameters);
    } catch (std::invalid_argument const&) { emptyError = true; }
    require(emptyError, "Empty output path accepted");
    auto blocked = directory / "blocked.cache";
    std::filesystem::create_directory(blocked.string() + ".tmp");
    bool writeError = false;
    try {
      cp::write_tuning_cache(blocked, key, parameters);
    } catch (std::runtime_error const&) { writeError = true; }
    require(writeError, "Blocked temporary cache output was accepted");
    std::cout << "PASS cache comparisons=" << comparisons << "\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
