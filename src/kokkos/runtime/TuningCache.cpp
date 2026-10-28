/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/TuningCache.hpp>

#include <corsika/kokkos/utility/Sha256.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace corsika::kokkos {
  namespace {
    std::string trim(std::string value) {
      auto const notSpace = [](unsigned char character) {
        return !std::isspace(character);
      };
      value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
      value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(),
                  value.end());
      return value;
    }

    std::map<std::string, std::string> parse(std::string const& contents) {
      std::map<std::string, std::string> fields;
      std::istringstream stream(contents);
      std::string line;
      while (std::getline(stream, line)) {
        line = trim(std::move(line));
        if (line.empty() || line.front() == '#') continue;
        auto const separator = line.find('=');
        if (separator == std::string::npos)
          throw std::runtime_error("invalid Kokkos tuning-cache line without '='");
        auto key = trim(line.substr(0, separator));
        auto value = trim(line.substr(separator + 1));
        if (key.empty() || !fields.emplace(std::move(key), std::move(value)).second)
          throw std::runtime_error("invalid or duplicate Kokkos tuning-cache key");
      }
      return fields;
    }

    std::string require_field(std::map<std::string, std::string> const& fields,
                              std::string const& name) {
      auto const found = fields.find(name);
      if (found == fields.end())
        throw std::runtime_error("Kokkos tuning cache is missing field '" + name + "'");
      return found->second;
    }

    std::size_t parse_positive(std::map<std::string, std::string> const& fields,
                               std::string const& name) {
      auto const text = require_field(fields, name);
      std::size_t consumed{};
      auto const value = std::stoull(text, &consumed);
      if (consumed != text.size() || value == 0)
        throw std::runtime_error("Kokkos tuning-cache field '" + name +
                                 "' must be positive");
      return value;
    }

    bool matches(std::map<std::string, std::string> const& fields, TuningKey const& key) {
      auto same = [&](std::string const& name, std::string const& expected) {
        auto const found = fields.find(name);
        return found != fields.end() && found->second == expected;
      };
      return same("format", tuningCacheFormat) && same("backend", key.backend_) &&
             same("device_name", key.deviceName_) &&
             same("architecture", key.architecture_) &&
             same("driver_version", key.driverVersion_) &&
             same("runtime_version", key.runtimeVersion_) &&
             same("kokkos_version", key.kokkosVersion_) &&
             same("compiler_version", key.compilerVersion_) &&
             same("project_revision", key.projectRevision_) &&
             same("proposal_table_hash", key.proposalTableHash_) &&
             same("device", std::to_string(key.device_)) &&
             same("threads", std::to_string(key.threads_));
    }

    std::string read_file(std::filesystem::path const& path) {
      std::ifstream input(path, std::ios::binary);
      if (!input)
        throw std::runtime_error("cannot open Kokkos tuning cache: " + path.string());
      std::ostringstream buffer;
      buffer << input.rdbuf();
      if (!input.good() && !input.eof())
        throw std::runtime_error("failed while reading Kokkos tuning cache: " +
                                 path.string());
      return buffer.str();
    }
  } // namespace

  TuningCacheResult load_tuning_cache(std::filesystem::path const& path,
                                      TuningKey const& key,
                                      bool const requireExactMatch) {
    if (path.empty() || !std::filesystem::is_regular_file(path)) {
      if (requireExactMatch)
        throw std::runtime_error(
            "an exactly matching Kokkos tuning cache was required but the "
            "file does not exist: " +
            path.string());
      return {};
    }
    auto const contents = read_file(path);
    auto const fields = parse(contents);
    if (!matches(fields, key)) {
      if (requireExactMatch)
        throw std::runtime_error(
            "Kokkos tuning cache does not match this backend, device, build, "
            "thread count, or proposal-native table");
      return {};
    }
    TuningCacheResult result;
    result.parameters_.batchSize_ = parse_positive(fields, "batch_size");
    result.parameters_.chunkSize_ = parse_positive(fields, "chunk_size");
    result.parameters_.teamSize_ = parse_positive(fields, "team_size");
    result.parameters_.trackTileSize_ = parse_positive(fields, "track_tile_size");
    result.parameters_.observerTileSize_ = parse_positive(fields, "observer_tile_size");
    result.parameters_.deviceQueues_ = parse_positive(fields, "device_queues");
    std::vector<std::uint8_t> bytes(contents.begin(), contents.end());
    result.contentHash_ = to_hex(sha256(bytes));
    result.matched_ = true;
    return result;
  }

  void write_tuning_cache(std::filesystem::path const& path, TuningKey const& key,
                          TuningParameters const& parameters) {
    if (path.empty())
      throw std::invalid_argument("Kokkos tuning-cache output path is empty");
    if (parameters.batchSize_ == 0 || parameters.chunkSize_ == 0 ||
        parameters.teamSize_ == 0 || parameters.trackTileSize_ == 0 ||
        parameters.observerTileSize_ == 0 || parameters.deviceQueues_ == 0)
      throw std::invalid_argument("Kokkos tuning parameters must all be positive");
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot create Kokkos tuning cache: " +
                               temporary.string());
    output << "format=" << tuningCacheFormat << '\n'
           << "backend=" << key.backend_ << '\n'
           << "device_name=" << key.deviceName_ << '\n'
           << "architecture=" << key.architecture_ << '\n'
           << "driver_version=" << key.driverVersion_ << '\n'
           << "runtime_version=" << key.runtimeVersion_ << '\n'
           << "kokkos_version=" << key.kokkosVersion_ << '\n'
           << "compiler_version=" << key.compilerVersion_ << '\n'
           << "project_revision=" << key.projectRevision_ << '\n'
           << "proposal_table_hash=" << key.proposalTableHash_ << '\n'
           << "device=" << key.device_ << '\n'
           << "threads=" << key.threads_ << '\n'
           << "batch_size=" << parameters.batchSize_ << '\n'
           << "chunk_size=" << parameters.chunkSize_ << '\n'
           << "team_size=" << parameters.teamSize_ << '\n'
           << "track_tile_size=" << parameters.trackTileSize_ << '\n'
           << "observer_tile_size=" << parameters.observerTileSize_ << '\n'
           << "device_queues=" << parameters.deviceQueues_ << '\n';
    output.close();
    if (!output)
      throw std::runtime_error("failed while writing Kokkos tuning cache: " +
                               temporary.string());
    std::filesystem::rename(temporary, path);
  }

} // namespace corsika::kokkos
