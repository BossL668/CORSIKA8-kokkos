/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosTuningCache.hpp>

#include <corsika/accelerator/em/common/tables/Sha256.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace corsika::accelerator::em {
  namespace {
    std::string trim(std::string value) {
      auto const not_space = [](unsigned char character) {
        return !std::isspace(character);
      };
      value.erase(value.begin(),
                  std::find_if(value.begin(), value.end(), not_space));
      value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
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
          throw std::runtime_error(
              "invalid Kokkos tuning-cache line without '='");
        auto key = trim(line.substr(0, separator));
        auto value = trim(line.substr(separator + 1));
        if (key.empty() || !fields.emplace(std::move(key), std::move(value)).second)
          throw std::runtime_error(
              "invalid or duplicate Kokkos tuning-cache key");
      }
      return fields;
    }

    std::string requireField(
        std::map<std::string, std::string> const& fields,
        std::string const& name) {
      auto const found = fields.find(name);
      if (found == fields.end())
        throw std::runtime_error(
            "Kokkos tuning cache is missing field '" + name + "'");
      return found->second;
    }

    std::size_t parsePositive(
        std::map<std::string, std::string> const& fields,
        std::string const& name) {
      auto const text = requireField(fields, name);
      std::size_t consumed{};
      auto const value = std::stoull(text, &consumed);
      if (consumed != text.size() || value == 0)
        throw std::runtime_error(
            "Kokkos tuning-cache field '" + name + "' must be positive");
      return value;
    }

    bool matches(std::map<std::string, std::string> const& fields,
                 KokkosTuningKey const& key) {
      auto same = [&](std::string const& name, std::string const& expected) {
        auto const found = fields.find(name);
        return found != fields.end() && found->second == expected;
      };
      return same("format", KokkosTuningCacheFormat) &&
             same("backend", key.backend) &&
             same("device_name", key.device_name) &&
             same("architecture", key.architecture) &&
             same("driver_version", key.driver_version) &&
             same("runtime_version", key.runtime_version) &&
             same("kokkos_version", key.kokkos_version) &&
             same("compiler_version", key.compiler_version) &&
             same("project_revision", key.project_revision) &&
             same("proposal_table_hash", key.proposal_table_hash) &&
             same("device", std::to_string(key.device)) &&
             same("threads", std::to_string(key.threads));
    }

    std::string readFile(std::filesystem::path const& path) {
      std::ifstream input(path, std::ios::binary);
      if (!input)
        throw std::runtime_error(
            "cannot open Kokkos tuning cache: " + path.string());
      std::ostringstream buffer;
      buffer << input.rdbuf();
      if (!input.good() && !input.eof())
        throw std::runtime_error(
            "failed while reading Kokkos tuning cache: " + path.string());
      return buffer.str();
    }
  } // namespace

  KokkosTuningCacheResult loadKokkosTuningCache(
      std::filesystem::path const& path, KokkosTuningKey const& key,
      bool const require_exact_match) {
    if (path.empty() || !std::filesystem::is_regular_file(path)) {
      if (require_exact_match)
        throw std::runtime_error(
            "an exactly matching Kokkos tuning cache was required but the "
            "file does not exist: " + path.string());
      return {};
    }
    auto const contents = readFile(path);
    auto const fields = parse(contents);
    if (!matches(fields, key)) {
      if (require_exact_match)
        throw std::runtime_error(
            "Kokkos tuning cache does not match this backend, device, build, "
            "thread count, or proposal-native table");
      return {};
    }
    KokkosTuningCacheResult result;
    result.parameters.batch_size = parsePositive(fields, "batch_size");
    result.parameters.chunk_size = parsePositive(fields, "chunk_size");
    result.parameters.team_size = parsePositive(fields, "team_size");
    result.parameters.track_tile_size =
        parsePositive(fields, "track_tile_size");
    result.parameters.observer_tile_size =
        parsePositive(fields, "observer_tile_size");
    result.parameters.device_queues = parsePositive(fields, "device_queues");
    std::vector<std::uint8_t> bytes(contents.begin(), contents.end());
    result.content_hash = gpu::em::tables::toHex(
        gpu::em::tables::sha256(bytes));
    result.matched = true;
    return result;
  }

  void writeKokkosTuningCache(
      std::filesystem::path const& path, KokkosTuningKey const& key,
      KokkosTuningParameters const& parameters) {
    if (path.empty())
      throw std::invalid_argument("Kokkos tuning-cache output path is empty");
    if (parameters.batch_size == 0 || parameters.chunk_size == 0 ||
        parameters.team_size == 0 || parameters.track_tile_size == 0 ||
        parameters.observer_tile_size == 0 || parameters.device_queues == 0)
      throw std::invalid_argument(
          "Kokkos tuning parameters must all be positive");
    if (!path.parent_path().empty())
      std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error(
          "cannot create Kokkos tuning cache: " + temporary.string());
    output << "format=" << KokkosTuningCacheFormat << '\n'
           << "backend=" << key.backend << '\n'
           << "device_name=" << key.device_name << '\n'
           << "architecture=" << key.architecture << '\n'
           << "driver_version=" << key.driver_version << '\n'
           << "runtime_version=" << key.runtime_version << '\n'
           << "kokkos_version=" << key.kokkos_version << '\n'
           << "compiler_version=" << key.compiler_version << '\n'
           << "project_revision=" << key.project_revision << '\n'
           << "proposal_table_hash=" << key.proposal_table_hash << '\n'
           << "device=" << key.device << '\n'
           << "threads=" << key.threads << '\n'
           << "batch_size=" << parameters.batch_size << '\n'
           << "chunk_size=" << parameters.chunk_size << '\n'
           << "team_size=" << parameters.team_size << '\n'
           << "track_tile_size=" << parameters.track_tile_size << '\n'
           << "observer_tile_size=" << parameters.observer_tile_size << '\n'
           << "device_queues=" << parameters.device_queues << '\n';
    output.close();
    if (!output)
      throw std::runtime_error(
          "failed while writing Kokkos tuning cache: " + temporary.string());
    std::filesystem::rename(temporary, path);
  }

} // namespace corsika::accelerator::em
