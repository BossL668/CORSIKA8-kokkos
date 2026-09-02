/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace corsika::accelerator::em {

  inline constexpr char KokkosTuningCacheFormat[] =
      "c8-kokkos-tuning-v1";

  struct KokkosTuningKey {
    std::string backend;
    std::string device_name;
    std::string architecture;
    std::string driver_version;
    std::string runtime_version;
    std::string kokkos_version;
    std::string compiler_version;
    std::string project_revision;
    std::string proposal_table_hash;
    int device{};
    int threads{};
  };

  struct KokkosTuningParameters {
    std::size_t batch_size{};
    std::size_t chunk_size{};
    std::size_t team_size{};
    std::size_t track_tile_size{};
    std::size_t observer_tile_size{};
    std::size_t device_queues{1};
  };

  struct KokkosTuningCacheResult {
    KokkosTuningParameters parameters{};
    std::string content_hash;
    bool matched{};
  };

  KokkosTuningCacheResult loadKokkosTuningCache(
      std::filesystem::path const&, KokkosTuningKey const&,
      bool require_exact_match);

  void writeKokkosTuningCache(
      std::filesystem::path const&, KokkosTuningKey const&,
      KokkosTuningParameters const&);

} // namespace corsika::accelerator::em
