/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <filesystem>
#include <string>
#include <corsika/kokkos/runtime/TuningParameters.hpp>

namespace corsika::kokkos {

  inline constexpr char tuningCacheFormat[] = "c8-kokkos-tuning-v1";

  struct TuningKey {
    std::string backend_;
    std::string deviceName_;
    std::string architecture_;
    std::string driverVersion_;
    std::string runtimeVersion_;
    std::string kokkosVersion_;
    std::string compilerVersion_;
    std::string projectRevision_;
    std::string proposalTableHash_;
    int device_{};
    int threads_{};
  };

  struct TuningCacheResult {
    TuningParameters parameters_{};
    std::string contentHash_;
    bool matched_{};
  };

  TuningCacheResult load_tuning_cache(std::filesystem::path const&, TuningKey const&,
                                      bool requireExactMatch);

  void write_tuning_cache(std::filesystem::path const&, TuningKey const&,
                          TuningParameters const&);

} // namespace corsika::kokkos
