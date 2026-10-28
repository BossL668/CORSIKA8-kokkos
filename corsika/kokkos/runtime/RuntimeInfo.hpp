/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <string>
#include <corsika/kokkos/runtime/BackendKind.hpp>

namespace corsika::kokkos {

  struct RuntimeInfo {
    BackendKind kind_{BackendKind::OpenMP};
    std::string backend_;
    std::string kokkosVersion_;
    std::string deviceName_;
    std::string architecture_;
    std::string driverVersion_;
    std::string runtimeVersion_;
    std::string compilerVersion_;
    std::string projectRevision_;
    int device_{};
    int concurrency_{};
    int hostThreads_{1};
    std::size_t deviceTotalMemoryBytes_{};
    std::size_t deviceFreeMemoryBytesAtInitialization_{};
    bool gpu_{};
    bool openmp_{};
    bool cooperativeRuntime_{};
  };

} // namespace corsika::kokkos
