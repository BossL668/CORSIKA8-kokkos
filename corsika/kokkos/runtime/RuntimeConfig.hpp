/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace corsika::kokkos {

  class RuntimeLease;

  struct RuntimeConfig {
    int device_{0};
    int threads_{0};
    std::filesystem::path tuningCache_;
    bool requireTuning_{};
    // Empty selects the build default. The experimental dual build defaults
    // to CUDA; it does not fall back to OpenMP after a device failure.
    std::string executionBackend_;
    // Internal opt-in; does not enable an application-level hybrid scheduler.
    bool cooperativeOwner_{};
    std::shared_ptr<RuntimeLease> runtimeLease_;
    // Optional diagnostic observer on the coordinator thread. true/false
    // bracket CPU slices submitted while CUDA's stream event is incomplete.
    // Empty in applications; physics and scheduling must not depend on it.
    std::function<void(bool)> cooperativeSliceObserver_;
    // Experimental queue policy; ignored by no backend and rejected on single ends.
    std::string cooperativePolicy_{"legacy"};
  };

} // namespace corsika::kokkos
