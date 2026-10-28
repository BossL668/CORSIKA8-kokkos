/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <string>
#include <vector>
namespace corsika::kokkos::application {
  /** Host-only GPU selection, resolved before model or backend initialization. */
  struct DeviceSelection {
    std::vector<std::string> ids_;
    bool hasMultipleDevices() const;
  };
  /** Parse the production device aliases, including the help and -- guards. */
  DeviceSelection parse_devices(int argc, char const* const* argv);
  /** Resolve physical indices and UUID aliases without initializing CUDA. */
  std::vector<std::string> resolve_gpu_ids(std::vector<std::string> const& requested,
                                           std::string const& listing);
  /** Query nvidia-smi and set the selected UUID visibility before runtime startup. */
  void select_single_gpu(DeviceSelection const& selection);
} // namespace corsika::kokkos::application
#include <corsika/detail/kokkos/application/DeviceSelection.inl>
