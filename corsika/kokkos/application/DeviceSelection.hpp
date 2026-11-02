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
  /** Return the GPU execution-space name used by the compiled coordinator. */
  [[nodiscard]] std::string gpu_execution();
  /** Return the environment-variable name used to restrict each GPU worker. */
  [[nodiscard]] std::string gpu_visibility_variable();
  /** Path to the HIP query helper; no device runtime is started in this process. */
  [[nodiscard]] std::string hip_device_query_program();
  /**
   * Validate HIP ordinals against the helper's index/PCI-address listing.
   *
   * Returns the requested order unchanged; rejects unknown or duplicate indices.
   */
  [[nodiscard]] std::vector<std::string> resolve_hip_indices(
      std::vector<std::string> const& requested, std::string const& listing);
  /** Select a GPU before runtime startup, querying devices in a child process. */
  void select_single_gpu(DeviceSelection const& selection);
} // namespace corsika::kokkos::application
#include <corsika/detail/kokkos/application/DeviceSelection.inl>
