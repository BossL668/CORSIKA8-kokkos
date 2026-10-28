/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <corsika/kokkos/application/DeviceSelection.hpp>
namespace CLI {
  class App;
}
namespace corsika::kokkos::application {
  /** Register the same visible device option and hidden compatibility aliases. */
  void add_device_cli_options(CLI::App& app, BackendOptions& options);
  /** Apply the production selection policy; unsupported compiled backends fail. */
  bool prepare_device_cli(CLI::App const& app, BackendOptions& options,
                          DeviceSelection const& selection);
} // namespace corsika::kokkos::application
