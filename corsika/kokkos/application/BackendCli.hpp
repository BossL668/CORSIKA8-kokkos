/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/application/BackendOptions.hpp>
#include <filesystem>
namespace CLI {
  class App;
}
namespace corsika::kokkos::application {
  void add_egs4_cli_options(CLI::App& app, BackendOptions& options);
  bool prepare_egs4_cli(CLI::App const& app, BackendOptions& options);
  bool validate_egs4_execution(BackendOptions const& options);
  void add_backend_cli_options(CLI::App& app, BackendOptions& options);
  bool validate_backend_cli_options(BackendOptions& options,
                                    std::filesystem::path const& executable);
} // namespace corsika::kokkos::application
