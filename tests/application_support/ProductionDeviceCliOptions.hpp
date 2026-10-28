/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "ProductionBackendOptions.hpp"
#include "ProductionDeviceSelection.hpp"
#include <CLI/CLI.hpp>
#include <iostream>

namespace production_application::air_shower {
  inline void addDeviceCliOptions(CLI::App& app, GpuCliOptions& options) {
    app.add_option("--device",
                   "NVIDIA GPU index/UUID(s): one runs directly; several cooperate on "
                   "each shower (0,1 or 0 1)")
        ->expected(1, 255)
        ->type_size(1)
        ->delimiter(',')
        ->group("Kokkos");
    app.add_option("--devices", "Compatibility alias for --device")
        ->expected(1, 255)
        ->type_size(1)
        ->delimiter(',')
        ->group("");
    // Internal workers select ordinal zero inside their UUID visibility mask.
    // Keep this legacy option separate: it must NOT select physical GPU zero.
    app.add_option("--kokkos-device", options.kokkos_device,
                   "Legacy Kokkos-visible ordinal")
        ->check(CLI::NonNegativeNumber)
        ->group("");
  }

  inline bool prepareDeviceCli(CLI::App const& app, GpuCliOptions& options,
                               gpu_cli::DeviceSelection const& selection) {
    if (selection.ids.empty()) return true;
    try {
      if (selection.multiple())
        throw std::invalid_argument("This build has no native multi-GPU coordinator");
#if !defined(CORSIKA8_KOKKOS_BACKEND_CUDA) && \
    !defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
      throw std::invalid_argument(
          "--device requires a CUDA or CUDA/OpenMP build; omit it on CPU-only systems");
#endif
      if (!app.count("--em-backend")) options.em_backend = "kokkos-proposal";
      if (!app.count("--radio-backend")) options.radio_backend = "kokkos";
      if (!app.count("--kokkos-execution")) options.kokkos_execution = "cuda";
      if ((options.em_backend != "kokkos-proposal" &&
           options.em_backend != "kokkos-egs4") ||
          (options.kokkos_execution != "cuda" &&
           options.kokkos_execution != "cuda-openmp" &&
           options.kokkos_execution != "openmp-cuda"))
        throw std::invalid_argument(
            "--device requires accelerated CUDA execution (kokkos-proposal or "
            "kokkos-egs4)");
      gpu_cli::selectSingleGpu(selection);
      options.kokkos_device = 0;
      return true;
    } catch (std::exception const& error) {
      std::cerr << "GPU selection: " << error.what() << '\n';
      return false;
    }
  }
} // namespace production_application::air_shower
