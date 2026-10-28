/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/framework/core/Logging.hpp>

#include <corsika/kokkos/application/DeviceCliOptions.hpp>
#include <CLI/CLI.hpp>
#include <stdexcept>
namespace corsika::kokkos::application {
  void add_device_cli_options(CLI::App& app, BackendOptions& options) {
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
    app.add_option("--kokkos-device", options.kokkosDevice_,
                   "Legacy Kokkos-visible ordinal")
        ->check(CLI::NonNegativeNumber)
        ->group("");
  }

  bool prepare_device_cli(CLI::App const& app, BackendOptions& options,
                          DeviceSelection const& selection) {
    if (selection.ids_.empty()) return true;
    try {
      if (selection.hasMultipleDevices())
        throw std::invalid_argument("This build has no native multi-GPU coordinator");
#if !defined(CORSIKA_KOKKOS_BACKEND_CUDA) && !defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
      throw std::invalid_argument(
          "--device requires a CUDA or CUDA/OpenMP build; omit it on CPU-only systems");
#endif
      if (!app.count("--em-backend")) options.emBackend_ = "kokkos-proposal";
      if (!app.count("--radio-backend") && !options.hostRadioAdapter_)
        options.radioBackend_ = "kokkos";
      if (!app.count("--kokkos-execution")) options.kokkosExecution_ = "cuda";
      if ((options.emBackend_ != "kokkos-proposal" &&
           options.emBackend_ != "kokkos-egs4") ||
          (options.kokkosExecution_ != "cuda" &&
           options.kokkosExecution_ != "cuda-openmp" &&
           options.kokkosExecution_ != "openmp-cuda"))
        throw std::invalid_argument(
            "--device requires accelerated CUDA execution (kokkos-proposal or "
            "kokkos-egs4)");
      select_single_gpu(selection);
      options.kokkosDevice_ = 0;
      return true;
    } catch (std::exception const& error) {
      CORSIKA_LOG_ERROR("GPU selection: {}", error.what());
      return false;
    }
  }
} // namespace corsika::kokkos::application
