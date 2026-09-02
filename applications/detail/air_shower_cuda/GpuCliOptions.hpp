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

namespace corsika::applications::air_shower {

  /**
   * Application-owned acceleration options.
   *
   * This type deliberately contains no CUDA runtime objects.  Keeping the CLI
   * contract independent of the optional backend lets a CPU-only build expose
   * the same help text and reject an explicitly requested CUDA backend with the
   * same fail-closed diagnostic as the production build.
   */
  struct GpuCliOptions {
    std::string em_backend{"proposal"};
    int gpu_device{0};
    std::size_t gpu_min_batch{4096};
    double gpu_memory_fraction{0.70};
    std::filesystem::path gpu_table_cache;
    std::string gpu_physics_source{"c8emrt"};
    std::filesystem::path gpu_aux_cache_dir;
    double gpu_table_tolerance{1.e-3};
    bool gpu_deterministic{true};
    bool gpu_detailed_stage_timing{false};
    bool gpu_full_step_records{false};
    bool gpu_resident_cross_species{true};

    int kokkos_num_threads{0};
    int kokkos_device{0};
    std::filesystem::path kokkos_tuning_cache;
    bool kokkos_require_tuning{false};

    std::string radio_backend{"cpu"};
    double gpu_radio_field_limit{1.};
    bool gpu_radio_track_diagnostics{false};
    double radio_sampling_rate_GHz{1.};
    double radio_window_duration_ns{400.};
    double radio_pretrigger_ns{10.};

    std::string cuda_replay_trace;
    std::filesystem::path cuda_replay_tape_out;

    int hadronic_plan_workers{4};
    double hadronic_plan_target_ms{5.};
    std::size_t hadronic_plan_max_batch{256};
    std::string hadronic_backend{"scalar"};
    int hadronic_workers{4};
    std::size_t hadronic_min_batch{64};
    double hadronic_target_batch_ms{5.};
    std::size_t hadronic_max_batch{256};
    double hadronic_initial_cost_ms{0.1};
    bool cpu_detailed_step_timing{false};
    std::filesystem::path hadronic_worker_executable;
  };

} // namespace corsika::applications::air_shower
