/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "GpuCliOptions.hpp"

#include <corsika/framework/core/Logging.hpp>
#include <corsika/validation/CudaReplayTrace.hpp>

#include <filesystem>

namespace corsika::applications::air_shower {

  void addGpuCliOptions(CLI::App& app, GpuCliOptions& options) {
    app.add_option("--em-backend", options.em_backend,
                   "Electromagnetic transport backend: proposal or cuda")
        ->check(CLI::IsMember({"proposal", "cuda"}))
        ->group("GPU EM");
    app.add_option("--gpu-device", options.gpu_device, "CUDA device index")
        ->check(CLI::NonNegativeNumber)
        ->group("GPU EM");
    app.add_option("--gpu-min-batch", options.gpu_min_batch,
                   "Minimum CUDA execution batch; smaller fronts take one "
                   "scalar expansion step")
        ->check(CLI::PositiveNumber)
        ->group("GPU EM");
    app.add_option(
           "--gpu-memory-fraction", options.gpu_memory_fraction,
           "Fraction of currently free device memory available to CUDA EM")
        ->check(CLI::Range(0.01, 1.0))
        ->group("GPU EM");
    app.add_option(
           "--gpu-table-cache", options.gpu_table_cache,
           "Versioned .c8emrt table required when --gpu-physics-source=c8emrt")
        ->group("GPU EM");
    app.add_option("--gpu-physics-source", options.gpu_physics_source,
                   "GPU physics source: c8emrt or proposal-native")
        ->check(CLI::IsMember({"c8emrt", "proposal-native"}))
        ->group("GPU EM");
    app.add_option(
           "--gpu-aux-cache-dir", options.gpu_aux_cache_dir,
           "XDG-compatible .c8emaux cache directory for proposal-native")
        ->group("GPU EM");
    app.add_option("--gpu-table-tolerance", options.gpu_table_tolerance,
                   "Maximum accepted relative table error")
        ->check(CLI::Range(1.e-8, 1.0))
        ->group("GPU EM");
    app.add_option("--gpu-deterministic", options.gpu_deterministic,
                   "Use history-keyed deterministic CUDA random numbers")
        ->group("GPU EM");
    app.add_flag(
           "--gpu-detailed-stage-timing",
           options.gpu_detailed_stage_timing,
           "Record per-stage CUDA event timings for the fused lepton "
           "pipeline (profiling only)")
        ->group("GPU EM");
    app.add_flag("--gpu-full-step-records", options.gpu_full_step_records,
                 "Disable compact GPU profile projection and return full "
                 "transport records (validation/debug only)")
        ->group("GPU EM");
    app.add_option(
           "--gpu-resident-cross-species",
           options.gpu_resident_cross_species,
           "Keep photon-to-lepton and lepton-to-photon secondaries in "
           "persistent device queues")
        ->group("GPU EM");
    app.add_option(
           "--radio-backend", options.radio_backend,
           "Radio projection backend for CUDA EM tracks: cpu or cuda")
        ->check(CLI::IsMember({"cpu", "cuda"}))
        ->group("Radio");
    app.add_option(
           "--gpu-radio-field-limit", options.gpu_radio_field_limit,
           "Checked fixed-point waveform range in V/m for deterministic "
           "CUDA CoREAS/ZHS accumulation")
        ->check(CLI::PositiveNumber)
        ->group("Radio");
    app.add_flag(
           "--gpu-radio-track-diagnostics",
           options.gpu_radio_track_diagnostics,
           "Collect scalar-compatible electron/positron track diagnostics "
           "on the GPU (validation only; adds reduction overhead)")
        ->group("Radio");
    app.add_option(
           "--radio-sampling-rate-ghz", options.radio_sampling_rate_GHz,
           "Time-domain radio sampling rate in GHz. Use at least 10 GHz "
           "(0.1 ns bins) for a 50--350 MHz CoREAS/ZHS comparison")
        ->check(CLI::Range(0.1, 1000.))
        ->group("Radio");
    app.add_option("--radio-window-duration-ns",
                   options.radio_window_duration_ns,
                   "Time-domain radio observer window duration in ns")
        ->check(CLI::Range(1., 1.e6))
        ->group("Radio");
    app.add_option(
           "--radio-pretrigger-ns", options.radio_pretrigger_ns,
           "Start antenna-file radio windows this many ns before the "
           "geometrical direct-arrival time")
        ->check(CLI::Range(0., 1.e6))
        ->group("Radio");
    app.add_option(
           "--cuda-replay-trace", options.cuda_replay_trace,
           "Write a process-level CSV trace for scalar/CUDA replay comparison")
        ->group("GPU EM");
    app.add_option(
           "--cuda-replay-tape-out", options.cuda_replay_tape_out,
           "Record every scalar transport segment and the exact radio "
           "observer snapshot for offline CUDA decision replay")
        ->group("GPU EM");
    app.add_option(
           "--hadronic-plan-workers", options.hadronic_plan_workers,
           "Worker count used for the retrospective hadronic final-state "
           "batch oracle (does not enable parallel execution)")
        ->check(CLI::Range(1, 256))
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-plan-target-ms", options.hadronic_plan_target_ms,
           "Target measured final-state cost per retrospective hadronic batch")
        ->check(CLI::Range(1.e-6, 1.e6))
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-plan-max-batch", options.hadronic_plan_max_batch,
           "Maximum interactions per retrospective homogeneous hadronic batch")
        ->check(CLI::PositiveNumber)
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-backend", options.hadronic_backend,
           "Low-energy hadronic final-state backend: scalar or fluka-process")
        ->check(CLI::IsMember({"scalar", "fluka-process"}))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-workers", options.hadronic_workers,
                   "Persistent FLUKA worker process count")
        ->check(CLI::Range(1, 256))
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-min-batch", options.hadronic_min_batch,
           "Minimum parked FLUKA vertices before the predicted-cost flush test")
        ->check(CLI::PositiveNumber)
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-target-batch-ms",
                   options.hadronic_target_batch_ms,
                   "Online estimated FLUKA time per homogeneous worker batch")
        ->check(CLI::Range(1.e-6, 1.e6))
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-max-batch", options.hadronic_max_batch,
           "Maximum FLUKA requests in one homogeneous process-worker batch")
        ->check(CLI::PositiveNumber)
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-initial-cost-ms", options.hadronic_initial_cost_ms,
           "Initial per-interaction cost before a class has worker timing samples")
        ->check(CLI::Range(1.e-6, 1.e6))
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-worker-executable",
           options.hadronic_worker_executable,
           "Path to fluka_batch_worker; empty selects the executable beside c8_air_shower")
        ->group("Hadronic scheduling");
    app.add_flag(
           "--cpu-detailed-step-timing", options.cpu_detailed_step_timing,
           "Record scalar cross-section/tracking/continuous/discrete phase times")
        ->group("Hadronic scheduling");
  }

  bool validateGpuCliOptions(GpuCliOptions& options,
                             std::filesystem::path const& executable) {
    if (!options.cuda_replay_trace.empty()) {
      options.gpu_full_step_records = true;
      validation::CudaReplayTrace::instance().open(
          options.cuda_replay_trace,
          options.em_backend == "cuda" ? "cuda" : "refactor_proposal");
    }
    if (!options.cuda_replay_tape_out.empty() &&
        options.em_backend != "proposal") {
      CORSIKA_LOG_CRITICAL(
          "--cuda-replay-tape-out records the scalar decision oracle and "
          "therefore requires --em-backend proposal");
      return false;
    }

    if (options.em_backend == "cuda") {
#ifndef CORSIKA8_WITH_CUDA_EM
      CORSIKA_LOG_CRITICAL(
          "--em-backend cuda was requested, but this c8_air_shower binary was "
          "built without CORSIKA_ENABLE_CUDA");
      return false;
#else
      if (options.gpu_physics_source == "c8emrt" &&
          options.gpu_table_cache.empty()) {
        CORSIKA_LOG_CRITICAL(
            "--gpu-physics-source c8emrt requires --gpu-table-cache");
        return false;
      }
      if (options.gpu_physics_source == "c8emrt" &&
          !std::filesystem::is_regular_file(options.gpu_table_cache)) {
        CORSIKA_LOG_CRITICAL("CUDA EM table cache is not a regular file: {}",
                             options.gpu_table_cache.string());
        return false;
      }
#endif
    }
    if (options.radio_backend == "cuda") {
      if (options.em_backend != "cuda") {
        CORSIKA_LOG_CRITICAL(
            "--radio-backend cuda requires --em-backend cuda");
        return false;
      }
#ifndef CORSIKA8_WITH_CUDA_EM
      CORSIKA_LOG_CRITICAL(
          "--radio-backend cuda was requested, but CUDA support is unavailable");
      return false;
#endif
    }
    if (options.hadronic_backend == "fluka-process") {
      if (options.em_backend != "cuda") {
        CORSIKA_LOG_CRITICAL(
            "--hadronic-backend fluka-process currently requires "
            "--em-backend cuda and HybridCascade");
        return false;
      }
#ifndef WITH_FLUKA
      CORSIKA_LOG_CRITICAL(
          "--hadronic-backend fluka-process requires a WITH_FLUKA build");
      return false;
#endif
#ifndef CORSIKA8_WITH_CUDA_EM
      CORSIKA_LOG_CRITICAL(
          "--hadronic-backend fluka-process requires the CUDA HybridCascade build");
      return false;
#endif
      if (options.hadronic_worker_executable.empty()) {
        options.hadronic_worker_executable =
            std::filesystem::absolute(executable).parent_path() /
            "fluka_batch_worker";
      }
      if (!std::filesystem::is_regular_file(
              options.hadronic_worker_executable)) {
        CORSIKA_LOG_CRITICAL(
            "FLUKA worker executable is not a regular file: {}",
            options.hadronic_worker_executable.string());
        return false;
      }
    }
    return true;
  }

} // namespace corsika::applications::air_shower
