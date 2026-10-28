/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/application/BackendCli.hpp>
#include <corsika/kokkos/application/DeviceCliOptions.hpp>
#include <corsika/output/ReplayTrace.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <CLI/CLI.hpp>
#include <stdexcept>
#if defined(CORSIKA8_WITH_KOKKOS_EM) || defined(CORSIKA8_WITH_NATIVE_EGS4)
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#endif
namespace corsika::kokkos::application {
  void add_egs4_cli_options(CLI::App& app, BackendOptions& options) {
    app.add_option("--egs4-stepfc", options.egs4Stepfc_,
                   "C7-EGS4 charged-particle step factor (default: 1)")
        ->check(CLI::PositiveNumber)
        ->group("EGS4 EM");
  }

  bool prepare_egs4_cli(CLI::App const& app, BackendOptions& options) {
    if (options.emBackend_ != "kokkos-egs4") {
      if (app.count("--egs4-stepfc")) {
        CORSIKA_LOG_CRITICAL("--egs4-stepfc requires --em-backend kokkos-egs4");
        return false;
      }
      return true;
    }
    if (!app.count("--radio-backend")) options.radioBackend_ = "kokkos";
    if (app.count("--pdg") &&
        (app.count("--force-interaction") || app.count("--force-decay"))) {
      int id = app["--pdg"]->as<int>();
      if (id == 11 || id == -11 || id == 22) {
        CORSIKA_LOG_CRITICAL(
            "Forced scalar primary actions are not supported for EGS4 EM primaries");
        return false;
      }
    }
    return true;
  }

  bool validate_egs4_execution(BackendOptions const& options) {
    if (options.emBackend_ != "kokkos-egs4") return true;
#ifndef CORSIKA8_WITH_NATIVE_EGS4
    CORSIKA_LOG_CRITICAL("EGS4 is not compiled; build with CORSIKA_ENABLE_EGS4=ON");
    return false;
#else
    if (options.hadronicBackend_ != "scalar" || !options.cudaReplayTrace_.empty()) {
      CORSIKA_LOG_CRITICAL(
          "EGS4 requires scalar hadronic scheduling and does not support PROPOSAL replay "
          "traces");
      return false;
    }
    try {
      auto execution = kokkos::resolve_execution_backend(options.kokkosExecution_);
      if (execution != "cuda" && execution != "openmp")
        throw std::invalid_argument(
            "EGS4 supports cuda or openmp, not cooperative execution");
      if (execution == "cuda" && options.kokkosNumThreads_ > 1)
        throw std::invalid_argument("CUDA uses one host scheduling thread");
    } catch (std::exception const& error) {
      CORSIKA_LOG_CRITICAL("{}", error.what());
      return false;
    }
    return true;
#endif
  }

  void add_backend_cli_options(CLI::App& app, BackendOptions& options) {
    app.add_option("--em-backend", options.emBackend_,
                   "Electromagnetic transport backend: proposal (native CPU), "
                   "kokkos-proposal, or kokkos-egs4")
        ->check(CLI::IsMember({"proposal", "kokkos-proposal", "kokkos-egs4"}))
        ->group("GPU EM");
    corsika::kokkos::application::add_egs4_cli_options(app, options);
    app.add_option("--gpu-min-batch", options.gpuMinBatch_,
                   "Minimum Kokkos execution batch; smaller fronts take one "
                   "scalar expansion step")
        ->check(CLI::PositiveNumber)
        ->group("GPU EM");
    app.add_option("--gpu-resident-batch-limit", options.gpuResidentBatchLimit_,
                   "Optional common resident-wavefront checkpoint for deterministic "
                   "replay; zero selects the memory-budgeted automatic limit")
        ->check(CLI::NonNegativeNumber)
        ->group("GPU EM");
    app.add_option(
           "--gpu-memory-fraction", options.gpuMemoryFraction_,
           "Fraction of currently free device memory available to the Kokkos backend")
        ->check(CLI::Range(0.01, 1.0))
        ->group("GPU EM");
    app.add_option("--gpu-physics-source", options.gpuPhysicsSource_,
                   "PROPOSAL backend physics source: proposal-native")
        ->check(CLI::IsMember({"proposal-native"}))
        ->group("GPU EM");
    app.add_option("--gpu-aux-cache-dir", options.gpuAuxCacheDir_,
                   "XDG-compatible .c8emaux cache directory for proposal-native")
        ->group("GPU EM");
    app.add_option("--gpu-deterministic", options.gpuDeterministic_,
                   "Use history-keyed deterministic random numbers")
        ->group("GPU EM");
    app.add_flag("--gpu-detailed-stage-timing", options.gpuDetailedStageTiming_,
                 "Record per-stage CUDA event timings for the fused lepton "
                 "pipeline (profiling only)")
        ->group("GPU EM");
    app.add_flag("--gpu-full-step-records", options.gpuFullStepRecords_,
                 "Disable compact GPU profile projection and return full "
                 "transport records (validation/debug only)")
        ->group("GPU EM");
    app.add_option("--gpu-resident-cross-species", options.gpuResidentCrossSpecies_,
                   "Keep photon-to-lepton and lepton-to-photon secondaries in "
                   "persistent device queues")
        ->group("GPU EM");
    app.add_option("--kokkos-execution", options.kokkosExecution_,
                   "Execution space in this binary: cuda, openmp, cuda-openmp (GPU "
                   "primary) or openmp-cuda (CPU primary) for the "
                   "experimental dual build; empty selects the build default")
        ->check(CLI::IsMember(
            {"cuda", "openmp", "hip", "sycl", "cuda-openmp", "openmp-cuda"}))
        ->group("GPU EM");
    app.add_option("--kokkos-cooperative-policy", options.kokkosCooperativePolicy_,
                   "Cooperative queue scheduling: legacy or experimental adaptive")
        ->check(CLI::IsMember({"legacy", "adaptive"}))
        ->group("Kokkos");
    app.add_option("--kokkos-num-threads", options.kokkosNumThreads_,
                   "OpenMP thread count for an OpenMP-only Kokkos build; zero uses the "
                   "runtime default")
        ->check(CLI::NonNegativeNumber)
        ->group("Kokkos");
    corsika::kokkos::application::add_device_cli_options(app, options);
    app.add_option("--kokkos-tuning-cache", options.kokkosTuningCache_,
                   "Device-specific Kokkos tuning cache")
        ->group("Kokkos");
    app.add_flag("--kokkos-require-tuning", options.kokkosRequireTuning_,
                 "Fail unless an exactly matching Kokkos tuning cache exists")
        ->group("Kokkos");
    app.add_option("--radio-backend", options.radioBackend_,
                   "Radio projection backend: cpu or kokkos")
        ->check(CLI::IsMember({"cpu", "kokkos"}))
        ->group("Radio");
    app.add_option("--gpu-radio-field-limit", options.gpuRadioFieldLimit_,
                   "Checked fixed-point waveform range in V/m for deterministic "
                   "CUDA CoREAS/ZHS accumulation")
        ->check(CLI::PositiveNumber)
        ->group("Radio");
    app.add_flag("--gpu-radio-track-diagnostics", options.gpuRadioTrackDiagnostics_,
                 "Collect scalar-compatible electron/positron track diagnostics "
                 "on the GPU (validation only; adds reduction overhead)")
        ->group("Radio");
    app.add_option("--radio-sampling-rate-ghz", options.radioSamplingRateGHz_,
                   "Time-domain radio sampling rate in GHz. Use at least 10 GHz "
                   "(0.1 ns bins) for a 50--350 MHz CoREAS/ZHS comparison")
        ->check(CLI::Range(0.1, 1000.))
        ->group("Radio");
    app.add_option("--radio-window-duration-ns", options.radioWindowDurationNs_,
                   "Time-domain radio observer window duration in ns")
        ->check(CLI::Range(1., 1.e6))
        ->group("Radio");
    app.add_option("--radio-pretrigger-ns", options.radioPretriggerNs_,
                   "Start antenna-file radio windows this many ns before the "
                   "geometrical direct-arrival time")
        ->check(CLI::Range(0., 1.e6))
        ->group("Radio");
    app.add_option("--cuda-replay-trace", options.cudaReplayTrace_,
                   "Write a process-level CSV trace for scalar/CUDA replay comparison")
        ->group("GPU EM");
    app.add_option("--cuda-replay-tape-out", options.cudaReplayTapeOut_,
                   "Record every scalar transport segment and the exact radio "
                   "observer snapshot for offline CUDA decision replay")
        ->group("GPU EM");
    app.add_option("--hadronic-plan-workers", options.hadronicPlanWorkers_,
                   "Worker count used for the retrospective hadronic final-state "
                   "batch oracle (does not enable parallel execution)")
        ->check(CLI::Range(1, 256))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-plan-target-ms", options.hadronicPlanTargetMs_,
                   "Target measured final-state cost per retrospective hadronic batch")
        ->check(CLI::Range(1.e-6, 1.e6))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-plan-max-batch", options.hadronicPlanMaxBatch_,
                   "Maximum interactions per retrospective homogeneous hadronic batch")
        ->check(CLI::PositiveNumber)
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-backend", options.hadronicBackend_,
                   "Low-energy hadronic final-state backend: scalar or fluka-process")
        ->check(CLI::IsMember({"scalar"}))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-workers", options.hadronicWorkers_,
                   "Persistent FLUKA worker process count")
        ->check(CLI::Range(1, 256))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-min-batch", options.hadronicMinBatch_,
                   "Minimum parked FLUKA vertices before the predicted-cost flush test")
        ->check(CLI::PositiveNumber)
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-target-batch-ms", options.hadronicTargetBatchMs_,
                   "Online estimated FLUKA time per homogeneous worker batch")
        ->check(CLI::Range(1.e-6, 1.e6))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-max-batch", options.hadronicMaxBatch_,
                   "Maximum FLUKA requests in one homogeneous process-worker batch")
        ->check(CLI::PositiveNumber)
        ->group("Hadronic scheduling");
    app.add_option(
           "--hadronic-initial-cost-ms", options.hadronicInitialCostMs_,
           "Initial per-interaction cost before a class has worker timing samples")
        ->check(CLI::Range(1.e-6, 1.e6))
        ->group("Hadronic scheduling");
    app.add_option("--hadronic-worker-executable", options.hadronicWorkerExecutable_,
                   "Path to fluka_batch_worker; empty selects the executable beside "
                   "c8_air_shower")
        ->group("Hadronic scheduling");
    app.add_flag("--cpu-detailed-step-timing", options.cpuDetailedStepTiming_,
                 "Record scalar cross-section/tracking/continuous/discrete phase times")
        ->group("Hadronic scheduling");
  }

  bool validate_backend_cli_options(BackendOptions& options,
                                    std::filesystem::path const& /* executable */) {
    if (!options.kokkosExecution_.empty() && options.emBackend_ != "kokkos-proposal" &&
        options.emBackend_ != "kokkos-egs4") {
      CORSIKA_LOG_CRITICAL(
          "--kokkos-execution requires --em-backend kokkos-proposal or kokkos-egs4");
      return false;
    }
    if (!options.cudaReplayTrace_.empty()) {
      options.gpuFullStepRecords_ = true;
      ReplayTrace::instance().open(
          options.cudaReplayTrace_,
          options.emBackend_ == "proposal" ? "refactor_proposal" : options.emBackend_);
    }
    if (!options.cudaReplayTapeOut_.empty() && options.emBackend_ != "proposal") {
      CORSIKA_LOG_CRITICAL(
          "--cuda-replay-tape-out records the scalar decision oracle and "
          "therefore requires --em-backend proposal");
      return false;
    }

    if (options.kokkosCooperativePolicy_ != "legacy" &&
        (options.emBackend_ != "kokkos-proposal" ||
         options.kokkosExecution_ != "cuda-openmp")) {
      CORSIKA_LOG_CRITICAL("Adaptive scheduling requires Kokkos cuda-openmp execution");
      return false;
    }
    if (!corsika::kokkos::application::validate_egs4_execution(options)) return false;
    if (options.emBackend_ == "kokkos-proposal") {
#ifndef CORSIKA8_WITH_KOKKOS_EM
      CORSIKA_LOG_CRITICAL(
          "--em-backend kokkos-proposal was requested, but this c8_air_shower binary "
          "was built without CORSIKA_ENABLE_KOKKOS");
      return false;
#else
      if (options.gpuPhysicsSource_ != "proposal-native") {
        CORSIKA_LOG_CRITICAL(
            "--em-backend kokkos-proposal requires --gpu-physics-source proposal-native");
        return false;
      }
      if (options.radioBackend_ != "kokkos" &&
          !(options.hostRadioAdapter_ && options.radioBackend_ == "cpu")) {
        CORSIKA_LOG_CRITICAL(
            "--em-backend kokkos-proposal requires --radio-backend kokkos; portable "
            "EM and radio execution spaces cannot be mixed");
        return false;
      }
      if (options.hadronicBackend_ != "scalar") {
        CORSIKA_LOG_CRITICAL(
            "Kokkos EM is mutually exclusive with the shower-internal "
            "hadronic worker pool; use --hadronic-backend scalar");
        return false;
      }
      std::string selectedExecution;
      try {
        selectedExecution = kokkos::resolve_execution_backend(options.kokkosExecution_);
      } catch (std::exception const& error) {
        CORSIKA_LOG_CRITICAL("{}", error.what());
        return false;
      }
      if ((selectedExecution == "cuda-openmp" || selectedExecution == "openmp-cuda") &&
          options.hadronicWorkers_ != 1) {
        CORSIKA_LOG_CRITICAL(
            "Cooperative EM requires scalar hadrons: --hadronic-workers must be 1");
        return false;
      }
      if (selectedExecution != "openmp" && selectedExecution != "cuda-openmp" &&
          selectedExecution != "openmp-cuda" && options.kokkosNumThreads_ > 1) {
        CORSIKA_LOG_CRITICAL(
            "A GPU Kokkos build uses Serial host scheduling and rejects "
            "--kokkos-num-threads > 1");
        return false;
      }
#endif
    }

    if (options.radioBackend_ == "kokkos") {
      if (options.emBackend_ != "kokkos-proposal" &&
          options.emBackend_ != "kokkos-egs4" && !options.hostRadioAdapter_) {
        CORSIKA_LOG_CRITICAL(
            "--radio-backend kokkos requires --em-backend kokkos-proposal or "
            "kokkos-egs4");
        return false;
      }
#ifndef CORSIKA8_WITH_KOKKOS_EM
      CORSIKA_LOG_CRITICAL(
          "--radio-backend kokkos was requested, but Kokkos support is unavailable");
      return false;
#endif
    }

    return true;
  }
} // namespace corsika::kokkos::application
