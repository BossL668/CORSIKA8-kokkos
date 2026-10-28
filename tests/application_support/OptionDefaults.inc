/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/application/BackendOptions.hpp>
#include "ProductionBackendOptions.hpp"
void check_options(corsika::kokkos::application::BackendOptions const& candidate,
                   production_application::air_shower::GpuCliOptions const& original) {
  CHECK(candidate.emBackend_ == original.em_backend);
  CHECK(candidate.egs4Stepfc_ == original.egs4_stepfc);
  CHECK(candidate.gpuDevice_ == original.gpu_device);
  CHECK(candidate.gpuMinBatch_ == original.gpu_min_batch);
  CHECK(candidate.gpuResidentBatchLimit_ == original.gpu_resident_batch_limit);
  CHECK(candidate.gpuMemoryFraction_ == original.gpu_memory_fraction);
  CHECK(candidate.gpuPhysicsSource_ == original.gpu_physics_source);
  CHECK(candidate.gpuAuxCacheDir_ == original.gpu_aux_cache_dir);
  CHECK(candidate.gpuDeterministic_ == original.gpu_deterministic);
  CHECK(candidate.gpuDetailedStageTiming_ == original.gpu_detailed_stage_timing);
  CHECK(candidate.gpuFullStepRecords_ == original.gpu_full_step_records);
  CHECK(candidate.gpuResidentCrossSpecies_ == original.gpu_resident_cross_species);
  CHECK(candidate.kokkosNumThreads_ == original.kokkos_num_threads);
  CHECK(candidate.kokkosExecution_ == original.kokkos_execution);
  CHECK(candidate.kokkosCooperativePolicy_ == original.kokkos_cooperative_policy);
  CHECK(candidate.kokkosDevice_ == original.kokkos_device);
  CHECK(candidate.kokkosTuningCache_ == original.kokkos_tuning_cache);
  CHECK(candidate.kokkosRequireTuning_ == original.kokkos_require_tuning);
  CHECK(candidate.radioBackend_ == original.radio_backend);
  CHECK(candidate.radioTrackInputs_ == original.radio_track_inputs);
  CHECK(candidate.gpuRadioFieldLimit_ == original.gpu_radio_field_limit);
  CHECK(candidate.gpuRadioTrackDiagnostics_ == original.gpu_radio_track_diagnostics);
  CHECK(candidate.radioSamplingRateGHz_ == original.radio_sampling_rate_GHz);
  CHECK(candidate.radioWindowDurationNs_ == original.radio_window_duration_ns);
  CHECK(candidate.radioPretriggerNs_ == original.radio_pretrigger_ns);
  CHECK(candidate.cudaReplayTrace_ == original.cuda_replay_trace);
  CHECK(candidate.cudaReplayTapeOut_ == original.cuda_replay_tape_out);
  CHECK(candidate.hadronicPlanWorkers_ == original.hadronic_plan_workers);
  CHECK(candidate.hadronicPlanTargetMs_ == original.hadronic_plan_target_ms);
  CHECK(candidate.hadronicPlanMaxBatch_ == original.hadronic_plan_max_batch);
  CHECK(candidate.hadronicBackend_ == original.hadronic_backend);
  CHECK(candidate.hadronicWorkers_ == original.hadronic_workers);
  CHECK(candidate.hadronicMinBatch_ == original.hadronic_min_batch);
  CHECK(candidate.hadronicTargetBatchMs_ == original.hadronic_target_batch_ms);
  CHECK(candidate.hadronicMaxBatch_ == original.hadronic_max_batch);
  CHECK(candidate.hadronicInitialCostMs_ == original.hadronic_initial_cost_ms);
  CHECK(candidate.cpuDetailedStepTiming_ == original.cpu_detailed_step_timing);
  CHECK(candidate.hadronicWorkerExecutable_ == original.hadronic_worker_executable);
}
void check_defaults() { check_options({}, {}); }
