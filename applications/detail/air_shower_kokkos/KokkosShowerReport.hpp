/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "GpuCliOptions.hpp"

#include <corsika/framework/core/HadronicProcessPool.hpp>
#include <corsika/framework/core/HadronicWorkQueue.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/accelerator/em/common/CorsikaOutputSink.hpp>
#include <corsika/accelerator/em/PhysicalAcceleratedEmRouter.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/accelerator/em/common/ProposalCpuFallbackHandler.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/tables/Sha256.hpp>
#include <corsika/validation/CudaReplayTrace.hpp>

#ifdef WITH_FLUKA
#include <corsika/modules/FLUKA.hpp>
#endif

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace corsika::applications::air_shower {

  /**
   * Serialize one completed hybrid shower without owning or changing any
   * transport state.  The statement order intentionally mirrors the original
  * application so the YAML schema and insertion order stay stable.
   */
  struct KokkosShowerReportBuilder {
    template <
      typename TGpuEmStepRegistry, typename TSequence, typename TRunSession,
      typename TBackend, typename TEas,
      typename TRouter, typename TOutputSink, typename TFallbackHandler,
      typename TPhotoHighEnergy, typename TPhotoLowEnergy,
      typename TPhotoHighStatistics, typename TPhotoLowStatistics,
      typename THighEnergyCounter, typename TLowEnergyCounter,
      typename TPhotoFallback>
    static void record(
      TRunSession& cuda_session, GpuCliOptions const& gpu_cli,
      TBackend& backend, TEas& EAS, TRouter& router,
      TOutputSink& output_sink, TFallbackHandler& fallback_handler,
      TPhotoHighEnergy& photoHadronicHighEnergy,
      TPhotoLowEnergy& photoHadronicLowEnergy,
      TPhotoHighStatistics const& photo_hadronic_before,
      TPhotoLowStatistics const& photo_hadronic_le_before,
      THighEnergyCounter& heCounted, TLowEnergyCounter& leIntCounted,
      std::uint64_t high_energy_hadronic_interactions_before,
      std::uint64_t low_energy_hadronic_interactions_before,
      std::size_t high_energy_hadronic_timings_before,
      std::size_t low_energy_hadronic_timings_before,
      HadronicWorkClassifierConfig const& hadronic_work_classifier,
      HadronicProcessPool* hadronic_process_pool,
      HadronicProcessPoolStatistics const& hadronic_pool_statistics_before,
      TPhotoFallback const& photoHadronicQgsjetFallback,
      std::string const& high_energy_hadronic_model,
      HEPEnergyType heHadronModelThreshold, double emthinfrac,
      double maxWeight, bool automaticMaxWeight,
      bool thinningCanActivateFromUnitWeight,
      bool backend_reused_for_shower, bool gpu_muon_transport_available,
      bool gpu_radio_enabled, gpu::em::GpuEmConfig const& gpu_config,
      Code beamCode, HEPEnergyType primaryTotalEnergy,
      unsigned int output_shower_id) {
    using GpuEmStepRegistry = TGpuEmStepRegistry;
    using Sequence = TSequence;
    using namespace corsika::gpu::em;
    using namespace corsika::gpu::em::tables;

    auto const hadronic_plan_workers = gpu_cli.hadronic_plan_workers;
    auto const hadronic_plan_target_ms = gpu_cli.hadronic_plan_target_ms;
    auto const hadronic_plan_max_batch = gpu_cli.hadronic_plan_max_batch;
    auto const& hadronic_backend = gpu_cli.hadronic_backend;
    auto const hadronic_min_batch = gpu_cli.hadronic_min_batch;
    auto const hadronic_target_batch_ms =
        gpu_cli.hadronic_target_batch_ms;
    auto const hadronic_max_batch = gpu_cli.hadronic_max_batch;
    auto const hadronic_initial_cost_ms =
        gpu_cli.hadronic_initial_cost_ms;
      auto const gpu_deterministic = gpu_cli.gpu_deterministic;
      auto const gpu_radio_field_limit = gpu_cli.gpu_radio_field_limit;
      auto const accelerated_backend_name =
          gpu_cli.em_backend == "kokkos" ? "kokkos" : "cuda";

      for (auto const& record : router.stepRecords()) {
        validation::CudaReplayTrace::instance().recordGpuStep(
            record.history_id, record.step_id, record.pid,
            record.process_id, record.start_energy_GeV,
            record.end_energy_GeV, record.deposited_energy_GeV,
            record.weight, record.start_position_m[0],
            record.start_position_m[1], record.start_position_m[2],
            record.start_time_s);
      }

      auto& replay_trace = validation::CudaReplayTrace::instance();
      auto const& interaction_records = router.interactionRecords();
      for (auto const& record : interaction_records) {
        replay_trace.recordGpuInteraction(record);
      }
      std::vector<bool> interaction_claimed(
          interaction_records.size(), false);
      std::unordered_map<std::uint64_t, std::vector<std::size_t>>
          interactions_by_history;
      interactions_by_history.reserve(interaction_records.size());
      for (std::size_t index = 0; index < interaction_records.size();
           ++index) {
        interactions_by_history[interaction_records[index].particle.history_id]
            .push_back(index);
      }
      auto claim_interaction = [&interaction_records, &interaction_claimed,
                                &interactions_by_history](
                                   std::uint64_t history_id,
                                   std::int32_t process_id,
                                   double final_state_fraction)
          -> gpu::em::EmInteractionRecord const* {
        auto const history = interactions_by_history.find(history_id);
        if (history == interactions_by_history.end()) { return nullptr; }
        for (auto const index : history->second) {
          if (interaction_claimed[index]) { continue; }
          auto const& candidate = interaction_records[index];
          if (candidate.process_id != process_id) { continue; }
          // The final-state scalar is the sampled loss fraction for pair,
          // Compton, bremsstrahlung, ionization and Epair.  Photoelectric
          // stores the emitted-electron energy split, while annihilation
          // stores the two-photon rho split, so those two processes cannot be
          // matched to the interaction v through this field.
          if (process_id != gpu::em::PhotonPairProcessId &&
              process_id != gpu::em::PhotoelectricProcessId &&
              process_id != gpu::em::AnnihilationProcessId) {
            auto const scale = std::max(
                {1., std::abs(candidate.energy_fraction),
                 std::abs(final_state_fraction)});
            if (std::abs(candidate.energy_fraction - final_state_fraction) >
                32. * std::numeric_limits<double>::epsilon() * scale) {
              continue;
            }
          }
          interaction_claimed[index] = true;
          return &candidate;
        }
        return nullptr;
      };
      for (auto const& record : router.photonFinalStateRecords()) {
        replay_trace.recordGpuPhotonFinalState(
            record,
            claim_interaction(
                record.parent_history_id, record.process_id,
                record.energy_split_fraction));
      }
      for (auto const& record : router.leptonFinalStateRecords()) {
        replay_trace.recordGpuLeptonFinalState(
            record,
            claim_interaction(
                record.parent_history_id, record.process_id,
                record.photon_energy_fraction));
      }

      auto const& backend_stats = backend.statistics();
      auto const& router_stats = router.statistics();
      auto const& sink_stats = output_sink.statistics();
      auto const& proposal_fallback_stats =
          fallback_handler.statistics();
      auto const& hybrid_timing =
          EAS.timingStatistics();
      auto const photo_hadronic_after =
          photoHadronicHighEnergy.statistics();
      auto const photo_hadronic_le_after =
          photoHadronicLowEnergy.statistics();
      auto const photo_hadronic_sophia =
          photo_hadronic_le_after.preferred_interactions -
          photo_hadronic_le_before.preferred_interactions;
      auto const photo_hadronic_threshold_fallback =
          photo_hadronic_le_after.fallback_interactions -
          photo_hadronic_le_before.fallback_interactions;
      auto const photo_hadronic_preferred =
          photo_hadronic_after.preferred_interactions -
          photo_hadronic_before.preferred_interactions;
      auto const photo_hadronic_fallback =
          photo_hadronic_after.fallback_interactions -
          photo_hadronic_before.fallback_interactions;
      CORSIKA_LOG_INFO(
          "CUDA EM summary: staged={}, photon steps={}, lepton steps={}, "
          "wavefronts={}, GPU final states={}, CPU fallbacks={}, "
          "CPU expansion steps={}, forced CPU decays={}/{}, "
          "specified CPU final states={}, "
          "observed={}, escaped={}, "
          "weighted deposit={} GeV, radio tracks={}, "
          "resident wavefronts(photon/lepton)={}/{}, "
          "backend wall(photon/lepton)={}/{} ms, "
          "host postprocess(photon/lepton)={}/{} ms, "
          "annihilation={}, thinning(hillas/statistical/discarded)="
          "{}/{}/{}, peak device={} MiB, kernel={} ms, transfer={} ms",
          router_stats.particles_staged, router_stats.photons_advanced,
          router_stats.leptons_advanced, router_stats.wavefronts,
          backend_stats.gpu_final_states,
          router_stats.particles_returned_for_cpu_fallback,
          router_stats.cpu_wavefront_expansion_steps_executed,
          router_stats.particles_returned_for_cpu_decay,
          router_stats.forced_cpu_decays_executed,
          router_stats.specified_cpu_final_states, router_stats.particles_observed,
          router_stats.particles_escaped, sink_stats.weighted_deposited_energy_GeV,
          sink_stats.radio_tracks,
          router_stats.resident_photon_wavefronts,
          router_stats.resident_lepton_wavefronts,
          router_stats.photon_backend_wall_time_ms,
          router_stats.lepton_backend_wall_time_ms,
          router_stats.photon_host_postprocess_time_ms,
          router_stats.lepton_host_postprocess_time_ms,
          backend_stats.annihilation_final_states,
          backend_stats.thinning_hillas_vertices,
          backend_stats.thinning_statistical_vertices,
          backend_stats.thinning_particles_discarded,
          static_cast<double>(backend_stats.peak_device_bytes) /
              (1024. * 1024.),
          backend_stats.kernel_time_ms, backend_stats.transfer_time_ms);

      auto const& low_energy_timing_samples =
          leIntCounted.getTimingSamples();
      auto const& high_energy_timing_samples =
          heCounted.getTimingSamples();
      ClassifiedHadronicWorkQueue<std::uint64_t>
          hadronic_worker_queue;
      std::map<HadronicWorkKey, HadronicWorkClassStatistics>
          hadronic_final_state_classes;
      std::uint64_t hadronic_worker_sequence = 0;
      double low_energy_final_state_time_ms = 0.;
      double high_energy_final_state_time_ms = 0.;
      std::uint64_t deferred_timing_samples = 0;
      auto stage_timing_samples =
          [&](auto const& samples, std::size_t const first,
              double& model_time_ms) {
            for (std::size_t index = first;
                 index < samples.size(); ++index) {
              auto const& sample = samples[index];
              if (sample.deferred) {
                ++deferred_timing_samples;
                continue;
              }
              model_time_ms += sample.final_state_time_ms;
              auto const key = classifyHadronicWork(
                  sample.projectile, sample.kinetic_energy,
                  hadronic_work_classifier);
              if (!key) {
                throw std::runtime_error(
                    "InteractionCounter recorded a non-hadronic "
                    "projectile in the hadronic sequence");
              }
              hadronic_final_state_classes[*key].record(
                  sample.final_state_time_ms);
              hadronic_worker_queue.enqueue(
                  *key, hadronic_worker_sequence,
                  std::max(sample.final_state_time_ms, 1.e-9),
                  hadronic_worker_sequence);
              ++hadronic_worker_sequence;
            }
          };
      stage_timing_samples(
          low_energy_timing_samples,
          low_energy_hadronic_timings_before,
          low_energy_final_state_time_ms);
      stage_timing_samples(
          high_energy_timing_samples,
          high_energy_hadronic_timings_before,
          high_energy_final_state_time_ms);
      if (hadronic_process_pool) {
        for (auto const& [key, statistics] :
             hybrid_timing
                 .hadronic_worker_final_state_classes) {
          auto& combined =
              hadronic_final_state_classes[key];
          combined.steps += statistics.steps;
          combined.total_time_ms +=
              statistics.total_time_ms;
          combined.squared_time_ms2 +=
              statistics.squared_time_ms2;
          combined.minimum_time_ms =
              std::min(
                  combined.minimum_time_ms,
                  statistics.minimum_time_ms);
          combined.maximum_time_ms =
              std::max(
                  combined.maximum_time_ms,
                  statistics.maximum_time_ms);
          low_energy_final_state_time_ms +=
              statistics.total_time_ms;
        }
      }
      auto const hadronic_worker_assignments =
          planHadronicWorkerAssignments(
              hadronic_worker_queue,
              static_cast<std::size_t>(hadronic_plan_workers),
              hadronic_plan_target_ms,
              hadronic_plan_max_batch);
      double hadronic_worker_predicted_makespan_ms = 0.;
      double hadronic_worker_minimum_load_ms =
          std::numeric_limits<double>::infinity();
      std::size_t hadronic_worker_batch_count = 0;
      for (auto const& assignment :
           hadronic_worker_assignments) {
        hadronic_worker_predicted_makespan_ms =
            std::max(
                hadronic_worker_predicted_makespan_ms,
                assignment.estimated_cost);
        hadronic_worker_minimum_load_ms =
            std::min(
                hadronic_worker_minimum_load_ms,
                assignment.estimated_cost);
        hadronic_worker_batch_count +=
            assignment.batches.size();
      }
      if (hadronic_worker_assignments.empty()) {
        hadronic_worker_minimum_load_ms = 0.;
      }
      auto const hadronic_final_state_serial_time_ms =
          low_energy_final_state_time_ms +
          high_energy_final_state_time_ms;

      YAML::Node shower_metadata;
      shower_metadata["rng_domain_version"] = RandomDomainVersion;
      shower_metadata["physics_alignment_revision"] = "2026-09-06";
      shower_metadata["hadronic_models"]["high_energy"]["name"] =
          high_energy_hadronic_model;
      shower_metadata["hadronic_models"]["high_energy"]["interactions"] =
          static_cast<std::uint64_t>(
              heCounted.getCount() -
              high_energy_hadronic_interactions_before);
      shower_metadata["hadronic_models"]["high_energy"]
                     ["final_state_time_ms"] =
          high_energy_final_state_time_ms;
#ifdef WITH_FLUKA
      shower_metadata["hadronic_models"]["low_energy"]["name"] =
          "FLUKA";
      shower_metadata["hadronic_models"]["low_energy"]["version"] =
          ::fluka::get_version();
#else
      shower_metadata["hadronic_models"]["low_energy"]["name"] =
          "UrQMD";
#endif
      shower_metadata["hadronic_models"]["low_energy"]["interactions"] =
          static_cast<std::uint64_t>(
              leIntCounted.getCount() -
              low_energy_hadronic_interactions_before);
      shower_metadata["hadronic_models"]["low_energy"]
                     ["final_state_time_ms"] =
          low_energy_final_state_time_ms;
      shower_metadata["hadronic_models"]["transition_energy_GeV"] =
          heHadronModelThreshold / 1_GeV;
      shower_metadata["thinning"]["em_fraction"] =
          emthinfrac;
      shower_metadata["thinning"]["maximum_weight"] =
          maxWeight;
      shower_metadata["thinning"]
                     ["automatic_maximum_weight"] =
          automaticMaxWeight;
      shower_metadata["thinning"]
                     ["can_activate_from_unit_weight"] =
          thinningCanActivateFromUnitWeight;
      shower_metadata["process_registry"]["registrations"] =
          GpuEmStepRegistry::registrationCount();
      shower_metadata["process_registry"]["device_replaced"] =
          GpuEmStepRegistry::replacedOnDeviceCount();
      shower_metadata["process_registry"]["record_replayed"] =
          GpuEmStepRegistry::
              replayedFromDeviceRecordCount();
      shower_metadata["process_registry"]["deferred_to_cpu"] =
          GpuEmStepRegistry::deferredToCpuCount();
      shower_metadata["process_registry"]
                     ["inapplicable_to_routed_em"] =
          GpuEmStepRegistry::inapplicableToRoutedEmCount();
      shower_metadata["process_registry"]["diagnostic_only"] =
          GpuEmStepRegistry::diagnosticOnlyCount();
      shower_metadata["process_registry"]
                     ["unregistered_continuous_processes"] =
          GpuEmStepRegistry::template
              unregisteredContinuousProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_secondaries_processes"] =
          GpuEmStepRegistry::template
              unregisteredSecondariesProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_interaction_processes"] =
          GpuEmStepRegistry::template
              unregisteredInteractionProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_decay_processes"] =
          GpuEmStepRegistry::template
              unregisteredDecayProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_boundary_processes"] =
          GpuEmStepRegistry::template
              unregisteredBoundaryProcessCount<Sequence>();
      shower_metadata["process_registry"]
                     ["unregistered_stack_processes"] =
          GpuEmStepRegistry::template
              unregisteredStackProcessCount<Sequence>();
      shower_metadata["process_registry"]["accepted"] = true;
      shower_metadata["forced_primary"]
                     ["interaction_executed"] =
          hybrid_timing.forced_primary_interactions;
      shower_metadata["forced_primary"]["decay_executed"] =
          hybrid_timing.forced_primary_decays;
      shower_metadata["gpu_particles"] =
          router_stats.photons_advanced +
          router_stats.leptons_advanced;
      shower_metadata["backend_lifecycle"]["reused"] =
          backend_reused_for_shower;
      shower_metadata["backend_lifecycle"]["shower_ordinal"] =
          backend_stats.shower_ordinal;
      shower_metadata["backend_lifecycle"]
                     ["one_time_initialization_ms"] =
          backend_stats.one_time_initialization_ms;
      shower_metadata["backend_lifecycle"]
                     ["static_host_to_device_bytes"] =
      backend_stats.static_host_to_device_bytes;
      if (gpu_cli.em_backend == "kokkos") {
        auto accelerator = shower_metadata["accelerator"];
        accelerator["backend"] = backend_stats.accelerator_backend;
        accelerator["device_name"] =
            backend_stats.accelerator_device_name;
        accelerator["architecture"] =
            backend_stats.accelerator_architecture;
        accelerator["driver_version"] =
            backend_stats.accelerator_driver_version;
        accelerator["runtime_version"] =
            backend_stats.accelerator_runtime_version;
        accelerator["compiler_version"] =
            backend_stats.accelerator_compiler_version;
        accelerator["project_revision"] =
            backend_stats.accelerator_project_revision;
        accelerator["device"] = backend_stats.accelerator_device;
        accelerator["concurrency"] =
            backend_stats.accelerator_concurrency;
        accelerator["host_threads"] =
            backend_stats.accelerator_host_threads;
        accelerator["gpu"] = backend_stats.accelerator_gpu;
        accelerator["openmp"] = backend_stats.accelerator_openmp;
        if (backend_stats.cooperative.enabled) {
          auto cooperative = accelerator["cooperative"];
          auto const& c = backend_stats.cooperative;
          cooperative["experimental"] = true;
          if (backend_stats.accelerator_backend == "openmp-cuda") {
            cooperative["scheduling_policy"] = "cpu-primary-v7-tail-checkpoint";
            cooperative["host_checkpoint_minimum_semantics"] = "max(1, min(native minimum, initial batch / 4)); return surviving states without particle cuts";
            for(auto count:c.adaptive_job_input_histogram[1])
              cooperative["host_input_batch_log2_histogram"].push_back(count);
            cooperative["host_species_coalesces"] = c.adaptive_species_coalesces[1];
            cooperative["host_species_maximum_deferrals"] = 8;
            cooperative["host_species_selection_semantics"] = "defer sub-minimum species only while the other has a useful front; at most eight host choices; drain lone tails immediately";
            cooperative["auxiliary_blocking_wait_enabled"] = c.auxiliary_blocking_wait_enabled;
            cooperative["auxiliary_blocking_wait_calls"] = c.auxiliary_blocking_wait_calls;
            cooperative["auxiliary_blocking_wait_ms"] = c.auxiliary_blocking_wait_ms;
            cooperative["auxiliary_wait_semantics"] = "reused blocking CUDA event on helper stream; default single-endpoint and GPU-primary fences unchanged; wait time overlaps primary work";
            cooperative["primary_endpoint"] = "openmp";
            cooperative["auxiliary_endpoint"] = "cuda";
            cooperative["specified_fallback_batch_limit"] = 4096;
            cooperative["specified_fallback_semantics"] = "coordinator-owned batches; flush at 4096 or empty front without joining peer; protect EM products until front drains";
            cooperative["primary_reserve_semantics"] = "one full native OpenMP arena per species; only surplus may seed auxiliary work";
            cooperative["auxiliary_grant_semantics"] = "new work >= min(CUDA minimum batch, capacity); owned tails and pressure spills still drain";
            cooperative["primary_input_semantics"] = "resident queue first, then waiting input; same fill order as standalone OpenMP";
            cooperative["auxiliary_continuation_call_limit"] = 4;
            cooperative["auxiliary_retention_semantics"] = "retention budget plus at most one ordinary bounded result; checked before the next call";
            cooperative["subshower_cuda_autonomous_continuations"] = c.subshower_cuda_autonomous_continuations;
            cooperative["subshower_cuda_foreground_packets"] = c.subshower_cuda_foreground_packets;
            cooperative["subshower_cuda_foreground_continuations"] = c.subshower_cuda_foreground_continuations;
            cooperative["cuda_completion_packets"] = c.subshower_cuda_submissions-c.subshower_cuda_autonomous_continuations;
            cooperative["cuda_completion_buffer_delay_ms"] = c.cuda_completion_buffer_delay_ms;
            cooperative["cuda_completion_mailbox_capacity"] = c.cuda_completion_mailbox_capacity;
            cooperative["cuda_completion_retention_budget_bytes"] = c.cuda_completion_retention_budget_bytes;
            cooperative["cuda_completion_peak_bytes"] = c.cuda_completion_peak_bytes;
            cooperative["maximum_cuda_packet_calls"] = c.maximum_cuda_packet_calls;
            cooperative["cuda_continuation_stop_order"] = std::vector<std::string>{
                "completed", "no_progress", "memory", "call_limit", "handoff", "time", "disabled"};
            cooperative["cuda_continuation_stops"] = c.cuda_continuation_stops;
            cooperative["auxiliary_continuation_semantics"] = "fixed photon16/lepton8 checkpoints; at most four calls per packet; no adaptive time controller; yield at safe boundary when primary needs work";
            cooperative["host_profile_shards"] = c.host_profile_shards;
            cooperative["host_profile_shard_bytes"] = c.host_profile_shard_bytes;
            cooperative["host_lepton_wave_limit"] = 1024;
            cooperative["auxiliary_lepton_wave_limit"] = 8;
            for (unsigned e=0;e<2;++e) for (unsigned k=0;k<2;++k) {
              auto channel=cooperative["priority_endpoints"][e==0?"cuda":"openmp"][k==0?"photon":"lepton"];
              channel["transport_records"] = c.adaptive_transport_records[e][k];
              channel["resident_wavefronts"] = c.adaptive_resident_wavefronts[e][k];
            }
          }
          if (c.adaptive_policy) {
            cooperative["scheduling_policy"] = "adaptive-v13-independent-service-horizon";
            cooperative["epoch_target_semantics"] = "symmetric measured work target 50--800 ms; not clamped to CUDA receipt polling; soft complete-call boundary";
            cooperative["specified_fallback_batch_limit"] = 4096;
            cooperative["specified_fallback_semantics"] = "coordinator-owned batches; flush at 4096 or empty front without joining peer; protect EM products until front drains";
            cooperative["host_profile_shards"] = c.host_profile_shards;
            cooperative["host_profile_shard_bytes"] = c.host_profile_shard_bytes;
            cooperative["host_profile_shard_semantics"] = "bounded OpenMP lepton-step integer replicas; checked final merge before decoding; canonical final-state and Moliere reduction unchanged";
            cooperative["autonomous_estimate_semantics"] = "driver-local cost update after each completed call; coordinator state updated only at commit";
            cooperative["subshower_cuda_autonomous_continuations"] = c.subshower_cuda_autonomous_continuations;
            cooperative["subshower_cuda_foreground_packets"] = c.subshower_cuda_foreground_packets;
            cooperative["subshower_cuda_foreground_continuations"] = c.subshower_cuda_foreground_continuations;
            cooperative["scalar_foreground_semantics"] = "GPU submissions while the coordinator yielded to scalar work; not measured kernel overlap";
            cooperative["cuda_completion_packets"] = c.subshower_cuda_submissions-c.subshower_cuda_autonomous_continuations;
            cooperative["cuda_completion_buffer_delay_ms"] = c.cuda_completion_buffer_delay_ms;
            cooperative["cuda_completion_mailbox_capacity"] = c.cuda_completion_mailbox_capacity;
            cooperative["cuda_completion_retention_budget_bytes"] = c.cuda_completion_retention_budget_bytes;
            cooperative["cuda_completion_peak_bytes"] = c.cuda_completion_peak_bytes;
            cooperative["maximum_cuda_packet_calls"] = c.maximum_cuda_packet_calls;
            cooperative["cuda_continuation_stop_order"] = std::vector<std::string>{
                "completed", "no_progress", "memory", "call_limit", "handoff", "time", "disabled"};
            cooperative["cuda_continuation_stops"] = c.cuda_continuation_stops;
            cooperative["cuda_result_service_delay_semantics"] = "last autonomous call end to coordinator receipt; buffer delay includes intervening GPU work";
            cooperative["maximum_host_epochs_per_cuda_job_semantics"] = "per time/memory/call-bounded driver packet, not per logical resident call";
            cooperative["adaptive_migration_ms"] = c.adaptive_migration_ms;
            cooperative["adaptive_calibration_rule"] = "inputs >= 7/8 * min(input_limit, 4096); elapsed-time-weighted";
            cooperative["adaptive_completion_reason_order"] = std::vector<std::string>{
                "completed", "minimum_batch", "workspace", "history_lease", "wave_lease", "other"};
            for (unsigned e=0;e<2;++e) for (unsigned k=0;k<2;++k) {
              auto channel=cooperative["adaptive"][e==0?"cuda":"openmp"][k==0?"photon":"lepton"];
              channel["records_per_ms"] = c.adaptive_records_per_ms[e][k];
              channel["observations"] = c.adaptive_observations[e][k];
              channel["wave_limit"] = c.adaptive_wave_limit[e][k];
              channel["transport_records"] = c.adaptive_transport_records[e][k];
              channel["resident_wavefronts"] = c.adaptive_resident_wavefronts[e][k];
              channel["input_limit"] = c.adaptive_input_limit[e][k];
              channel["full_batch_observations"] = c.adaptive_full_observations[e][k];
              channel["completion_reasons"] = c.adaptive_completion_reasons[e][k];
            }
            for(unsigned e=0;e<2;++e) {
              auto endpoint=cooperative["adaptive"][e==0?"cuda":"openmp"];
              endpoint["species_coalesces"] = c.adaptive_species_coalesces[e];
              endpoint["job_input_histogram_floor_log2"] = c.adaptive_job_input_histogram[e];
            }
          }
          cooperative["independent_drivers"] = c.independent_drivers;
          cooperative["independent_subshowers"] = c.independent_subshowers;
          cooperative["subshower_cuda_submissions"] = c.subshower_cuda_submissions;
          cooperative["subshower_cuda_commits"] = c.subshower_cuda_commits;
          cooperative["subshower_openmp_epochs"] = c.subshower_openmp_epochs;
          cooperative["maximum_host_epochs_per_cuda_job"] = c.maximum_host_epochs_per_cuda_job;
          cooperative["subshower_tail_migrations"] = c.subshower_tail_migrations;
          cooperative["subshower_proactive_refills"] = c.subshower_proactive_refills;
          cooperative["subshower_inflight_waiting_migrations"] = c.subshower_inflight_waiting_migrations;
          cooperative["subshower_inflight_waiting_particles"] = c.subshower_inflight_waiting_particles;
          cooperative["subshower_cuda_epoch_reductions"] = c.subshower_cuda_epoch_reductions;
          cooperative["subshower_gpu_lepton_wave_limit"] = c.subshower_gpu_lepton_wave_limit;
          cooperative["subshower_initial_host_share"] = c.subshower_initial_host_share;
          cooperative["subshower_maximum_cuda_epoch_ms"] = c.subshower_maximum_cuda_epoch_ms;
          cooperative["subshower_host_queue_peak_bytes"] = c.subshower_host_queue_peak_bytes;
          cooperative["subshower_requeued_spill_particles"] = c.subshower_requeued_spill_particles;
          cooperative["subshower_cross_endpoint_spill_particles"] = c.subshower_cross_endpoint_spill_particles;
          cooperative["coordinator_idle_wait_ms"] = c.coordinator_idle_wait_ms;
          cooperative["cuda_result_service_delay_ms"] = c.cuda_result_service_delay_ms;
          cooperative["independent_joint_calls"] = c.independent_joint_calls;
          cooperative["cuda_driver_wall_ms"] = c.cuda_driver_wall_ms;
          cooperative["joint_wall_ms"] = c.joint_wall_ms;
          cooperative["endpoint_window_overlap_ms"] = c.endpoint_window_overlap_ms;
          cooperative["cuda_finished_before_host_ms"] = c.cuda_finished_before_host_ms;
          cooperative["host_finished_before_cuda_ms"] = c.host_finished_before_cuda_ms;
          cooperative["cuda_input_particles"] = c.cuda_input_particles;
          cooperative["openmp_input_particles"] = c.openmp_input_particles;
          cooperative["openmp_slices"] = c.openmp_slices;
          cooperative["slices_while_cuda_pending"] = c.slices_while_cuda_pending;
          cooperative["oversized_slices_2ms"] = c.oversized_slices;
          cooperative["openmp_wall_ms"] = c.openmp_wall_ms;
          cooperative["openmp_while_cuda_pending_ms"] = c.openmp_while_cuda_pending_ms;
          cooperative["maximum_slice_ms"] = c.maximum_slice_ms;
          cooperative["migration_bytes"] = c.migration_bytes;
          cooperative["photon_calls"] = c.photon_calls;
          cooperative["lepton_calls"] = c.lepton_calls;
          cooperative["openmp_batch_target"] = c.openmp_batch_target;
          cooperative["openmp_workspace_bytes"] = c.openmp_workspace_bytes;
          cooperative["overlap_timing_semantics"] =
              "Independent host-driver call window intersection; not measured kernel-overlap time";
        }
        auto tuning = accelerator["tuning"];
        tuning["cache_matched"] = backend_stats.tuning_cache_matched;
        tuning["cache_required"] = backend_stats.tuning_cache_required;
        tuning["cache_hash"] = backend_stats.tuning_cache_hash;
        tuning["batch_size"] = static_cast<std::uint64_t>(
            backend_stats.tuning_batch_size);
        tuning["chunk_size"] = static_cast<std::uint64_t>(
            backend_stats.tuning_chunk_size);
        tuning["team_size"] = static_cast<std::uint64_t>(
            backend_stats.tuning_team_size);
        tuning["track_tile_size"] = static_cast<std::uint64_t>(
            backend_stats.tuning_track_tile_size);
        tuning["observer_tile_size"] = static_cast<std::uint64_t>(
            backend_stats.tuning_observer_tile_size);
        tuning["device_queues"] = static_cast<std::uint64_t>(
            backend_stats.tuning_device_queues);
      }
      shower_metadata["cpu_particle_steps"] =
          EAS.schedulerStatistics()
              .acquired_particle_steps -
          router_stats.particles_staged;
      shower_metadata["particles_staged"] =
          router_stats.particles_staged;
      if (gpu_cli.em_backend == "kokkos") {
        shower_metadata["routing"]["attempts"] =
            router_stats.route_attempts;
        shower_metadata["routing"]["non_em_rejections"] =
            router_stats.route_non_em_rejections;
        shower_metadata["routing"]["backend_rejections"] =
            router_stats.route_backend_rejections;
        shower_metadata["routing"]["environment_rejections"] =
            router_stats.route_environment_rejections;
      }
      shower_metadata["gpu_muon_transport_enabled"] =
          gpu_muon_transport_available;
      shower_metadata["wavefronts"] =
          router_stats.wavefronts;
      shower_metadata["resident_photon_wavefronts"] =
          router_stats.resident_photon_wavefronts;
      shower_metadata["resident_lepton_wavefronts"] =
          router_stats.resident_lepton_wavefronts;
      shower_metadata["below_minimum_batch_checkpoints"] =
          router_stats.below_minimum_batch_checkpoints;
      shower_metadata["workspace_limit_checkpoints"] =
          router_stats.workspace_limit_checkpoints;
      shower_metadata["input_batch_splits"] =
          router_stats.input_batch_splits;
      shower_metadata["maximum_input_batch"] =
          static_cast<std::uint64_t>(
              router_stats.maximum_input_batch);
      shower_metadata["gpu_final_states"] =
          backend_stats.gpu_final_states;
      shower_metadata["first_interaction_candidates"] =
          backend_stats.first_interaction_candidates;
      shower_metadata["first_interactions_written"] =
          output_sink.statistics().first_interactions;
      shower_metadata["physical_secondaries"] =
          backend_stats.physical_secondaries_generated;
      shower_metadata["cpu_generic_fallbacks"] =
          router_stats
              .particles_returned_for_cpu_fallback;
      shower_metadata["cpu_memory_spill_particles"] =
          router_stats
              .particles_returned_for_cpu_memory_spill;
      shower_metadata["cpu_decay_particles"] =
          router_stats.particles_returned_for_cpu_decay;
      shower_metadata["forced_cpu_decays_executed"] =
          router_stats.forced_cpu_decays_executed;
      shower_metadata["cpu_specified_final_states"] =
          router_stats.specified_cpu_final_states;
      shower_metadata["deferred_cpu_fallbacks_queued"] =
          router_stats.deferred_cpu_fallbacks_queued;
      shower_metadata["deferred_cpu_fallbacks_flushed"] =
          router_stats.deferred_cpu_fallbacks_flushed;
      shower_metadata["deferred_cpu_fallback_flushes"] =
          router_stats.deferred_cpu_fallback_flushes;
      shower_metadata["deferred_fallback_scalar_expansion_rounds"] =
          router_stats.deferred_fallback_scalar_expansion_rounds;
      shower_metadata["deferred_front_gpu_flushes"] =
          router_stats.deferred_front_gpu_flushes;
      shower_metadata["deferred_product_gpu_flushes"] =
          router_stats.deferred_product_gpu_flushes;
      shower_metadata["maximum_deferred_cpu_fallback_batch"] =
          static_cast<std::uint64_t>(
              router_stats.maximum_deferred_cpu_fallback_batch);
      shower_metadata["cpu_completed_selected_losses"] =
          proposal_fallback_stats
              .completed_selected_losses;
      shower_metadata["cpu_completed_native_selection_replays"] =
          proposal_fallback_stats
              .completed_native_selection_replays;
      shower_metadata["photo_hadronic_generator"]
                     ["sophia_interactions"] =
          photo_hadronic_sophia;
      shower_metadata["photo_hadronic_generator"]
                     ["threshold_fallback_interactions"] =
          photo_hadronic_threshold_fallback;
      shower_metadata["photo_hadronic_generator"]
                     ["preferred_interactions"] =
          photo_hadronic_preferred;
      shower_metadata["photo_hadronic_generator"]
                     ["fallback_interactions"] =
          photo_hadronic_fallback;
      shower_metadata["photo_hadronic_generator"]
                     ["discarded_final_states"] = 0;
      shower_metadata["photo_hadronic_generator"]
                     ["fallback_model_initialized"] =
          photoHadronicQgsjetFallback.initialized();
      shower_metadata["hybrid_timing_ms"]["total_run"] =
          hybrid_timing.total_run_time_ms;
      shower_metadata["hybrid_timing_ms"]["output_start"] =
          hybrid_timing.output_start_time_ms;
      shower_metadata["hybrid_timing_ms"]["set_nodes"] =
          hybrid_timing.set_nodes_time_ms;
      shower_metadata["hybrid_timing_ms"]["route_stage"] =
          hybrid_timing.route_stage_time_ms;
      shower_metadata["hybrid_timing_ms"]["scalar_stepper"] =
          hybrid_timing.scalar_stepper_time_ms;
      shower_metadata["hybrid_timing_ms"]["do_stack"] =
          hybrid_timing.do_stack_time_ms;
      shower_metadata["hybrid_timing_ms"]["router_advance"] =
          hybrid_timing.router_advance_time_ms;
      shower_metadata["hybrid_timing_ms"]
                     ["init_cascade_equations"] =
          hybrid_timing.init_cascade_equations_time_ms;
      shower_metadata["hybrid_timing_ms"]
                     ["cascade_equations"] =
          hybrid_timing.cascade_equations_time_ms;
      shower_metadata["hybrid_timing_ms"]["output_end"] =
          hybrid_timing.output_end_time_ms;
      shower_metadata["hadronic_process_pool"]["backend"] =
          hadronic_backend;
      shower_metadata["hadronic_process_pool"]["enabled"] =
          hybrid_timing.hadronic_process_pool_enabled;
      shower_metadata["hadronic_process_pool"]["workers"] =
          hadronic_process_pool
              ? hadronic_process_pool->workerCount()
              : 0;
      shower_metadata["hadronic_process_pool"]
                     ["minimum_pending_interactions"] =
          hadronic_min_batch;
      shower_metadata["hadronic_process_pool"]
                     ["target_batch_cost_ms"] =
          hadronic_target_batch_ms;
      shower_metadata["hadronic_process_pool"]
                     ["maximum_batch_items"] =
          hadronic_max_batch;
      shower_metadata["hadronic_process_pool"]
                     ["initial_interaction_cost_ms"] =
          hadronic_initial_cost_ms;
      shower_metadata["hadronic_process_pool"]
                     ["target_total_cost_ms"] =
          hadronic_target_batch_ms *
          static_cast<double>(
              hadronic_process_pool
                  ? hadronic_process_pool->workerCount()
                  : 0);
      shower_metadata["hadronic_process_pool"]
                     ["prepared_interactions"] =
          hybrid_timing.hadronic_interactions_prepared;
      shower_metadata["hadronic_process_pool"]
                     ["committed_interactions"] =
          hybrid_timing.hadronic_interactions_committed;
      shower_metadata["hadronic_process_pool"]["flushes"] =
          hybrid_timing.hadronic_worker_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["cost_triggered_flushes"] =
          hybrid_timing
              .hadronic_worker_cost_triggered_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["capacity_triggered_flushes"] =
          hybrid_timing
              .hadronic_worker_capacity_triggered_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["drain_triggered_flushes"] =
          hybrid_timing
              .hadronic_worker_drain_triggered_flushes;
      shower_metadata["hadronic_process_pool"]
                     ["classified_batches"] =
          hybrid_timing
              .hadronic_worker_classified_batches;
      shower_metadata["hadronic_process_pool"]["batches"] =
          hybrid_timing.hadronic_worker_batches;
      shower_metadata["hadronic_process_pool"]["secondaries"] =
          hybrid_timing.hadronic_worker_secondaries;
      shower_metadata["hadronic_process_pool"]
                     ["prepare_time_ms"] =
          hybrid_timing.hadronic_prepare_time_ms;
      shower_metadata["hadronic_process_pool"]
                     ["execute_time_ms"] =
          hybrid_timing.hadronic_worker_execute_time_ms;
      shower_metadata["hadronic_process_pool"]
                     ["commit_time_ms"] =
          hybrid_timing.hadronic_commit_time_ms;
      shower_metadata["hadronic_process_pool"]
                     ["deferred_counter_samples"] =
          deferred_timing_samples;
      YAML::Node hadronic_flush_fingerprints;
      for (auto const& fingerprint :
           hybrid_timing.hadronic_flush_fingerprints) {
        YAML::Node node;
        node["requests"] = fingerprint.requests;
        node["request_hash"] =
            std::to_string(fingerprint.request_hash);
        node["response_hash"] =
            std::to_string(fingerprint.response_hash);
        hadronic_flush_fingerprints.push_back(
            std::move(node));
      }
      shower_metadata["hadronic_process_pool"]
                     ["flush_fingerprints"] =
          std::move(hadronic_flush_fingerprints);
      YAML::Node hadronic_flush_loads;
      for (auto const& load :
           hybrid_timing.hadronic_flush_load_records) {
        YAML::Node node;
        node["trigger"] = std::string(
            hadronicQueueFlushTriggerName(
                load.trigger));
        node["requests"] = load.requests;
        YAML::Node predicted;
        YAML::Node actual;
        for (auto const value :
             load.predicted_cost_by_worker) {
          predicted.push_back(value);
        }
        for (auto const value :
             load
                 .actual_final_state_time_by_worker_ms) {
          actual.push_back(value);
        }
        node["predicted_cost_by_worker"] =
            std::move(predicted);
        node["actual_final_state_time_by_worker_ms"] =
            std::move(actual);
        hadronic_flush_loads.push_back(
            std::move(node));
      }
      shower_metadata["hadronic_process_pool"]
                     ["flush_loads"] =
          std::move(hadronic_flush_loads);
      shower_metadata["hadronic_process_pool"]
                     ["maximum_parked_projectiles"] =
          EAS.schedulerStatistics()
              .maximum_suspended_particles;
      if (hadronic_process_pool) {
        auto const pool_after =
            hadronic_process_pool->statistics();
        shower_metadata["hadronic_process_pool"]
                       ["ipc_batches"] =
            pool_after.batches -
            hadronic_pool_statistics_before.batches;
        shower_metadata["hadronic_process_pool"]
                       ["ipc_requests"] =
            pool_after.requests -
            hadronic_pool_statistics_before.requests;
        shower_metadata["hadronic_process_pool"]
                       ["ipc_secondaries"] =
            pool_after.secondaries -
            hadronic_pool_statistics_before.secondaries;
        shower_metadata["hadronic_process_pool"]
                       ["bytes_sent"] =
            pool_after.bytes_sent -
            hadronic_pool_statistics_before.bytes_sent;
        shower_metadata["hadronic_process_pool"]
                       ["bytes_received"] =
            pool_after.bytes_received -
            hadronic_pool_statistics_before.bytes_received;
        shower_metadata["hadronic_process_pool"]
                       ["dispatch_time_ms"] =
            pool_after.dispatch_time_ms -
            hadronic_pool_statistics_before
                .dispatch_time_ms;
        shower_metadata["hadronic_process_pool"]
                       ["poll_wait_time_ms"] =
            pool_after.poll_wait_time_ms -
            hadronic_pool_statistics_before
                .poll_wait_time_ms;
        shower_metadata["hadronic_process_pool"]
                       ["receive_time_ms"] =
            pool_after.receive_time_ms -
            hadronic_pool_statistics_before
                .receive_time_ms;
        shower_metadata["hadronic_process_pool"]
                       ["wire_protocol_version"] =
            HadronicBatchProtocolVersion;
        shower_metadata["hadronic_process_pool"]
                       ["bulk_response_batches"] =
            pool_after.bulk_response_batches -
            hadronic_pool_statistics_before
                .bulk_response_batches;
        shower_metadata["hadronic_process_pool"]
                       ["bulk_response_payload_reads"] =
            pool_after.bulk_response_payload_reads -
            hadronic_pool_statistics_before
                .bulk_response_payload_reads;
      }
      YAML::Node actual_hadronic_worker_loads;
      double actual_worker_minimum_ms =
          std::numeric_limits<double>::infinity();
      double actual_worker_maximum_ms = 0.;
      for (std::size_t worker = 0;
           hadronic_process_pool &&
           worker <
               hadronic_process_pool->workerCount();
           ++worker) {
        auto const load_found =
            hybrid_timing
                .hadronic_worker_final_state_time_by_worker_ms
                .find(worker);
        auto const load_ms =
            load_found ==
                    hybrid_timing
                        .hadronic_worker_final_state_time_by_worker_ms
                        .end()
                ? 0.
                : load_found->second;
        actual_worker_minimum_ms =
            std::min(actual_worker_minimum_ms, load_ms);
        actual_worker_maximum_ms =
            std::max(actual_worker_maximum_ms, load_ms);
        YAML::Node worker_node;
        worker_node["worker_id"] = worker;
        worker_node["final_state_time_ms"] =
            load_ms;
        worker_node["batches"] =
            hybrid_timing
                .hadronic_worker_batches_by_worker
                .count(worker)
                ? hybrid_timing
                      .hadronic_worker_batches_by_worker
                      .at(worker)
                : 0;
        worker_node["requests"] =
            hybrid_timing
                .hadronic_worker_requests_by_worker
                .count(worker)
                ? hybrid_timing
                      .hadronic_worker_requests_by_worker
                      .at(worker)
                : 0;
        actual_hadronic_worker_loads.push_back(
            std::move(worker_node));
      }
      if (!hadronic_process_pool) {
        actual_worker_minimum_ms = 0.;
      }
      shower_metadata["hadronic_process_pool"]
                     ["actual_worker_loads"] =
          std::move(actual_hadronic_worker_loads);
      shower_metadata["hadronic_process_pool"]
                     ["actual_generator_load_imbalance_ms"] =
          actual_worker_maximum_ms -
          actual_worker_minimum_ms;
      YAML::Node scalar_steps_by_pdg;
      for (auto const& [pdg, count] :
           hybrid_timing.scalar_steps_by_pdg) {
        scalar_steps_by_pdg[std::to_string(pdg)] =
            count;
      }
      shower_metadata["scalar_steps_by_pdg"] =
          std::move(scalar_steps_by_pdg);
      YAML::Node scalar_time_by_pdg;
      for (auto const& [pdg, time_ms] :
           hybrid_timing
               .scalar_stepper_time_by_pdg_ms) {
        scalar_time_by_pdg[std::to_string(pdg)] =
            time_ms;
      }
      shower_metadata[
          "scalar_stepper_time_by_pdg_ms"] =
          std::move(scalar_time_by_pdg);
      YAML::Node scalar_phase_timing;
      for (auto const& [pdg, phases] :
           hybrid_timing
               .scalar_phase_timing_by_pdg) {
        YAML::Node node;
        node["steps"] = phases.steps;
        node["geometry_boundary_crossings"] =
            phases.geometry_boundary_crossings;
        node["geometry_step_limits"] =
            phases.geometry_step_limits;
        node["interactions"] = phases.interactions;
        node["decays"] = phases.decays;
        node["secondary_processing_calls"] =
            phases.secondary_processing_calls;
        node["cross_section_ms"] =
            phases.cross_section_time_ms;
        node["distance_sampling_ms"] =
            phases.distance_sampling_time_ms;
        node["tracking_ms"] =
            phases.tracking_time_ms;
        node["continuous_limit_ms"] =
            phases.continuous_limit_time_ms;
        node["continuous_process_ms"] =
            phases.continuous_process_time_ms;
        node["geometry_boundary_ms"] =
            phases.geometry_boundary_time_ms;
        node["geometry_step_limit_ms"] =
            phases.geometry_step_limit_time_ms;
        node["interaction_ms"] =
            phases.interaction_time_ms;
        node["decay_ms"] = phases.decay_time_ms;
        node["secondary_processing_ms"] =
            phases.secondary_processing_time_ms;
        scalar_phase_timing[
            std::to_string(pdg)] =
            std::move(node);
      }
      shower_metadata[
          "scalar_phase_timing_by_pdg_ms"] =
          std::move(scalar_phase_timing);
      shower_metadata["hadronic_work_classifier"]
                     ["transition_energy_GeV"] =
          hybrid_timing.hadronic_work_classifier
              .transition_energy_GeV;
      shower_metadata["hadronic_work_classifier"]
                     ["energy_bins_per_octave"] =
          hybrid_timing.hadronic_work_classifier
              .energy_bins_per_octave;
      YAML::Node hadronic_work_classes;
      for (auto const& [key, statistics] :
           hybrid_timing.hadronic_work_classes) {
        YAML::Node work_class;
        work_class["model"] =
            std::string(hadronicModelClassName(key.model));
        work_class["species"] =
            std::string(hadronicSpeciesClassName(key.species));
        work_class["energy_bin"] = key.energy_bin;
        work_class["energy_lower_GeV"] =
            hadronicEnergyBinLowerGeV(
                key, hybrid_timing.hadronic_work_classifier);
        work_class["energy_upper_GeV"] =
            hadronicEnergyBinUpperGeV(
                key, hybrid_timing.hadronic_work_classifier);
        work_class["mass_number_bin"] =
            static_cast<unsigned int>(key.mass_number_bin);
        work_class["steps"] = statistics.steps;
        work_class["total_time_ms"] =
            statistics.total_time_ms;
        work_class["mean_time_ms"] =
            statistics.meanTimeMs();
        work_class["standard_deviation_ms"] =
            statistics.standardDeviationMs();
        work_class["minimum_time_ms"] =
            statistics.steps == 0
                ? 0.
                : statistics.minimum_time_ms;
        work_class["maximum_time_ms"] =
            statistics.maximum_time_ms;
        hadronic_work_classes.push_back(
            std::move(work_class));
      }
      shower_metadata["hadronic_work_classes"] =
          std::move(hadronic_work_classes);
      YAML::Node hadronic_final_state_class_nodes;
      for (auto const& [key, statistics] :
           hadronic_final_state_classes) {
        YAML::Node work_class;
        work_class["model"] =
            std::string(hadronicModelClassName(key.model));
        work_class["species"] =
            std::string(hadronicSpeciesClassName(key.species));
        work_class["energy_bin"] = key.energy_bin;
        work_class["energy_lower_GeV"] =
            hadronicEnergyBinLowerGeV(
                key, hadronic_work_classifier);
        work_class["energy_upper_GeV"] =
            hadronicEnergyBinUpperGeV(
                key, hadronic_work_classifier);
        work_class["mass_number_bin"] =
            static_cast<unsigned int>(key.mass_number_bin);
        work_class["interactions"] = statistics.steps;
        work_class["total_time_ms"] =
            statistics.total_time_ms;
        work_class["mean_time_ms"] =
            statistics.meanTimeMs();
        work_class["standard_deviation_ms"] =
            statistics.standardDeviationMs();
        work_class["minimum_time_ms"] =
            statistics.minimum_time_ms;
        work_class["maximum_time_ms"] =
            statistics.maximum_time_ms;
        hadronic_final_state_class_nodes.push_back(
            std::move(work_class));
      }
      shower_metadata["hadronic_final_state_classes"] =
          std::move(hadronic_final_state_class_nodes);
      shower_metadata["hadronic_worker_oracle"]["mode"] =
          hadronic_process_pool
              ? "disabled_actual_process_pool_active"
              : "retrospective_measured_final_state_only";
      shower_metadata["hadronic_worker_oracle"]["workers"] =
          hadronic_plan_workers;
      shower_metadata["hadronic_worker_oracle"]
                     ["target_batch_cost_ms"] =
          hadronic_plan_target_ms;
      shower_metadata["hadronic_worker_oracle"]
                     ["maximum_batch_items"] =
          hadronic_plan_max_batch;
      shower_metadata["hadronic_worker_oracle"]["interactions"] =
          hadronic_worker_sequence;
      shower_metadata["hadronic_worker_oracle"]["batches"] =
          hadronic_worker_batch_count;
      shower_metadata["hadronic_worker_oracle"]
                     ["serial_final_state_time_ms"] =
          hadronic_final_state_serial_time_ms;
      shower_metadata["hadronic_worker_oracle"]
                     ["predicted_makespan_ms"] =
          hadronic_worker_predicted_makespan_ms;
      shower_metadata["hadronic_worker_oracle"]
                     ["predicted_kernel_speedup"] =
          hadronic_worker_predicted_makespan_ms > 0.
              ? hadronic_final_state_serial_time_ms /
                    hadronic_worker_predicted_makespan_ms
              : 0.;
      shower_metadata["hadronic_worker_oracle"]
                     ["load_imbalance_ms"] =
          hadronic_worker_predicted_makespan_ms -
          hadronic_worker_minimum_load_ms;
      YAML::Node hadronic_worker_loads;
      for (auto const& assignment :
           hadronic_worker_assignments) {
        YAML::Node worker;
        worker["worker_id"] = assignment.worker_id;
        worker["estimated_load_ms"] =
            assignment.estimated_cost;
        worker["batches"] = assignment.batches.size();
        hadronic_worker_loads.push_back(std::move(worker));
      }
      shower_metadata["hadronic_worker_oracle"]["worker_loads"] =
          std::move(hadronic_worker_loads);
      shower_metadata["cpu_fallback_steps_executed"] =
          router_stats.cpu_fallback_steps_executed;
      shower_metadata["cpu_wavefront_expansion_steps_executed"] =
          router_stats
              .cpu_wavefront_expansion_steps_executed;
      shower_metadata[
          "particles_returned_for_cpu_wavefront_expansion"] =
          router_stats
              .particles_returned_for_cpu_wavefront_expansion;
      shower_metadata["small_batch_expansions"] =
          router_stats.small_batch_expansions;
      shower_metadata["stalled_boundary_gpu_flushes"] =
          router_stats.stalled_boundary_gpu_flushes;
      shower_metadata[
          "scalar_expansion_budget_gpu_flushes"] =
          router_stats
              .scalar_expansion_budget_gpu_flushes;
      shower_metadata["cpu_specified_fallback_time_ms"] =
          router_stats
              .specified_cpu_fallback_time_ms;
      shower_metadata["photon_backend_wall_time_ms"] =
          router_stats.photon_backend_wall_time_ms;
      shower_metadata["lepton_backend_wall_time_ms"] =
          router_stats.lepton_backend_wall_time_ms;
      shower_metadata["photon_host_postprocess_time_ms"] =
          router_stats.photon_host_postprocess_time_ms;
      shower_metadata["lepton_host_postprocess_time_ms"] =
          router_stats.lepton_host_postprocess_time_ms;
      YAML::Node fallbacks_by_process;
      YAML::Node fallbacks_by_process_name;
      for (auto const& [process_id, count] :
           router_stats.cpu_fallbacks_by_process) {
        fallbacks_by_process[
            std::to_string(process_id)] = count;
        fallbacks_by_process_name[
            gpuEmProcessName(process_id)] = count;
      }
      shower_metadata["cpu_fallbacks_by_process"] =
          std::move(fallbacks_by_process);
      shower_metadata["cpu_fallbacks_by_process_name"] =
          std::move(fallbacks_by_process_name);
      YAML::Node fallbacks_by_reason;
      YAML::Node fallbacks_by_reason_name;
      for (auto const& [reason, count] :
           router_stats.cpu_fallbacks_by_reason) {
        fallbacks_by_reason[
            std::to_string(reason)] = count;
        auto const typed_reason =
            static_cast<ProposalFallbackReason>(
                reason);
        fallbacks_by_reason_name[
            proposalFallbackReasonName(typed_reason)] =
            count;
      }
      shower_metadata["cpu_fallbacks_by_reason"] =
          std::move(fallbacks_by_reason);
      shower_metadata["cpu_fallbacks_by_reason_name"] =
          std::move(fallbacks_by_reason_name);
      shower_metadata["observed"] =
          router_stats.particles_observed;
      shower_metadata["escaped"] =
          router_stats.particles_escaped;
      shower_metadata["cut"] =
          router_stats.particles_cut;
      shower_metadata["weighted_deposit_GeV"] =
          sink_stats.weighted_deposited_energy_GeV;
      shower_metadata["weighted_muon_parent_productions"] =
          sink_stats.weighted_muon_parent_productions;
      shower_metadata["radio_tracks"] =
          sink_stats.radio_tracks;
      shower_metadata["radio"]["backend"] =
          gpu_radio_enabled ? accelerated_backend_name : "cpu";
      shower_metadata["radio"]["track_observer_pairs"] =
          backend_stats.radio.track_observer_pairs;
      shower_metadata["radio"]["fused_track_observer_pairs"] =
          backend_stats.radio.fused_track_observer_pairs;
      shower_metadata["radio"]["coreas_contributions"] =
          backend_stats.radio.coreas_contributions;
      shower_metadata["radio"]["zhs_contributions"] =
          backend_stats.radio.zhs_contributions;
      shower_metadata["radio"]["zhs_subtracks"] =
          backend_stats.radio.zhs_subtracks;
      shower_metadata["radio"]["deterministic"] =
          gpu_deterministic;
      shower_metadata["radio"]
                     ["fixed_point_field_limit_V_per_m"] =
          gpu_radio_field_limit;
      shower_metadata["radio"]["fixed_point_overflows"] =
          backend_stats.radio.fixed_point_overflows;
      shower_metadata["radio"]["track_diagnostics_enabled"] =
          backend_stats.radio.track_diagnostics_enabled;
      shower_metadata["radio"]["segment_count"] =
          backend_stats.radio.lepton_tracks;
      shower_metadata["radio"]["weighted_segment_count"] =
          backend_stats.radio.weighted_segment_count;
      shower_metadata["radio"]["track_length_m"] =
          backend_stats.radio.track_length_m;
      shower_metadata["radio"]["weighted_track_length_m"] =
          backend_stats.radio.weighted_track_length_m;
      shower_metadata["radio"]
                     ["electron_weighted_track_length_m"] =
          backend_stats.radio
              .electron_weighted_track_length_m;
      shower_metadata["radio"]
                     ["positron_weighted_track_length_m"] =
          backend_stats.radio
              .positron_weighted_track_length_m;
      shower_metadata["radio"]
                     ["signed_charge_weighted_track_length_m"] =
          backend_stats.radio
              .signed_charge_weighted_track_length_m;
      shower_metadata["radio"]
                     ["energy_weighted_track_length_GeV_m"] =
          backend_stats.radio
              .energy_weighted_track_length_GeV_m;
      shower_metadata["radio"]["maximum_segment_length_m"] =
          backend_stats.radio.maximum_segment_length_m;
      shower_metadata["radio"]["weighted_direction_change_rad"] =
          backend_stats.radio.weighted_direction_change_rad;
      shower_metadata["radio"]
                     ["weighted_direction_change_squared_rad2"] =
          backend_stats.radio
              .weighted_direction_change_squared_rad2;
      shower_metadata["radio"]
                     ["weighted_beta_deficit_track_length_m"] =
          backend_stats.radio
              .weighted_beta_deficit_track_length_m;
      shower_metadata["radio"]["weighted_time_residual_s"] =
          backend_stats.radio.weighted_time_residual_s;
      shower_metadata["radio"]["maximum_direction_change_rad"] =
          backend_stats.radio.maximum_direction_change_rad;
      shower_metadata["radio"]
                     ["signed_charge_weighted_direction_change"]
                     ["x"] =
          backend_stats.radio
              .signed_charge_weighted_direction_change[0];
      shower_metadata["radio"]
                     ["signed_charge_weighted_direction_change"]
                     ["y"] =
          backend_stats.radio
              .signed_charge_weighted_direction_change[1];
      shower_metadata["radio"]
                     ["signed_charge_weighted_direction_change"]
                     ["z"] =
          backend_stats.radio
              .signed_charge_weighted_direction_change[2];
      auto radio_energy_bins =
          shower_metadata["radio"]
                         ["weighted_track_length_by_kinetic_energy"];
      radio_energy_bins["units"]["upper_edge"] = "GeV";
      radio_energy_bins["units"]["weighted_track_length"] =
          "m";
      constexpr std::array<double, 14>
          radio_energy_upper_edges_GeV{
              1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2,
              5.e-2, 1.e-1, 2.e-1, 5.e-1, 1.,
              2., 5., 10., 100.};
      for (std::size_t index = 0; index < 15; ++index) {
        radio_energy_bins["upper_edge_GeV"].push_back(
            index < radio_energy_upper_edges_GeV.size()
                ? YAML::Node(
                      radio_energy_upper_edges_GeV[index])
                : YAML::Node("inf"));
        radio_energy_bins["weighted_track_length_m"]
            .push_back(
                backend_stats.radio
                    .weighted_track_length_by_kinetic_energy_m
                        [index]);
      }
      shower_metadata["radio"]["device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.radio.device_bytes);
      shower_metadata["radio"]["host_to_device_bytes"] =
          backend_stats.radio.host_to_device_bytes;
      shower_metadata["radio"]["device_to_host_bytes"] =
          backend_stats.radio.device_to_host_bytes;
      shower_metadata["radio"]["device_time_ms"] =
          backend_stats.radio.device_time_ms;
      shower_metadata["radio"]["kernel_launch_time_ms"] =
          backend_stats.radio.kernel_time_ms;
      shower_metadata["radio"]["transfer_time_ms"] =
          backend_stats.radio.transfer_time_ms;
      shower_metadata["radio"]["input_slot_waits"] =
          backend_stats.radio.input_slot_waits;
      shower_metadata["radio"]["input_slot_host_wait_time_ms"] =
          backend_stats.radio.input_slot_host_wait_time_ms;
      shower_metadata["radio"]["track_precompute_enabled"] =
          backend_stats.radio.track_precompute_enabled;
      shower_metadata["radio"]["track_tile_size"] =
          backend_stats.radio.track_tile_size;
      shower_metadata["radio"]["observer_tile_size"] =
          backend_stats.radio.observer_tile_size;
      shower_metadata["radio"]["track_precompute_batches"] =
          backend_stats.radio.track_precompute_batches;
      shower_metadata["radio"]["track_precomputed_records"] =
          backend_stats.radio.track_precomputed_records;
      shower_metadata["radio"]["direct_projection_batches"] =
          backend_stats.radio.direct_projection_batches;
      shower_metadata["radio"]["direct_projection_records"] =
          backend_stats.radio.direct_projection_records;
      shower_metadata["radio"]["projection_tiles"] =
          backend_stats.radio.projection_tiles;
      shower_metadata["radio"]["track_workspace_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.radio.track_workspace_bytes);
      shower_metadata["radio"]["maximum_track_batch"] =
          static_cast<std::uint64_t>(
              backend_stats.radio.maximum_track_batch);
      shower_metadata["radio"]
                     ["track_precompute_device_time_ms"] =
          backend_stats.radio.track_precompute_device_time_ms;
      shower_metadata["radio"]["projection_device_time_ms"] =
          backend_stats.radio.projection_device_time_ms;
      shower_metadata["profile"]["backend"] =
          backend_stats.profile.enabled ? accelerated_backend_name : "host";
      shower_metadata["profile"]["deterministic"] =
          backend_stats.profile.deterministic;
      shower_metadata["profile"]["bins"] =
          static_cast<std::uint64_t>(
              backend_stats.profile.bins);
      shower_metadata["profile"]["steps"] =
          backend_stats.profile.steps;
      shower_metadata["profile"]["deposited_steps"] =
          backend_stats.profile.deposited_steps;
      shower_metadata["profile"]["fixed_point_overflows"] =
          backend_stats.profile.fixed_point_overflows;
      shower_metadata["profile"]["invalid_records"] =
          backend_stats.profile.invalid_records;
      shower_metadata["profile"]["device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.profile.device_bytes);
      shower_metadata["profile"]["device_to_host_bytes"] =
          backend_stats.profile.device_to_host_bytes;
      shower_metadata["profile"]["kernel_time_ms"] =
          backend_stats.profile.kernel_time_ms;
      shower_metadata["profile"]["transfer_time_ms"] =
          backend_stats.profile.transfer_time_ms;
      shower_metadata["thinning"]["hillas_vertices"] =
          backend_stats.thinning_hillas_vertices;
      shower_metadata["thinning"]["statistical_vertices"] =
          backend_stats.thinning_statistical_vertices;
      shower_metadata["thinning"]["particles_discarded"] =
          backend_stats.thinning_particles_discarded;
      shower_metadata["processes"]["annihilation"] =
          backend_stats.annihilation_final_states;
      shower_metadata["processes"]["photon_pair"] =
          backend_stats.photon_pair_final_states;
      shower_metadata["processes"]["bremsstrahlung"] =
          backend_stats.brems_final_states;
      shower_metadata["processes"]["compton"] =
          backend_stats.compton_final_states;
      shower_metadata["processes"]["photoelectric"] =
          backend_stats.photoelectric_final_states;
      shower_metadata["processes"]["ionization"] =
          backend_stats.ionization_final_states;
      shower_metadata["processes"]["electron_pair"] =
          backend_stats.electron_pair_final_states;
      shower_metadata["epair_sampler"]["rejection_trials"] =
          backend_stats.electron_pair_rejection_trials;
      shower_metadata["epair_sampler"]["zero_weight_samples"] =
          backend_stats.electron_pair_zero_weight_samples;
      shower_metadata["epair_sampler"]["cpu_fallbacks"] =
          backend_stats.electron_pair_rejection_fallbacks;
      shower_metadata["epair_sampler"]["envelope_violations"] =
          backend_stats.electron_pair_envelope_violations;
      shower_metadata["moliere"]["trials"] =
          backend_stats.moliere_trials;
      shower_metadata["moliere"]["deflections"] =
          backend_stats.moliere_deflections;
      shower_metadata["moliere"]["zero_deflections"] =
          backend_stats.moliere_zero_deflections;
      shower_metadata["moliere"]["newton_iterations"] =
          backend_stats.moliere_newton_iterations;
      shower_metadata["moliere"]["maximum_newton_iterations"] =
          backend_stats.moliere_max_newton_iterations;
      shower_metadata["moliere"]["mean_newton_iterations"] =
          backend_stats.moliere_deflections == 0
              ? 0.
              : static_cast<double>(
                    backend_stats.moliere_newton_iterations) /
                    static_cast<double>(
                        backend_stats.moliere_deflections);
      shower_metadata["queue_overflows"] =
          backend_stats.queue_overflows;
      shower_metadata["peak_device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.peak_device_bytes);
      shower_metadata["table_device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.table_device_bytes);
      shower_metadata["gpu_physics_source"] =
          "proposal-native";
      if (backend_stats.physics_source ==
          GpuPhysicsSource::ProposalNative) {
        shower_metadata["proposal_native"]["proposal_version"] =
            backend_stats.native_proposal_version;
        shower_metadata["proposal_native"]
                       ["cubic_interpolation_version"] =
            backend_stats.native_cubic_interpolation_version;
        shower_metadata["proposal_native"]["table_sha256"] =
            toHex(backend_stats.native_table_hash);
        shower_metadata["proposal_native"]["node_count"] =
            backend_stats.native_table_nodes;
        shower_metadata["proposal_native"]["device_bytes"] =
            static_cast<std::uint64_t>(
                backend_stats.native_table_device_bytes);
        shower_metadata["proposal_native"]["aux_sha256"] =
            toHex(backend_stats.auxiliary_cache_hash);
        shower_metadata["proposal_native"]["aux_cache_hit"] =
            backend_stats.auxiliary_cache_hit;
        shower_metadata["proposal_native"]["proposal_cache_table_count"] =
            backend_stats.proposal_cache_table_count;
        shower_metadata["proposal_native"]["proposal_cache_hit_count"] =
            backend_stats.proposal_cache_hit_count;
        shower_metadata["proposal_native"]["proposal_cache_all_hit"] =
            backend_stats.proposal_cache_all_hit;
        shower_metadata["proposal_native"]["newton_iterations"] =
            backend_stats.native_newton_iterations;
        shower_metadata["proposal_native"]["bisection_iterations"] =
            backend_stats.native_bisection_iterations;
        shower_metadata["proposal_native"]["inverse_failures"] =
            backend_stats.native_inverse_failures;
      }
      shower_metadata["workspace_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.physical_workspace_bytes);
      shower_metadata["maximum_resident_photon_batch"] =
          static_cast<std::uint64_t>(
              backend_stats.maximum_resident_photon_batch);
      shower_metadata["maximum_resident_lepton_batch"] =
          static_cast<std::uint64_t>(
              backend_stats.maximum_resident_lepton_batch);
      if (backend_stats.automatic_capacity_budget_bytes != 0) {
        auto capacity = shower_metadata["automatic_resident_capacity"];
        capacity["policy"] = "kokkos-budgeted-arenas-v1";
        capacity["budget_bytes"] = backend_stats.automatic_capacity_budget_bytes;
        capacity["planned_retained_bytes"] = backend_stats.automatic_capacity_planned_bytes;
        capacity["planned_peak_bytes"] = backend_stats.automatic_capacity_peak_bytes;
        capacity["allocator_reserve_bytes"] =
            backend_stats.automatic_capacity_allocator_reserve_bytes;
      }
      shower_metadata["cross_species"]["queue_device_bytes"] =
          static_cast<std::uint64_t>(
              backend_stats.cross_species_queue_device_bytes);
      shower_metadata["cross_species"]["enabled"] =
          gpu_config.resident_cross_species;
      shower_metadata["cross_species"]["queue_capacity_per_pid"] =
          static_cast<std::uint64_t>(
              backend_stats.cross_species_queue_capacity_per_pid);
      shower_metadata["cross_species"]["peak_pending_photons"] =
          static_cast<std::uint64_t>(
              backend_stats.peak_pending_photons);
      shower_metadata["cross_species"]["peak_pending_leptons"] =
          static_cast<std::uint64_t>(
              backend_stats.peak_pending_leptons);
      shower_metadata["cross_species"]["particles_kept_on_device"] =
          backend_stats.cross_species_particles_kept_on_device;
      shower_metadata["cross_species"]["device_to_device_bytes"] =
          backend_stats.cross_species_device_to_device_bytes;
      shower_metadata["cross_species"]["host_spills"] =
          backend_stats.cross_species_host_spills;
      shower_metadata["cross_species"]["spill_rebalances"] =
          backend_stats.cross_species_spill_rebalances;
      shower_metadata["cross_species"]["particles_spilled_to_cpu"] =
          backend_stats.cross_species_particles_spilled_to_cpu;
      shower_metadata["cross_species"]
                     ["low_energy_ordering_checks"] =
          backend_stats.cross_species_low_energy_ordering_checks;
      shower_metadata["cross_species"]
                     ["cpu_spill_steps_executed"] =
          router_stats.cpu_memory_spill_steps_executed;
      shower_metadata["wavefront_bucketing"]["key"] =
          "PID x medium_id x logarithmic_energy_bin";
      shower_metadata["wavefront_bucketing"]
                     ["energy_bins_per_octave"] = 16;
      shower_metadata["wavefront_bucketing"]["stable"] = true;
      shower_metadata["wavefront_bucketing"]
                     ["minimum_radix_sort_size"] =
          static_cast<std::uint64_t>(256);
      shower_metadata["wavefront_bucketing"]["batches"] =
          backend_stats.wavefront_bucketing_batches;
      shower_metadata["wavefront_bucketing"]["particles"] =
          backend_stats.wavefront_bucketing_particles;
      shower_metadata["wavefront_bucketing"]["small_batches"] =
          backend_stats.wavefront_bucketing_small_batches;
      shower_metadata["wavefront_bucketing"]["small_particles"] =
          backend_stats.wavefront_bucketing_small_particles;
      shower_metadata["lepton_transport"]["interaction_candidates"] =
          backend_stats.lepton_transport_interaction_candidates;
      shower_metadata["lepton_transport"]["continuous_steps"] =
          backend_stats.lepton_transport_continuous_steps;
      shower_metadata["lepton_transport"]["particle_cuts"] =
          backend_stats.lepton_transport_cuts;
      shower_metadata["lepton_transport"]["layer_boundaries"] =
          backend_stats.lepton_transport_boundaries;
      shower_metadata["lepton_transport"]["observations"] =
          backend_stats.lepton_transport_observations;
      shower_metadata["lepton_transport"]["escapes"] =
          backend_stats.lepton_transport_escapes;
      shower_metadata["lepton_transport"]["magnetic_steps"] =
          backend_stats.lepton_transport_magnetic_steps;
      shower_metadata["lepton_transport"]["decay_candidates"] =
          backend_stats.lepton_transport_decay_candidates;
      shower_metadata["cross_species"]["final_pending_photons"] =
          static_cast<std::uint64_t>(
              backend.pendingPhotonCount());
      shower_metadata["cross_species"]["final_pending_leptons"] =
          static_cast<std::uint64_t>(
              backend.pendingLeptonCount());
      shower_metadata["host_to_device_bytes"] =
          backend_stats.physical_host_to_device_bytes;
      shower_metadata["device_to_host_bytes"] =
          backend_stats.physical_device_to_host_bytes;
      shower_metadata["pipeline_control"]
                     ["photon_selection_transport_summary_fusions"] =
          backend_stats
              .photon_selection_transport_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_selection_transport_summary_fusions"] =
          backend_stats
              .lepton_selection_transport_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["photon_transport_final_state_summary_fusions"] =
          backend_stats
              .photon_transport_final_state_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_transport_vertex_summary_fusions"] =
          backend_stats
              .lepton_transport_vertex_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_vertex_final_state_summary_fusions"] =
          backend_stats
              .lepton_vertex_final_state_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["photon_final_state_endpoint_summary_fusions"] =
          backend_stats
              .photon_final_state_endpoint_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["lepton_final_state_endpoint_summary_fusions"] =
          backend_stats
              .lepton_final_state_endpoint_summary_fusions;
      shower_metadata["pipeline_control"]
                     ["host_synchronizations_eliminated"] =
          backend_stats
              .pipeline_host_synchronizations_eliminated;
      shower_metadata["pipeline_control"]
                     ["device_to_host_bytes_eliminated"] =
          backend_stats
              .pipeline_device_to_host_bytes_eliminated;
      shower_metadata["lepton_pipeline_timing"]["enabled"] =
          backend_stats.lepton_pipeline_timing.enabled;
      shower_metadata["lepton_pipeline_timing"]["wavefronts"] =
          backend_stats.lepton_pipeline_timing.wavefronts;
      shower_metadata["lepton_pipeline_timing"]["selection_ms"] =
          backend_stats.lepton_pipeline_timing.selection_ms;
      shower_metadata["lepton_pipeline_timing"]["transport_ms"] =
          backend_stats.lepton_pipeline_timing.transport_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["transport_physics_ms"] =
          backend_stats.lepton_pipeline_timing
              .transport_physics_ms;
      shower_metadata["lepton_pipeline_timing"]["moliere_ms"] =
          backend_stats.lepton_pipeline_timing.moliere_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["transport_control_ms"] =
          backend_stats.lepton_pipeline_timing
              .transport_control_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["transport_compaction_ms"] =
          backend_stats.lepton_pipeline_timing
              .transport_compaction_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["interaction_extraction_ms"] =
          backend_stats.lepton_pipeline_timing
              .interaction_extraction_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["vertex_selection_ms"] =
          backend_stats.lepton_pipeline_timing
              .vertex_selection_ms;
      shower_metadata["lepton_pipeline_timing"]["final_state_ms"] =
          backend_stats.lepton_pipeline_timing.final_state_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_classification_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_classification_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_scan_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_scan_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_summary_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_summary_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["final_state_write_ms"] =
          backend_stats.lepton_pipeline_timing
              .final_state_write_ms;
      shower_metadata["lepton_pipeline_timing"]
                     ["endpoint_compaction_ms"] =
          backend_stats.lepton_pipeline_timing
              .endpoint_compaction_ms;
      shower_metadata["lepton_pipeline_timing"]["post_endpoint_ms"] =
          backend_stats.lepton_pipeline_timing.post_endpoint_ms;
      shower_metadata["kernel_time_ms"] =
          backend_stats.kernel_time_ms;
      shower_metadata["transfer_time_ms"] =
          backend_stats.transfer_time_ms;
      shower_metadata["timing_schema_version"] = 2;
      shower_metadata["timing_semantics"]["kernel_time_ms"] =
          "sum of CUDA-event durations across streams; overlapping streams "
          "must not be added to shower wall time";
      shower_metadata["timing_semantics"]["transfer_time_ms"] =
          "legacy host API wall time around selected synchronous CUDA copies";
      shower_metadata["transfer_timing"]
                     ["device_event_timing_enabled"] =
          backend_stats.transfer_timing.device_event_timing_enabled;
      shower_metadata["transfer_timing"]["operations"] =
          backend_stats.transfer_timing.operations;
      shower_metadata["transfer_timing"]
                     ["host_to_device_operations"] =
          backend_stats.transfer_timing.host_to_device_operations;
      shower_metadata["transfer_timing"]
                     ["device_to_host_operations"] =
          backend_stats.transfer_timing.device_to_host_operations;
      shower_metadata["transfer_timing"]
                     ["device_to_device_operations"] =
          backend_stats.transfer_timing.device_to_device_operations;
      shower_metadata["transfer_timing"]["host_api_time_ms"] =
          backend_stats.transfer_timing.host_api_time_ms;
      shower_metadata["transfer_timing"]["device_copy_time_ms"] =
          backend_stats.transfer_timing.device_copy_time_ms;
      shower_metadata["transfer_timing"]["host_wait_upper_bound_ms"] =
          backend_stats.transfer_timing.host_wait_upper_bound_ms;
      shower_metadata["synchronization_timing"]
                     ["physical_pipeline_waits"] =
          backend_stats.synchronization_timing.physical_pipeline_waits;
      shower_metadata["synchronization_timing"]
                     ["physical_pipeline_wait_time_ms"] =
          backend_stats.synchronization_timing
              .physical_pipeline_wait_time_ms;
      shower_metadata["synchronization_timing"]
                     ["profile_input_waits"] =
          backend_stats.synchronization_timing.profile_input_waits;
      shower_metadata["synchronization_timing"]
                     ["profile_input_wait_time_ms"] =
          backend_stats.synchronization_timing.profile_input_wait_time_ms;
      auto scalar_em_steps = std::uint64_t{0};
      for (auto const& [pdg, count] :
           hybrid_timing.scalar_steps_by_pdg) {
        if (pdg == 22 || pdg == 11 || pdg == -11) {
          scalar_em_steps += count;
        }
      }
      auto const ledger_initial_GeV =
          primaryTotalEnergy / 1_GeV;
      auto const ledger_source_GeV =
          ledger_initial_GeV +
          router_stats.medium_rest_mass_input_GeV +
          router_stats.mass_convention_correction_GeV;
      auto const ledger_terminal_GeV =
          router_stats.deposited_energy_GeV +
          router_stats.cut_rest_mass_energy_GeV +
          router_stats.observed_total_energy_GeV +
          router_stats.escaped_total_energy_GeV +
          router_stats.unwritten_photoelectric_binding_energy_GeV -
          router_stats.observation_cut_overlap_energy_GeV;
      auto const ledger_residual_GeV =
          ledger_source_GeV - ledger_terminal_GeV;
      auto const ledger_relative_error =
          std::abs(ledger_residual_GeV) /
          std::max(
              ledger_source_GeV,
              std::numeric_limits<double>::min());
      auto const ledger_complete_coverage =
          is_em(beamCode) && emthinfrac <= 0. &&
          scalar_em_steps == 0 &&
          router_stats
                  .particles_returned_for_cpu_fallback ==
              0 &&
          router_stats
                  .particles_returned_for_cpu_memory_spill ==
              0 &&
          router_stats.specified_cpu_final_states == 0;
      auto ledger =
          shower_metadata["energy_ledger"];
      ledger["definition"] =
          "initial_total + medium_electron_rest + mass_convention_correction = "
          "deposit + cut_rest + "
          "observed_total + escaped_total + unwritten_photoelectric_binding - "
          "observation_cut_overlap";
      ledger["complete_coverage"] =
          ledger_complete_coverage;
      ledger["scalar_em_steps"] = scalar_em_steps;
      ledger["initial_total_GeV"] =
          ledger_initial_GeV;
      ledger["medium_rest_mass_input_GeV"] =
          router_stats.medium_rest_mass_input_GeV;
      ledger["deposited_GeV"] =
          router_stats.deposited_energy_GeV;
      ledger["cut_rest_mass_energy_GeV"] =
          router_stats.cut_rest_mass_energy_GeV;
      ledger["observed_total_energy_GeV"] =
          router_stats.observed_total_energy_GeV;
      ledger["escaped_total_energy_GeV"] =
          router_stats.escaped_total_energy_GeV;
      ledger["unwritten_photoelectric_binding_energy_GeV"] =
          router_stats.unwritten_photoelectric_binding_energy_GeV;
      ledger["observation_cut_overlap_energy_GeV"] =
          router_stats.observation_cut_overlap_energy_GeV;
      ledger["mass_convention_correction_GeV"] =
          router_stats.mass_convention_correction_GeV;
      ledger["source_GeV"] = ledger_source_GeV;
      ledger["terminal_GeV"] =
          ledger_terminal_GeV;
      ledger["residual_GeV"] =
          ledger_residual_GeV;
      ledger["relative_closure_error"] =
          ledger_relative_error;
      ledger["acceptance_tolerance"] = 1.e-4;
      ledger["accepted"] =
          ledger_complete_coverage &&
          ledger_relative_error <= 1.e-4;
      CORSIKA_LOG_INFO(
          "CUDA EM energy ledger: complete={}, source={} GeV, "
          "terminal={} GeV, residual={} GeV, relative error={}",
          ledger_complete_coverage, ledger_source_GeV,
          ledger_terminal_GeV, ledger_residual_GeV,
          ledger_relative_error);
      if (ledger_complete_coverage &&
          ledger_relative_error > 1.e-4) {
        throw std::runtime_error(
            "CUDA EM strict energy closure exceeded 1e-4");
      }
      if (cuda_session.runOutput()) {
        cuda_session.runOutput()->recordComplete(
            output_shower_id,
            std::move(shower_metadata));
      }

    }
  };

} // namespace corsika::applications::air_shower
