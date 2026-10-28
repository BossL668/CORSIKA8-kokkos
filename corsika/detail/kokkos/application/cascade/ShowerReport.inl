/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>
#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <corsika/kokkos/em/random/RandomDomains.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>
#include <corsika/output/ReplayTrace.hpp>

#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
#include <corsika/kokkos/scheduling/StaticFrontier.hpp>
#endif
#ifdef WITH_FLUKA
#include <corsika/modules/FLUKA.hpp>
#endif

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
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

namespace corsika::kokkos::application {
  template <typename TStepRegistry, typename TSequence, typename TRunSession,
            typename TBackend, typename TEas, typename TRouter, typename TOutputSink,
            typename TFallbackHandler, typename TPhotoHighEnergy,
            typename TPhotoLowEnergy, typename TPhotoHighStatistics,
            typename TPhotoLowStatistics, typename THighEnergyCounter,
            typename TLowEnergyCounter, typename TPhotoFallback>
  inline void ShowerReportBuilder::record(
      TRunSession& cudaSession, BackendOptions const& gpuCli, TBackend& backend,
      TEas& EAS, TRouter& router, TOutputSink& outputSink,
      TFallbackHandler& fallbackHandler, TPhotoHighEnergy& photoHadronicHighEnergy,
      TPhotoLowEnergy& photoHadronicLowEnergy,
      TPhotoHighStatistics const& photoHadronicBefore,
      TPhotoLowStatistics const& photoHadronicLeBefore, THighEnergyCounter& heCounted,
      TLowEnergyCounter& leIntCounted, std::uint64_t highEnergyHadronicInteractionsBefore,
      std::uint64_t lowEnergyHadronicInteractionsBefore,
      std::size_t highEnergyHadronicTimingsBefore,
      std::size_t lowEnergyHadronicTimingsBefore,
      HadronicWorkClassifierConfig const& hadronicWorkClassifier,
      HadronicProcessPool* hadronicProcessPool,
      HadronicProcessPoolStatistics const& hadronicPoolStatisticsBefore,
      TPhotoFallback const& photoHadronicQgsjetFallback,
      std::string const& highEnergyHadronicModel, HEPEnergyType heHadronModelThreshold,
      double emthinfrac, double maxWeight, bool automaticMaxWeight,
      bool thinningCanActivateFromUnitWeight, bool backendReusedForShower,
      bool gpuMuonTransportAvailable, bool gpuRadioEnabled,
      corsika::kokkos::BackendConfig const& gpuConfig, Code beamCode,
      HEPEnergyType primaryTotalEnergy, unsigned int outputShowerId) {
    using step_registry_type = TStepRegistry;
    using sequence_type = TSequence;
    using namespace corsika::kokkos;

    auto const hadronicPlanWorkers = gpuCli.hadronicPlanWorkers_;
    auto const hadronicPlanTargetMs = gpuCli.hadronicPlanTargetMs_;
    auto const hadronicPlanMaxBatch = gpuCli.hadronicPlanMaxBatch_;
    auto const& hadronicBackend = gpuCli.hadronicBackend_;
    auto const hadronicMinBatch = gpuCli.hadronicMinBatch_;
    auto const hadronicTargetBatchMs = gpuCli.hadronicTargetBatchMs_;
    auto const hadronicMaxBatch = gpuCli.hadronicMaxBatch_;
    auto const hadronicInitialCostMs = gpuCli.hadronicInitialCostMs_;
    auto const gpuDeterministic = gpuCli.gpuDeterministic_;
    auto const gpuRadioFieldLimit = gpuCli.gpuRadioFieldLimit_;
    auto const acceleratedBackendName =
        gpuCli.emBackend_ == "kokkos-proposal" ? "kokkos" : "cuda";

    for (auto const& record : router.stepRecords()) {
      corsika::ReplayTrace::instance().recordGpuStep(
          record.historyId_, record.stepId_, record.pid_, record.processId_,
          record.startEnergyGeV_, record.endEnergyGeV_, record.depositedEnergyGeV_,
          record.weight_, record.startPositionM_[0], record.startPositionM_[1],
          record.startPositionM_[2], record.startTimeS_);
    }

    auto& replayTrace = corsika::ReplayTrace::instance();
    auto const& interactionRecords = router.interactionRecords();
    for (auto const& record : interactionRecords) {
      replayTrace.recordGpuInteraction(record);
    }
    std::vector<bool> interactionClaimed(interactionRecords.size(), false);
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> interactionsByHistory;
    interactionsByHistory.reserve(interactionRecords.size());
    for (std::size_t index = 0; index < interactionRecords.size(); ++index) {
      interactionsByHistory[interactionRecords[index].particle_.historyId_].push_back(
          index);
    }
    auto claimInteraction =
        [&interactionRecords, &interactionClaimed, &interactionsByHistory](
            std::uint64_t historyId, std::int32_t processId,
            double finalStateFraction) -> corsika::kokkos::InteractionRecord const* {
      auto const history = interactionsByHistory.find(historyId);
      if (history == interactionsByHistory.end()) { return nullptr; }
      for (auto const index : history->second) {
        if (interactionClaimed[index]) { continue; }
        auto const& candidate = interactionRecords[index];
        if (candidate.processId_ != processId) { continue; }
        // The final-state scalar is the sampled loss fraction for pair,
        // Compton, bremsstrahlung, ionization and Epair.  Photoelectric
        // stores the emitted-electron energy split, while annihilation
        // stores the two-photon rho split, so those two processes cannot be
        // matched to the interaction v through this field.
        if (processId != corsika::kokkos::PhotonPairProcessId &&
            processId != corsika::kokkos::PhotoelectricProcessId &&
            processId != corsika::kokkos::AnnihilationProcessId) {
          auto const scale = std::max(
              {1., std::abs(candidate.energyFraction_), std::abs(finalStateFraction)});
          if (std::abs(candidate.energyFraction_ - finalStateFraction) >
              32. * std::numeric_limits<double>::epsilon() * scale) {
            continue;
          }
        }
        interactionClaimed[index] = true;
        return &candidate;
      }
      return nullptr;
    };
    for (auto const& record : router.photonFinalStateRecords()) {
      replayTrace.recordGpuPhotonFinalState(
          record, claimInteraction(record.parentHistoryId_, record.processId_,
                                   record.energySplitFraction_));
    }
    for (auto const& record : router.leptonFinalStateRecords()) {
      replayTrace.recordGpuLeptonFinalState(
          record, claimInteraction(record.parentHistoryId_, record.processId_,
                                   record.photonEnergyFraction_));
    }

    auto const& backendStats = backend.statistics();
    auto const& routerStats = router.statistics();
    auto const& sinkStats = outputSink.statistics();
    auto const& proposalFallbackStats = fallbackHandler.statistics();
    auto const& hybridTiming = EAS.timingStatistics();
    auto const photoHadronicAfter = photoHadronicHighEnergy.statistics();
    auto const photoHadronicLeAfter = photoHadronicLowEnergy.statistics();
    auto const photoHadronicSophia = photoHadronicLeAfter.preferredInteractions_ -
                                     photoHadronicLeBefore.preferredInteractions_;
    auto const photoHadronicThresholdFallback =
        photoHadronicLeAfter.fallbackInteractions_ -
        photoHadronicLeBefore.fallbackInteractions_;
    auto const photoHadronicPreferred = photoHadronicAfter.preferredInteractions_ -
                                        photoHadronicBefore.preferredInteractions_;
    auto const photoHadronicFallback = photoHadronicAfter.fallbackInteractions_ -
                                       photoHadronicBefore.fallbackInteractions_;
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
        routerStats.particlesStaged_, routerStats.photonsAdvanced_,
        routerStats.leptonsAdvanced_, routerStats.wavefronts_,
        backendStats.gpuFinalStates_, routerStats.particlesReturnedForCpuFallback_,
        routerStats.cpuWavefrontExpansionStepsExecuted_,
        routerStats.particlesReturnedForCpuDecay_, routerStats.forcedCpuDecaysExecuted_,
        routerStats.specifiedCpuFinalStates_, routerStats.particlesObserved_,
        routerStats.particlesEscaped_, sinkStats.weightedDepositedEnergyGeV_,
        sinkStats.radioTracks_, routerStats.residentPhotonWavefronts_,
        routerStats.residentLeptonWavefronts_, routerStats.photonBackendWallTimeMs_,
        routerStats.leptonBackendWallTimeMs_, routerStats.photonHostPostprocessTimeMs_,
        routerStats.leptonHostPostprocessTimeMs_, backendStats.annihilationFinalStates_,
        backendStats.thinningHillasVertices_, backendStats.thinningStatisticalVertices_,
        backendStats.thinningParticlesDiscarded_,
        static_cast<double>(backendStats.peakDeviceBytes_) / (1024. * 1024.),
        backendStats.kernelTimeMs_, backendStats.transferTimeMs_);

    auto const& lowEnergyTimingSamples = leIntCounted.getTimingSamples();
    auto const& highEnergyTimingSamples = heCounted.getTimingSamples();
    ClassifiedHadronicWorkQueue<std::uint64_t> hadronicWorkerQueue;
    std::map<HadronicWorkKey, HadronicWorkClassStatistics> hadronicFinalStateClasses;
    std::uint64_t hadronicWorkerSequence = 0;
    double lowEnergyFinalStateTimeMs = 0.;
    double highEnergyFinalStateTimeMs = 0.;
    std::uint64_t deferredTimingSamples = 0;
    auto stageTimingSamples = [&](auto const& samples, std::size_t const first,
                                  double& modelTimeMs) {
      for (std::size_t index = first; index < samples.size(); ++index) {
        auto const& sample = samples[index];
        if (sample.deferred_) {
          ++deferredTimingSamples;
          continue;
        }
        modelTimeMs += sample.finalStateTimeMs_;
        auto const key = classify_hadronic_work(sample.projectile_, sample.kineticEnergy_,
                                                hadronicWorkClassifier);
        if (!key) {
          throw std::runtime_error(
              "InteractionCounter recorded a non-hadronic "
              "projectile in the hadronic sequence");
        }
        hadronicFinalStateClasses[*key].record(sample.finalStateTimeMs_);
        hadronicWorkerQueue.enqueue(*key, hadronicWorkerSequence,
                                    std::max(sample.finalStateTimeMs_, 1.e-9),
                                    hadronicWorkerSequence);
        ++hadronicWorkerSequence;
      }
    };
    stageTimingSamples(lowEnergyTimingSamples, lowEnergyHadronicTimingsBefore,
                       lowEnergyFinalStateTimeMs);
    stageTimingSamples(highEnergyTimingSamples, highEnergyHadronicTimingsBefore,
                       highEnergyFinalStateTimeMs);
    if (hadronicProcessPool) {
      for (auto const& [key, statistics] :
           hybridTiming.hadronicWorkerFinalStateClasses_) {
        auto& combined = hadronicFinalStateClasses[key];
        combined.steps_ += statistics.steps_;
        combined.totalTimeMs_ += statistics.totalTimeMs_;
        combined.squaredTimeMs2_ += statistics.squaredTimeMs2_;
        combined.minimumTimeMs_ =
            std::min(combined.minimumTimeMs_, statistics.minimumTimeMs_);
        combined.maximumTimeMs_ =
            std::max(combined.maximumTimeMs_, statistics.maximumTimeMs_);
        lowEnergyFinalStateTimeMs += statistics.totalTimeMs_;
      }
    }
    auto const hadronicWorkerAssignments = plan_hadronic_worker_assignments(
        hadronicWorkerQueue, static_cast<std::size_t>(hadronicPlanWorkers),
        hadronicPlanTargetMs, hadronicPlanMaxBatch);
    double hadronicWorkerPredictedMakespanMs = 0.;
    double hadronicWorkerMinimumLoadMs = std::numeric_limits<double>::infinity();
    std::size_t hadronicWorkerBatchCount = 0;
    for (auto const& assignment : hadronicWorkerAssignments) {
      hadronicWorkerPredictedMakespanMs =
          std::max(hadronicWorkerPredictedMakespanMs, assignment.estimatedCost_);
      hadronicWorkerMinimumLoadMs =
          std::min(hadronicWorkerMinimumLoadMs, assignment.estimatedCost_);
      hadronicWorkerBatchCount += assignment.batches_.size();
    }
    if (hadronicWorkerAssignments.empty()) { hadronicWorkerMinimumLoadMs = 0.; }
    auto const hadronicFinalStateSerialTimeMs =
        lowEnergyFinalStateTimeMs + highEnergyFinalStateTimeMs;

    YAML::Node showerMetadata;
    showerMetadata["rng_domain_version"] = RandomDomainVersion;
    showerMetadata["physics_alignment_revision"] = "2026-09-06";
    showerMetadata["hadronic_models"]["high_energy"]["name"] = highEnergyHadronicModel;
    showerMetadata["hadronic_models"]["high_energy"]["interactions"] =
        static_cast<std::uint64_t>(heCounted.getCount() -
                                   highEnergyHadronicInteractionsBefore);
    showerMetadata["hadronic_models"]["high_energy"]["final_state_time_ms"] =
        highEnergyFinalStateTimeMs;
#ifdef WITH_FLUKA
    showerMetadata["hadronic_models"]["low_energy"]["name"] = "FLUKA";
    showerMetadata["hadronic_models"]["low_energy"]["version"] = ::fluka::get_version();
#else
    showerMetadata["hadronic_models"]["low_energy"]["name"] = "UrQMD";
#endif
    showerMetadata["hadronic_models"]["low_energy"]["interactions"] =
        static_cast<std::uint64_t>(leIntCounted.getCount() -
                                   lowEnergyHadronicInteractionsBefore);
    showerMetadata["hadronic_models"]["low_energy"]["final_state_time_ms"] =
        lowEnergyFinalStateTimeMs;
    showerMetadata["hadronic_models"]["transition_energy_GeV"] =
        heHadronModelThreshold / 1_GeV;
    showerMetadata["thinning"]["em_fraction"] = emthinfrac;
    showerMetadata["thinning"]["maximum_weight"] = maxWeight;
    showerMetadata["thinning"]["automatic_maximum_weight"] = automaticMaxWeight;
    showerMetadata["thinning"]["can_activate_from_unit_weight"] =
        thinningCanActivateFromUnitWeight;
    showerMetadata["process_registry"]["registrations"] =
        step_registry_type::registrationCount();
    showerMetadata["process_registry"]["device_replaced"] =
        step_registry_type::replacedOnDeviceCount();
    showerMetadata["process_registry"]["record_replayed"] =
        step_registry_type::replayedFromDeviceRecordCount();
    showerMetadata["process_registry"]["deferred_to_cpu"] =
        step_registry_type::deferredToCpuCount();
    showerMetadata["process_registry"]["inapplicable_to_routed_em"] =
        step_registry_type::inapplicableToRoutedEmCount();
    showerMetadata["process_registry"]["diagnostic_only"] =
        step_registry_type::diagnosticOnlyCount();
    showerMetadata["process_registry"]["unregistered_continuous_processes"] =
        step_registry_type::template unregisteredContinuousProcessCount<sequence_type>();
    showerMetadata["process_registry"]["unregistered_secondaries_processes"] =
        step_registry_type::template unregisteredSecondariesProcessCount<sequence_type>();
    showerMetadata["process_registry"]["unregistered_interaction_processes"] =
        step_registry_type::template unregisteredInteractionProcessCount<sequence_type>();
    showerMetadata["process_registry"]["unregistered_decay_processes"] =
        step_registry_type::template unregisteredDecayProcessCount<sequence_type>();
    showerMetadata["process_registry"]["unregistered_boundary_processes"] =
        step_registry_type::template unregisteredBoundaryProcessCount<sequence_type>();
    showerMetadata["process_registry"]["unregistered_stack_processes"] =
        step_registry_type::template unregisteredStackProcessCount<sequence_type>();
    showerMetadata["process_registry"]["accepted"] = true;
    showerMetadata["forced_primary"]["interaction_executed"] =
        hybridTiming.forcedPrimaryInteractions_;
    showerMetadata["forced_primary"]["decay_executed"] =
        hybridTiming.forcedPrimaryDecays_;
    showerMetadata["gpu_particles"] =
        routerStats.photonsAdvanced_ + routerStats.leptonsAdvanced_;
    showerMetadata["backend_lifecycle"]["reused"] = backendReusedForShower;
    showerMetadata["backend_lifecycle"]["shower_ordinal"] = backendStats.showerOrdinal_;
    showerMetadata["backend_lifecycle"]["one_time_initialization_ms"] =
        backendStats.oneTimeInitializationMs_;
    showerMetadata["backend_lifecycle"]["static_host_to_device_bytes"] =
        backendStats.staticHostToDeviceBytes_;
    if (gpuCli.emBackend_ == "kokkos-proposal") {
      auto accelerator = showerMetadata["accelerator"];
      accelerator["backend"] = backendStats.acceleratorBackend_;
      accelerator["device_name"] = backendStats.acceleratorDeviceName_;
      accelerator["architecture"] = backendStats.acceleratorArchitecture_;
      accelerator["driver_version"] = backendStats.acceleratorDriverVersion_;
      accelerator["runtime_version"] = backendStats.acceleratorRuntimeVersion_;
      accelerator["compiler_version"] = backendStats.acceleratorCompilerVersion_;
      accelerator["project_revision"] = backendStats.acceleratorProjectRevision_;
      accelerator["device"] = backendStats.acceleratorDevice_;
      accelerator["concurrency"] = backendStats.acceleratorConcurrency_;
      accelerator["host_threads"] = backendStats.acceleratorHostThreads_;
      accelerator["gpu"] = backendStats.acceleratorGpu_;
      accelerator["openmp"] = backendStats.acceleratorOpenmp_;
      if (backendStats.cooperative_.enabled_) {
        auto cooperative = accelerator["cooperative"];
        auto const& c = backendStats.cooperative_;
        cooperative["experimental"] = true;
        if (backendStats.acceleratorBackend_ == "openmp-cuda") {
          cooperative["scheduling_policy"] = "cpu-primary-v7-tail-checkpoint";
          cooperative["host_checkpoint_minimum_semantics"] =
              "max(1, min(native minimum, initial batch / 4)); return surviving states "
              "without particle cuts";
          for (auto count : c.adaptiveJobInputHistogram_[1])
            cooperative["host_input_batch_log2_histogram"].push_back(count);
          cooperative["host_species_coalesces"] = c.adaptiveSpeciesCoalesces_[1];
          cooperative["host_species_maximum_deferrals"] = 8;
          cooperative["host_species_selection_semantics"] =
              "defer sub-minimum species only while the other has a useful front; at "
              "most eight host choices; drain lone tails immediately";
          cooperative["auxiliary_blocking_wait_enabled"] =
              c.auxiliaryBlockingWaitEnabled_;
          cooperative["auxiliary_blocking_wait_calls"] = c.auxiliaryBlockingWaitCalls_;
          cooperative["auxiliary_blocking_wait_ms"] = c.auxiliaryBlockingWaitMs_;
          cooperative["auxiliary_wait_semantics"] =
              "reused blocking CUDA event on helper stream; default single-endpoint and "
              "GPU-primary fences unchanged; wait time overlaps primary work";
          cooperative["primary_endpoint"] = "openmp";
          cooperative["auxiliary_endpoint"] = "cuda";
          cooperative["specified_fallback_batch_limit"] = 4096;
          cooperative["specified_fallback_semantics"] =
              "coordinator-owned batches; flush at 4096 or empty front without joining "
              "peer; protect EM products until front drains";
          cooperative["primary_reserve_semantics"] =
              "one full native OpenMP arena per species; only surplus may seed auxiliary "
              "work";
          cooperative["auxiliary_grant_semantics"] =
              "new work >= min(CUDA minimum batch, capacity); owned tails and pressure "
              "spills still drain";
          cooperative["primary_input_semantics"] =
              "resident queue first, then waiting input; same fill order as standalone "
              "OpenMP";
          cooperative["auxiliary_continuation_call_limit"] = 4;
          cooperative["auxiliary_retention_semantics"] =
              "retention budget plus at most one ordinary bounded result; checked before "
              "the next call";
          cooperative["subshower_cuda_autonomous_continuations"] =
              c.subshowerCudaAutonomousContinuations_;
          cooperative["subshower_cuda_foreground_packets"] =
              c.subshowerCudaForegroundPackets_;
          cooperative["subshower_cuda_foreground_continuations"] =
              c.subshowerCudaForegroundContinuations_;
          cooperative["cuda_completion_packets"] =
              c.subshowerCudaSubmissions_ - c.subshowerCudaAutonomousContinuations_;
          cooperative["cuda_completion_buffer_delay_ms"] = c.cudaCompletionBufferDelayMs_;
          cooperative["cuda_completion_mailbox_capacity"] =
              c.cudaCompletionMailboxCapacity_;
          cooperative["cuda_completion_retention_budget_bytes"] =
              c.cudaCompletionRetentionBudgetBytes_;
          cooperative["cuda_completion_peak_bytes"] = c.cudaCompletionPeakBytes_;
          cooperative["maximum_cuda_packet_calls"] = c.maximumCudaPacketCalls_;
          cooperative["cuda_continuation_stop_order"] = std::vector<std::string>{
              "completed", "no_progress", "memory",  "call_limit",
              "handoff",   "time",        "disabled"};
          cooperative["cuda_continuation_stops"] = c.cudaContinuationStops_;
          cooperative["auxiliary_continuation_semantics"] =
              "fixed photon16/lepton8 checkpoints; at most four calls per packet; no "
              "adaptive time controller; yield at safe boundary when primary needs work";
          cooperative["host_profile_shards"] = c.hostProfileShards_;
          cooperative["host_profile_shard_bytes"] = c.hostProfileShardBytes_;
          cooperative["host_lepton_wave_limit"] = 1024;
          cooperative["auxiliary_lepton_wave_limit"] = 8;
          for (unsigned e = 0; e < 2; ++e)
            for (unsigned k = 0; k < 2; ++k) {
              auto channel = cooperative["priority_endpoints"][e == 0 ? "cuda" : "openmp"]
                                        [k == 0 ? "photon" : "lepton"];
              channel["transport_records"] = c.adaptiveTransportRecords_[e][k];
              channel["resident_wavefronts"] = c.adaptiveResidentWavefronts_[e][k];
            }
        }
        if (c.adaptivePolicy_) {
          cooperative["scheduling_policy"] = "adaptive-v13-independent-service-horizon";
          cooperative["epoch_target_semantics"] =
              "symmetric measured work target 50--800 ms; not clamped to CUDA receipt "
              "polling; soft complete-call boundary";
          cooperative["specified_fallback_batch_limit"] = 4096;
          cooperative["specified_fallback_semantics"] =
              "coordinator-owned batches; flush at 4096 or empty front without joining "
              "peer; protect EM products until front drains";
          cooperative["host_profile_shards"] = c.hostProfileShards_;
          cooperative["host_profile_shard_bytes"] = c.hostProfileShardBytes_;
          cooperative["host_profile_shard_semantics"] =
              "bounded OpenMP lepton-step integer replicas; checked final merge before "
              "decoding; canonical final-state and Moliere reduction unchanged";
          cooperative["autonomous_estimate_semantics"] =
              "driver-local cost update after each completed call; coordinator state "
              "updated only at commit";
          cooperative["subshower_cuda_autonomous_continuations"] =
              c.subshowerCudaAutonomousContinuations_;
          cooperative["subshower_cuda_foreground_packets"] =
              c.subshowerCudaForegroundPackets_;
          cooperative["subshower_cuda_foreground_continuations"] =
              c.subshowerCudaForegroundContinuations_;
          cooperative["scalar_foreground_semantics"] =
              "GPU submissions while the coordinator yielded to scalar work; not "
              "measured kernel overlap";
          cooperative["cuda_completion_packets"] =
              c.subshowerCudaSubmissions_ - c.subshowerCudaAutonomousContinuations_;
          cooperative["cuda_completion_buffer_delay_ms"] = c.cudaCompletionBufferDelayMs_;
          cooperative["cuda_completion_mailbox_capacity"] =
              c.cudaCompletionMailboxCapacity_;
          cooperative["cuda_completion_retention_budget_bytes"] =
              c.cudaCompletionRetentionBudgetBytes_;
          cooperative["cuda_completion_peak_bytes"] = c.cudaCompletionPeakBytes_;
          cooperative["maximum_cuda_packet_calls"] = c.maximumCudaPacketCalls_;
          cooperative["cuda_continuation_stop_order"] = std::vector<std::string>{
              "completed", "no_progress", "memory",  "call_limit",
              "handoff",   "time",        "disabled"};
          cooperative["cuda_continuation_stops"] = c.cudaContinuationStops_;
          cooperative["cuda_result_service_delay_semantics"] =
              "last autonomous call end to coordinator receipt; buffer delay includes "
              "intervening GPU work";
          cooperative["maximum_host_epochs_per_cuda_job_semantics"] =
              "per time/memory/call-bounded driver packet, not per logical resident call";
          cooperative["adaptive_migration_ms"] = c.adaptiveMigrationMs_;
          cooperative["adaptive_calibration_rule"] =
              "inputs >= 7/8 * min(input_limit, 4096); elapsed-time-weighted";
          cooperative["adaptive_completion_reason_order"] =
              std::vector<std::string>{"completed",     "minimum_batch", "workspace",
                                       "history_lease", "wave_lease",    "other"};
          for (unsigned e = 0; e < 2; ++e)
            for (unsigned k = 0; k < 2; ++k) {
              auto channel = cooperative["adaptive"][e == 0 ? "cuda" : "openmp"]
                                        [k == 0 ? "photon" : "lepton"];
              channel["records_per_ms"] = c.adaptiveRecordsPerMs_[e][k];
              channel["observations"] = c.adaptiveObservations_[e][k];
              channel["wave_limit"] = c.adaptiveWaveLimit_[e][k];
              channel["transport_records"] = c.adaptiveTransportRecords_[e][k];
              channel["resident_wavefronts"] = c.adaptiveResidentWavefronts_[e][k];
              channel["input_limit"] = c.adaptiveInputLimit_[e][k];
              channel["full_batch_observations"] = c.adaptiveFullObservations_[e][k];
              channel["completion_reasons"] = c.adaptiveCompletionReasons_[e][k];
            }
          for (unsigned e = 0; e < 2; ++e) {
            auto endpoint = cooperative["adaptive"][e == 0 ? "cuda" : "openmp"];
            endpoint["species_coalesces"] = c.adaptiveSpeciesCoalesces_[e];
            endpoint["job_input_histogram_floor_log2"] = c.adaptiveJobInputHistogram_[e];
          }
        }
        cooperative["independent_drivers"] = c.independentDrivers_;
        cooperative["independent_subshowers"] = c.independentSubshowers_;
        cooperative["subshower_cuda_submissions"] = c.subshowerCudaSubmissions_;
        cooperative["subshower_cuda_commits"] = c.subshowerCudaCommits_;
        cooperative["subshower_openmp_epochs"] = c.subshowerOpenmpEpochs_;
        cooperative["maximum_host_epochs_per_cuda_job"] = c.maximumHostEpochsPerCudaJob_;
        cooperative["subshower_tail_migrations"] = c.subshowerTailMigrations_;
        cooperative["subshower_proactive_refills"] = c.subshowerProactiveRefills_;
        cooperative["subshower_inflight_waiting_migrations"] =
            c.subshowerInflightWaitingMigrations_;
        cooperative["subshower_inflight_waiting_particles"] =
            c.subshowerInflightWaitingParticles_;
        cooperative["subshower_cuda_epoch_reductions"] = c.subshowerCudaEpochReductions_;
        cooperative["subshower_gpu_lepton_wave_limit"] = c.subshowerGpuLeptonWaveLimit_;
        cooperative["subshower_initial_host_share"] = c.subshowerInitialHostShare_;
        cooperative["subshower_maximum_cuda_epoch_ms"] = c.subshowerMaximumCudaEpochMs_;
        cooperative["subshower_host_queue_peak_bytes"] = c.subshowerHostQueuePeakBytes_;
        cooperative["subshower_requeued_spill_particles"] =
            c.subshowerRequeuedSpillParticles_;
        cooperative["subshower_cross_endpoint_spill_particles"] =
            c.subshowerCrossEndpointSpillParticles_;
        cooperative["coordinator_idle_wait_ms"] = c.coordinatorIdleWaitMs_;
        cooperative["cuda_result_service_delay_ms"] = c.cudaResultServiceDelayMs_;
        cooperative["independent_joint_calls"] = c.independentJointCalls_;
        cooperative["cuda_driver_wall_ms"] = c.cudaDriverWallMs_;
        cooperative["joint_wall_ms"] = c.jointWallMs_;
        cooperative["endpoint_window_overlap_ms"] = c.endpointWindowOverlapMs_;
        cooperative["cuda_finished_before_host_ms"] = c.cudaFinishedBeforeHostMs_;
        cooperative["host_finished_before_cuda_ms"] = c.hostFinishedBeforeCudaMs_;
        cooperative["cuda_input_particles"] = c.cudaInputParticles_;
        cooperative["openmp_input_particles"] = c.openmpInputParticles_;
        cooperative["openmp_slices"] = c.openmpSlices_;
        cooperative["slices_while_cuda_pending"] = c.slicesWhileCudaPending_;
        cooperative["oversized_slices_2ms"] = c.oversizedSlices_;
        cooperative["openmp_wall_ms"] = c.openmpWallMs_;
        cooperative["openmp_while_cuda_pending_ms"] = c.openmpWhileCudaPendingMs_;
        cooperative["maximum_slice_ms"] = c.maximumSliceMs_;
        cooperative["migration_bytes"] = c.migrationBytes_;
        cooperative["photon_calls"] = c.photonCalls_;
        cooperative["lepton_calls"] = c.leptonCalls_;
        cooperative["openmp_batch_target"] = c.openmpBatchTarget_;
        cooperative["openmp_workspace_bytes"] = c.openmpWorkspaceBytes_;
        cooperative["overlap_timing_semantics"] =
            "Independent host-driver call window intersection; not measured "
            "kernel-overlap time";
      }
      auto tuning = accelerator["tuning"];
      tuning["cache_matched"] = backendStats.tuningCacheMatched_;
      tuning["cache_required"] = backendStats.tuningCacheRequired_;
      tuning["cache_hash"] = backendStats.tuningCacheHash_;
      tuning["batch_size"] = static_cast<std::uint64_t>(backendStats.tuningBatchSize_);
      tuning["chunk_size"] = static_cast<std::uint64_t>(backendStats.tuningChunkSize_);
      tuning["team_size"] = static_cast<std::uint64_t>(backendStats.tuningTeamSize_);
      tuning["track_tile_size"] =
          static_cast<std::uint64_t>(backendStats.tuningTrackTileSize_);
      tuning["observer_tile_size"] =
          static_cast<std::uint64_t>(backendStats.tuningObserverTileSize_);
      tuning["device_queues"] =
          static_cast<std::uint64_t>(backendStats.tuningDeviceQueues_);
    }
    showerMetadata["cpu_particle_steps"] =
        EAS.schedulerStatistics().acquiredParticleSteps_ - routerStats.particlesStaged_;
    showerMetadata["particles_staged"] = routerStats.particlesStaged_;
    if (gpuCli.emBackend_ == "kokkos-proposal") {
      showerMetadata["routing"]["attempts"] = routerStats.routeAttempts_;
      showerMetadata["routing"]["non_em_rejections"] = routerStats.routeNonEmRejections_;
      showerMetadata["routing"]["backend_rejections"] =
          routerStats.routeBackendRejections_;
      showerMetadata["routing"]["environment_rejections"] =
          routerStats.routeEnvironmentRejections_;
    }
    showerMetadata["gpu_muon_transport_enabled"] = gpuMuonTransportAvailable;
    showerMetadata["wavefronts"] = routerStats.wavefronts_;
    showerMetadata["resident_photon_wavefronts"] = routerStats.residentPhotonWavefronts_;
    showerMetadata["resident_lepton_wavefronts"] = routerStats.residentLeptonWavefronts_;
    showerMetadata["below_minimum_batch_checkpoints"] =
        routerStats.belowMinimumBatchCheckpoints_;
    showerMetadata["workspace_limit_checkpoints"] =
        routerStats.workspaceLimitCheckpoints_;
    showerMetadata["input_batch_splits"] = routerStats.inputBatchSplits_;
    showerMetadata["maximum_input_batch"] =
        static_cast<std::uint64_t>(routerStats.maximumInputBatch_);
    showerMetadata["gpu_final_states"] = backendStats.gpuFinalStates_;
    showerMetadata["first_interaction_candidates"] =
        backendStats.firstInteractionCandidates_;
    showerMetadata["first_interactions_written"] =
        outputSink.statistics().firstInteractions_;
    showerMetadata["physical_secondaries"] = backendStats.physicalSecondariesGenerated_;
    showerMetadata["cpu_generic_fallbacks"] =
        routerStats.particlesReturnedForCpuFallback_;
    showerMetadata["cpu_memory_spill_particles"] =
        routerStats.particlesReturnedForCpuMemorySpill_;
    showerMetadata["cpu_decay_particles"] = routerStats.particlesReturnedForCpuDecay_;
    showerMetadata["forced_cpu_decays_executed"] = routerStats.forcedCpuDecaysExecuted_;
    showerMetadata["cpu_specified_final_states"] = routerStats.specifiedCpuFinalStates_;
    showerMetadata["deferred_cpu_fallbacks_queued"] =
        routerStats.deferredCpuFallbacksQueued_;
    showerMetadata["deferred_cpu_fallbacks_flushed"] =
        routerStats.deferredCpuFallbacksFlushed_;
    showerMetadata["deferred_cpu_fallback_flushes"] =
        routerStats.deferredCpuFallbackFlushes_;
    showerMetadata["deferred_fallback_scalar_expansion_rounds"] =
        routerStats.deferredFallbackScalarExpansionRounds_;
    showerMetadata["deferred_front_gpu_flushes"] = routerStats.deferredFrontGpuFlushes_;
    showerMetadata["deferred_product_gpu_flushes"] =
        routerStats.deferredProductGpuFlushes_;
    showerMetadata["maximum_deferred_cpu_fallback_batch"] =
        static_cast<std::uint64_t>(routerStats.maximumDeferredCpuFallbackBatch_);
    showerMetadata["cpu_completed_selected_losses"] =
        proposalFallbackStats.completedSelectedLosses_;
    showerMetadata["cpu_completed_native_selection_replays"] =
        proposalFallbackStats.completedNativeSelectionReplays_;
    showerMetadata["photo_hadronic_generator"]["sophia_interactions"] =
        photoHadronicSophia;
    showerMetadata["photo_hadronic_generator"]["threshold_fallback_interactions"] =
        photoHadronicThresholdFallback;
    showerMetadata["photo_hadronic_generator"]["preferred_interactions"] =
        photoHadronicPreferred;
    showerMetadata["photo_hadronic_generator"]["fallback_interactions"] =
        photoHadronicFallback;
    showerMetadata["photo_hadronic_generator"]["discarded_final_states"] = 0;
    showerMetadata["photo_hadronic_generator"]["fallback_model_initialized"] =
        photoHadronicQgsjetFallback.initialized();
    showerMetadata["hybrid_timing_ms"]["total_run"] = hybridTiming.totalRunTimeMs_;
    showerMetadata["hybrid_timing_ms"]["output_start"] = hybridTiming.outputStartTimeMs_;
    showerMetadata["hybrid_timing_ms"]["set_nodes"] = hybridTiming.setNodesTimeMs_;
    showerMetadata["hybrid_timing_ms"]["route_stage"] = hybridTiming.routeStageTimeMs_;
    showerMetadata["hybrid_timing_ms"]["scalar_stepper"] =
        hybridTiming.scalarStepperTimeMs_;
    showerMetadata["hybrid_timing_ms"]["do_stack"] = hybridTiming.doStackTimeMs_;
    showerMetadata["hybrid_timing_ms"]["router_advance"] =
        hybridTiming.routerAdvanceTimeMs_;
    showerMetadata["hybrid_timing_ms"]["init_cascade_equations"] =
        hybridTiming.initCascadeEquationsTimeMs_;
    showerMetadata["hybrid_timing_ms"]["cascade_equations"] =
        hybridTiming.cascadeEquationsTimeMs_;
    showerMetadata["hybrid_timing_ms"]["output_end"] = hybridTiming.outputEndTimeMs_;
    showerMetadata["hadronic_process_pool"]["backend"] = hadronicBackend;
    showerMetadata["hadronic_process_pool"]["enabled"] =
        hybridTiming.hadronicProcessPoolEnabled_;
    showerMetadata["hadronic_process_pool"]["workers"] =
        hadronicProcessPool ? hadronicProcessPool->workerCount() : 0;
    showerMetadata["hadronic_process_pool"]["minimum_pending_interactions"] =
        hadronicMinBatch;
    showerMetadata["hadronic_process_pool"]["target_batch_cost_ms"] =
        hadronicTargetBatchMs;
    showerMetadata["hadronic_process_pool"]["maximum_batch_items"] = hadronicMaxBatch;
    showerMetadata["hadronic_process_pool"]["initial_interaction_cost_ms"] =
        hadronicInitialCostMs;
    showerMetadata["hadronic_process_pool"]["target_total_cost_ms"] =
        hadronicTargetBatchMs *
        static_cast<double>(hadronicProcessPool ? hadronicProcessPool->workerCount() : 0);
    showerMetadata["hadronic_process_pool"]["prepared_interactions"] =
        hybridTiming.hadronicInteractionsPrepared_;
    showerMetadata["hadronic_process_pool"]["committed_interactions"] =
        hybridTiming.hadronicInteractionsCommitted_;
    showerMetadata["hadronic_process_pool"]["flushes"] =
        hybridTiming.hadronicWorkerFlushes_;
    showerMetadata["hadronic_process_pool"]["cost_triggered_flushes"] =
        hybridTiming.hadronicWorkerCostTriggeredFlushes_;
    showerMetadata["hadronic_process_pool"]["capacity_triggered_flushes"] =
        hybridTiming.hadronicWorkerCapacityTriggeredFlushes_;
    showerMetadata["hadronic_process_pool"]["drain_triggered_flushes"] =
        hybridTiming.hadronicWorkerDrainTriggeredFlushes_;
    showerMetadata["hadronic_process_pool"]["classified_batches"] =
        hybridTiming.hadronicWorkerClassifiedBatches_;
    showerMetadata["hadronic_process_pool"]["batches"] =
        hybridTiming.hadronicWorkerBatches_;
    showerMetadata["hadronic_process_pool"]["secondaries"] =
        hybridTiming.hadronicWorkerSecondaries_;
    showerMetadata["hadronic_process_pool"]["prepare_time_ms"] =
        hybridTiming.hadronicPrepareTimeMs_;
    showerMetadata["hadronic_process_pool"]["execute_time_ms"] =
        hybridTiming.hadronicWorkerExecuteTimeMs_;
    showerMetadata["hadronic_process_pool"]["commit_time_ms"] =
        hybridTiming.hadronicCommitTimeMs_;
    showerMetadata["hadronic_process_pool"]["deferred_counter_samples"] =
        deferredTimingSamples;
    YAML::Node hadronicFlushFingerprints;
    for (auto const& fingerprint : hybridTiming.hadronicFlushFingerprints_) {
      YAML::Node node;
      node["requests"] = fingerprint.requests_;
      node["request_hash"] = std::to_string(fingerprint.requestHash_);
      node["response_hash"] = std::to_string(fingerprint.responseHash_);
      hadronicFlushFingerprints.push_back(std::move(node));
    }
    showerMetadata["hadronic_process_pool"]["flush_fingerprints"] =
        std::move(hadronicFlushFingerprints);
    YAML::Node hadronicFlushLoads;
    for (auto const& load : hybridTiming.hadronicFlushLoadRecords_) {
      YAML::Node node;
      node["trigger"] = std::string(hadronic_queue_flush_trigger_name(load.trigger_));
      node["requests"] = load.requests_;
      YAML::Node predicted;
      YAML::Node actual;
      for (auto const value : load.predictedCostByWorker_) { predicted.push_back(value); }
      for (auto const value : load.actualFinalStateTimeByWorkerMs_) {
        actual.push_back(value);
      }
      node["predicted_cost_by_worker"] = std::move(predicted);
      node["actual_final_state_time_by_worker_ms"] = std::move(actual);
      hadronicFlushLoads.push_back(std::move(node));
    }
    showerMetadata["hadronic_process_pool"]["flush_loads"] =
        std::move(hadronicFlushLoads);
    showerMetadata["hadronic_process_pool"]["maximum_parked_projectiles"] =
        EAS.schedulerStatistics().maximumSuspendedParticles_;
    if (hadronicProcessPool) {
      auto const poolAfter = hadronicProcessPool->statistics();
      showerMetadata["hadronic_process_pool"]["ipc_batches"] =
          poolAfter.batches_ - hadronicPoolStatisticsBefore.batches_;
      showerMetadata["hadronic_process_pool"]["ipc_requests"] =
          poolAfter.requests_ - hadronicPoolStatisticsBefore.requests_;
      showerMetadata["hadronic_process_pool"]["ipc_secondaries"] =
          poolAfter.secondaries_ - hadronicPoolStatisticsBefore.secondaries_;
      showerMetadata["hadronic_process_pool"]["bytes_sent"] =
          poolAfter.bytesSent_ - hadronicPoolStatisticsBefore.bytesSent_;
      showerMetadata["hadronic_process_pool"]["bytes_received"] =
          poolAfter.bytesReceived_ - hadronicPoolStatisticsBefore.bytesReceived_;
      showerMetadata["hadronic_process_pool"]["dispatch_time_ms"] =
          poolAfter.dispatchTimeMs_ - hadronicPoolStatisticsBefore.dispatchTimeMs_;
      showerMetadata["hadronic_process_pool"]["poll_wait_time_ms"] =
          poolAfter.pollWaitTimeMs_ - hadronicPoolStatisticsBefore.pollWaitTimeMs_;
      showerMetadata["hadronic_process_pool"]["receive_time_ms"] =
          poolAfter.receiveTimeMs_ - hadronicPoolStatisticsBefore.receiveTimeMs_;
      showerMetadata["hadronic_process_pool"]["wire_protocol_version"] =
          HadronicBatchProtocolVersion;
      showerMetadata["hadronic_process_pool"]["bulk_response_batches"] =
          poolAfter.bulkResponseBatches_ -
          hadronicPoolStatisticsBefore.bulkResponseBatches_;
      showerMetadata["hadronic_process_pool"]["bulk_response_payload_reads"] =
          poolAfter.bulkResponsePayloadReads_ -
          hadronicPoolStatisticsBefore.bulkResponsePayloadReads_;
    }
    YAML::Node actualHadronicWorkerLoads;
    double actualWorkerMinimumMs = std::numeric_limits<double>::infinity();
    double actualWorkerMaximumMs = 0.;
    for (std::size_t worker = 0;
         hadronicProcessPool && worker < hadronicProcessPool->workerCount(); ++worker) {
      auto const loadFound =
          hybridTiming.hadronicWorkerFinalStateTimeByWorkerMs_.find(worker);
      auto const loadMs =
          loadFound == hybridTiming.hadronicWorkerFinalStateTimeByWorkerMs_.end()
              ? 0.
              : loadFound->second;
      actualWorkerMinimumMs = std::min(actualWorkerMinimumMs, loadMs);
      actualWorkerMaximumMs = std::max(actualWorkerMaximumMs, loadMs);
      YAML::Node workerNode;
      workerNode["worker_id"] = worker;
      workerNode["final_state_time_ms"] = loadMs;
      workerNode["batches"] = hybridTiming.hadronicWorkerBatchesByWorker_.count(worker)
                                  ? hybridTiming.hadronicWorkerBatchesByWorker_.at(worker)
                                  : 0;
      workerNode["requests"] =
          hybridTiming.hadronicWorkerRequestsByWorker_.count(worker)
              ? hybridTiming.hadronicWorkerRequestsByWorker_.at(worker)
              : 0;
      actualHadronicWorkerLoads.push_back(std::move(workerNode));
    }
    if (!hadronicProcessPool) { actualWorkerMinimumMs = 0.; }
    showerMetadata["hadronic_process_pool"]["actual_worker_loads"] =
        std::move(actualHadronicWorkerLoads);
    showerMetadata["hadronic_process_pool"]["actual_generator_load_imbalance_ms"] =
        actualWorkerMaximumMs - actualWorkerMinimumMs;
    YAML::Node scalarStepsByPdg;
    for (auto const& [pdg, count] : hybridTiming.scalarStepsByPdg_) {
      scalarStepsByPdg[std::to_string(pdg)] = count;
    }
    showerMetadata["scalar_steps_by_pdg"] = std::move(scalarStepsByPdg);
    YAML::Node scalarTimeByPdg;
    for (auto const& [pdg, timeMs] : hybridTiming.scalarStepperTimeByPdgMs_) {
      scalarTimeByPdg[std::to_string(pdg)] = timeMs;
    }
    showerMetadata["scalar_stepper_time_by_pdg_ms"] = std::move(scalarTimeByPdg);
    YAML::Node scalarPhaseTiming;
    for (auto const& [pdg, phases] : hybridTiming.scalarPhaseTimingByPdg_) {
      YAML::Node node;
      node["steps"] = phases.steps_;
      node["geometry_boundary_crossings"] = phases.geometryBoundaryCrossings_;
      node["geometry_step_limits"] = phases.geometryStepLimits_;
      node["interactions"] = phases.interactions_;
      node["decays"] = phases.decays_;
      node["secondary_processing_calls"] = phases.secondaryProcessingCalls_;
      node["cross_section_ms"] = phases.crossSectionTimeMs_;
      node["distance_sampling_ms"] = phases.distanceSamplingTimeMs_;
      node["tracking_ms"] = phases.trackingTimeMs_;
      node["continuous_limit_ms"] = phases.continuousLimitTimeMs_;
      node["continuous_process_ms"] = phases.continuousProcessTimeMs_;
      node["geometry_boundary_ms"] = phases.geometryBoundaryTimeMs_;
      node["geometry_step_limit_ms"] = phases.geometryStepLimitTimeMs_;
      node["interaction_ms"] = phases.interactionTimeMs_;
      node["decay_ms"] = phases.decayTimeMs_;
      node["secondary_processing_ms"] = phases.secondaryProcessingTimeMs_;
      scalarPhaseTiming[std::to_string(pdg)] = std::move(node);
    }
    showerMetadata["scalar_phase_timing_by_pdg_ms"] = std::move(scalarPhaseTiming);
    showerMetadata["hadronic_work_classifier"]["transition_energy_GeV"] =
        hybridTiming.hadronicWorkClassifier_.transitionEnergyGeV_;
    showerMetadata["hadronic_work_classifier"]["energy_bins_per_octave"] =
        hybridTiming.hadronicWorkClassifier_.energyBinsPerOctave_;
    YAML::Node hadronicWorkClasses;
    for (auto const& [key, statistics] : hybridTiming.hadronicWorkClasses_) {
      YAML::Node workClass;
      workClass["model"] = std::string(hadronic_model_class_name(key.model_));
      workClass["species"] = std::string(hadronic_species_class_name(key.species_));
      workClass["energy_bin"] = key.energyBin_;
      workClass["energy_lower_GeV"] =
          hadronic_energy_bin_lower_gev(key, hybridTiming.hadronicWorkClassifier_);
      workClass["energy_upper_GeV"] =
          hadronic_energy_bin_upper_gev(key, hybridTiming.hadronicWorkClassifier_);
      workClass["mass_number_bin"] = static_cast<unsigned int>(key.massNumberBin_);
      workClass["steps"] = statistics.steps_;
      workClass["total_time_ms"] = statistics.totalTimeMs_;
      workClass["mean_time_ms"] = statistics.meanTimeMs();
      workClass["standard_deviation_ms"] = statistics.standardDeviationMs();
      workClass["minimum_time_ms"] =
          statistics.steps_ == 0 ? 0. : statistics.minimumTimeMs_;
      workClass["maximum_time_ms"] = statistics.maximumTimeMs_;
      hadronicWorkClasses.push_back(std::move(workClass));
    }
    showerMetadata["hadronic_work_classes"] = std::move(hadronicWorkClasses);
    YAML::Node hadronicFinalStateClassNodes;
    for (auto const& [key, statistics] : hadronicFinalStateClasses) {
      YAML::Node workClass;
      workClass["model"] = std::string(hadronic_model_class_name(key.model_));
      workClass["species"] = std::string(hadronic_species_class_name(key.species_));
      workClass["energy_bin"] = key.energyBin_;
      workClass["energy_lower_GeV"] =
          hadronic_energy_bin_lower_gev(key, hadronicWorkClassifier);
      workClass["energy_upper_GeV"] =
          hadronic_energy_bin_upper_gev(key, hadronicWorkClassifier);
      workClass["mass_number_bin"] = static_cast<unsigned int>(key.massNumberBin_);
      workClass["interactions"] = statistics.steps_;
      workClass["total_time_ms"] = statistics.totalTimeMs_;
      workClass["mean_time_ms"] = statistics.meanTimeMs();
      workClass["standard_deviation_ms"] = statistics.standardDeviationMs();
      workClass["minimum_time_ms"] = statistics.minimumTimeMs_;
      workClass["maximum_time_ms"] = statistics.maximumTimeMs_;
      hadronicFinalStateClassNodes.push_back(std::move(workClass));
    }
    showerMetadata["hadronic_final_state_classes"] =
        std::move(hadronicFinalStateClassNodes);
    showerMetadata["hadronic_worker_oracle"]["mode"] =
        hadronicProcessPool ? "disabled_actual_process_pool_active"
                            : "retrospective_measured_final_state_only";
    showerMetadata["hadronic_worker_oracle"]["workers"] = hadronicPlanWorkers;
    showerMetadata["hadronic_worker_oracle"]["target_batch_cost_ms"] =
        hadronicPlanTargetMs;
    showerMetadata["hadronic_worker_oracle"]["maximum_batch_items"] =
        hadronicPlanMaxBatch;
    showerMetadata["hadronic_worker_oracle"]["interactions"] = hadronicWorkerSequence;
    showerMetadata["hadronic_worker_oracle"]["batches"] = hadronicWorkerBatchCount;
    showerMetadata["hadronic_worker_oracle"]["serial_final_state_time_ms"] =
        hadronicFinalStateSerialTimeMs;
    showerMetadata["hadronic_worker_oracle"]["predicted_makespan_ms"] =
        hadronicWorkerPredictedMakespanMs;
    showerMetadata["hadronic_worker_oracle"]["predicted_kernel_speedup"] =
        hadronicWorkerPredictedMakespanMs > 0.
            ? hadronicFinalStateSerialTimeMs / hadronicWorkerPredictedMakespanMs
            : 0.;
    showerMetadata["hadronic_worker_oracle"]["load_imbalance_ms"] =
        hadronicWorkerPredictedMakespanMs - hadronicWorkerMinimumLoadMs;
    YAML::Node hadronicWorkerLoads;
    for (auto const& assignment : hadronicWorkerAssignments) {
      YAML::Node worker;
      worker["worker_id"] = assignment.workerId_;
      worker["estimated_load_ms"] = assignment.estimatedCost_;
      worker["batches"] = assignment.batches_.size();
      hadronicWorkerLoads.push_back(std::move(worker));
    }
    showerMetadata["hadronic_worker_oracle"]["worker_loads"] =
        std::move(hadronicWorkerLoads);
    showerMetadata["cpu_fallback_steps_executed"] = routerStats.cpuFallbackStepsExecuted_;
    showerMetadata["cpu_wavefront_expansion_steps_executed"] =
        routerStats.cpuWavefrontExpansionStepsExecuted_;
    showerMetadata["particles_returned_for_cpu_wavefront_expansion"] =
        routerStats.particlesReturnedForCpuWavefrontExpansion_;
    showerMetadata["small_batch_expansions"] = routerStats.smallBatchExpansions_;
    showerMetadata["stalled_boundary_gpu_flushes"] =
        routerStats.stalledBoundaryGpuFlushes_;
    showerMetadata["scalar_expansion_budget_gpu_flushes"] =
        routerStats.scalarExpansionBudgetGpuFlushes_;
    showerMetadata["cpu_specified_fallback_time_ms"] =
        routerStats.specifiedCpuFallbackTimeMs_;
    showerMetadata["photon_backend_wall_time_ms"] = routerStats.photonBackendWallTimeMs_;
    showerMetadata["lepton_backend_wall_time_ms"] = routerStats.leptonBackendWallTimeMs_;
    showerMetadata["photon_host_postprocess_time_ms"] =
        routerStats.photonHostPostprocessTimeMs_;
    showerMetadata["lepton_host_postprocess_time_ms"] =
        routerStats.leptonHostPostprocessTimeMs_;
    YAML::Node fallbacksByProcess;
    YAML::Node fallbacksByProcessName;
    for (auto const& [processId, count] : routerStats.cpuFallbacksByProcess_) {
      fallbacksByProcess[std::to_string(processId)] = count;
      fallbacksByProcessName[em_process_name(processId)] = count;
    }
    showerMetadata["cpu_fallbacks_by_process"] = std::move(fallbacksByProcess);
    showerMetadata["cpu_fallbacks_by_process_name"] = std::move(fallbacksByProcessName);
    YAML::Node fallbacksByReason;
    YAML::Node fallbacksByReasonName;
    for (auto const& [reason, count] : routerStats.cpuFallbacksByReason_) {
      fallbacksByReason[std::to_string(reason)] = count;
      auto const typedReason = static_cast<FallbackReason>(reason);
      fallbacksByReasonName[proposal_fallback_reason_name(typedReason)] = count;
    }
    showerMetadata["cpu_fallbacks_by_reason"] = std::move(fallbacksByReason);
    showerMetadata["cpu_fallbacks_by_reason_name"] = std::move(fallbacksByReasonName);
    showerMetadata["observed"] = routerStats.particlesObserved_;
    showerMetadata["escaped"] = routerStats.particlesEscaped_;
    showerMetadata["cut"] = routerStats.particlesCut_;
    showerMetadata["weighted_deposit_GeV"] = sinkStats.weightedDepositedEnergyGeV_;
    showerMetadata["weighted_muon_parent_productions"] =
        sinkStats.weightedMuonParentProductions_;
    showerMetadata["radio_tracks"] = sinkStats.radioTracks_;
    showerMetadata["radio"]["backend"] = gpuRadioEnabled ? acceleratedBackendName : "cpu";
    showerMetadata["radio"]["host_tracks_enqueued"] =
        backendStats.radio_.hostTracksEnqueued_;
    showerMetadata["radio"]["host_track_batches"] = backendStats.radio_.hostTrackBatches_;
    showerMetadata["radio"]["host_track_inputs"] = gpuCli.radioTrackInputs_;
    showerMetadata["radio"]["track_observer_pairs"] =
        backendStats.radio_.trackObserverPairs_;
    showerMetadata["radio"]["fused_track_observer_pairs"] =
        backendStats.radio_.fusedTrackObserverPairs_;
    showerMetadata["radio"]["coreas_contributions"] =
        backendStats.radio_.coreasContributions_;
    showerMetadata["radio"]["zhs_contributions"] = backendStats.radio_.zhsContributions_;
    showerMetadata["radio"]["zhs_subtracks"] = backendStats.radio_.zhsSubtracks_;
    showerMetadata["radio"]["deterministic"] = gpuDeterministic;
    showerMetadata["radio"]["fixed_point_field_limit_V_per_m"] = gpuRadioFieldLimit;
    showerMetadata["radio"]["fixed_point_overflows"] =
        backendStats.radio_.fixedPointOverflows_;
    showerMetadata["radio"]["track_diagnostics_enabled"] =
        backendStats.radio_.trackDiagnosticsEnabled_;
    showerMetadata["radio"]["segment_count"] = backendStats.radio_.leptonTracks_;
    showerMetadata["radio"]["weighted_segment_count"] =
        backendStats.radio_.weightedSegmentCount_;
    showerMetadata["radio"]["track_length_m"] = backendStats.radio_.trackLengthM_;
    showerMetadata["radio"]["weighted_track_length_m"] =
        backendStats.radio_.weightedTrackLengthM_;
    showerMetadata["radio"]["electron_weighted_track_length_m"] =
        backendStats.radio_.electronWeightedTrackLengthM_;
    showerMetadata["radio"]["positron_weighted_track_length_m"] =
        backendStats.radio_.positronWeightedTrackLengthM_;
    showerMetadata["radio"]["signed_charge_weighted_track_length_m"] =
        backendStats.radio_.signedChargeWeightedTrackLengthM_;
    showerMetadata["radio"]["energy_weighted_track_length_GeV_m"] =
        backendStats.radio_.energyWeightedTrackLengthGeVM_;
    showerMetadata["radio"]["maximum_segment_length_m"] =
        backendStats.radio_.maximumSegmentLengthM_;
    showerMetadata["radio"]["weighted_direction_change_rad"] =
        backendStats.radio_.weightedDirectionChangeRad_;
    showerMetadata["radio"]["weighted_direction_change_squared_rad2"] =
        backendStats.radio_.weightedDirectionChangeSquaredRad2_;
    showerMetadata["radio"]["weighted_beta_deficit_track_length_m"] =
        backendStats.radio_.weightedBetaDeficitTrackLengthM_;
    showerMetadata["radio"]["weighted_time_residual_s"] =
        backendStats.radio_.weightedTimeResidualS_;
    showerMetadata["radio"]["maximum_direction_change_rad"] =
        backendStats.radio_.maximumDirectionChangeRad_;
    showerMetadata["radio"]["signed_charge_weighted_direction_change"]["x"] =
        backendStats.radio_.signedChargeWeightedDirectionChange_[0];
    showerMetadata["radio"]["signed_charge_weighted_direction_change"]["y"] =
        backendStats.radio_.signedChargeWeightedDirectionChange_[1];
    showerMetadata["radio"]["signed_charge_weighted_direction_change"]["z"] =
        backendStats.radio_.signedChargeWeightedDirectionChange_[2];
    auto radioEnergyBins =
        showerMetadata["radio"]["weighted_track_length_by_kinetic_energy"];
    radioEnergyBins["units"]["upper_edge"] = "GeV";
    radioEnergyBins["units"]["weighted_track_length"] = "m";
    constexpr std::array<double, 14> radioEnergyUpperEdgesGeV{
        1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2, 1.e-1,
        2.e-1, 5.e-1, 1.,    2.,    5.,    10.,   100.};
    for (std::size_t index = 0; index < 15; ++index) {
      radioEnergyBins["upper_edge_GeV"].push_back(
          index < radioEnergyUpperEdgesGeV.size()
              ? YAML::Node(radioEnergyUpperEdgesGeV[index])
              : YAML::Node("inf"));
      radioEnergyBins["weighted_track_length_m"].push_back(
          backendStats.radio_.weightedTrackLengthByKineticEnergyM_[index]);
    }
    showerMetadata["radio"]["device_bytes"] =
        static_cast<std::uint64_t>(backendStats.radio_.deviceBytes_);
    showerMetadata["radio"]["host_to_device_bytes"] =
        backendStats.radio_.hostToDeviceBytes_;
    showerMetadata["radio"]["device_to_host_bytes"] =
        backendStats.radio_.deviceToHostBytes_;
    showerMetadata["radio"]["device_time_ms"] = backendStats.radio_.deviceTimeMs_;
    showerMetadata["radio"]["kernel_launch_time_ms"] = backendStats.radio_.kernelTimeMs_;
    showerMetadata["radio"]["transfer_time_ms"] = backendStats.radio_.transferTimeMs_;
    showerMetadata["radio"]["input_slot_waits"] = backendStats.radio_.inputSlotWaits_;
    showerMetadata["radio"]["input_slot_host_wait_time_ms"] =
        backendStats.radio_.inputSlotHostWaitTimeMs_;
    showerMetadata["radio"]["track_precompute_enabled"] =
        backendStats.radio_.trackPrecomputeEnabled_;
    showerMetadata["radio"]["track_tile_size"] = backendStats.radio_.trackTileSize_;
    showerMetadata["radio"]["observer_tile_size"] = backendStats.radio_.observerTileSize_;
    showerMetadata["radio"]["track_precompute_batches"] =
        backendStats.radio_.trackPrecomputeBatches_;
    showerMetadata["radio"]["track_precomputed_records"] =
        backendStats.radio_.trackPrecomputedRecords_;
    showerMetadata["radio"]["direct_projection_batches"] =
        backendStats.radio_.directProjectionBatches_;
    showerMetadata["radio"]["direct_projection_records"] =
        backendStats.radio_.directProjectionRecords_;
    showerMetadata["radio"]["projection_tiles"] = backendStats.radio_.projectionTiles_;
    showerMetadata["radio"]["track_workspace_bytes"] =
        static_cast<std::uint64_t>(backendStats.radio_.trackWorkspaceBytes_);
    showerMetadata["radio"]["maximum_track_batch"] =
        static_cast<std::uint64_t>(backendStats.radio_.maximumTrackBatch_);
    showerMetadata["radio"]["track_precompute_device_time_ms"] =
        backendStats.radio_.trackPrecomputeDeviceTimeMs_;
    showerMetadata["radio"]["projection_device_time_ms"] =
        backendStats.radio_.projectionDeviceTimeMs_;
    showerMetadata["profile"]["backend"] =
        backendStats.profile_.enabled_ ? acceleratedBackendName : "host";
    showerMetadata["profile"]["deterministic"] = backendStats.profile_.deterministic_;
    showerMetadata["profile"]["bins"] =
        static_cast<std::uint64_t>(backendStats.profile_.bins_);
    showerMetadata["profile"]["steps"] = backendStats.profile_.steps_;
    showerMetadata["profile"]["deposited_steps"] = backendStats.profile_.depositedSteps_;
    showerMetadata["profile"]["fixed_point_overflows"] =
        backendStats.profile_.fixedPointOverflows_;
    showerMetadata["profile"]["invalid_records"] = backendStats.profile_.invalidRecords_;
    showerMetadata["profile"]["device_bytes"] =
        static_cast<std::uint64_t>(backendStats.profile_.deviceBytes_);
    showerMetadata["profile"]["device_to_host_bytes"] =
        backendStats.profile_.deviceToHostBytes_;
    showerMetadata["profile"]["kernel_time_ms"] = backendStats.profile_.kernelTimeMs_;
    showerMetadata["profile"]["transfer_time_ms"] = backendStats.profile_.transferTimeMs_;
    showerMetadata["thinning"]["hillas_vertices"] = backendStats.thinningHillasVertices_;
    showerMetadata["thinning"]["statistical_vertices"] =
        backendStats.thinningStatisticalVertices_;
    showerMetadata["thinning"]["particles_discarded"] =
        backendStats.thinningParticlesDiscarded_;
    showerMetadata["processes"]["annihilation"] = backendStats.annihilationFinalStates_;
    showerMetadata["processes"]["photon_pair"] = backendStats.photonPairFinalStates_;
    showerMetadata["processes"]["bremsstrahlung"] = backendStats.bremsFinalStates_;
    showerMetadata["processes"]["compton"] = backendStats.comptonFinalStates_;
    showerMetadata["processes"]["photoelectric"] = backendStats.photoelectricFinalStates_;
    showerMetadata["processes"]["ionization"] = backendStats.ionizationFinalStates_;
    showerMetadata["processes"]["electron_pair"] = backendStats.electronPairFinalStates_;
    showerMetadata["epair_sampler"]["rejection_trials"] =
        backendStats.electronPairRejectionTrials_;
    showerMetadata["epair_sampler"]["zero_weight_samples"] =
        backendStats.electronPairZeroWeightSamples_;
    showerMetadata["epair_sampler"]["cpu_fallbacks"] =
        backendStats.electronPairRejectionFallbacks_;
    showerMetadata["epair_sampler"]["envelope_violations"] =
        backendStats.electronPairEnvelopeViolations_;
    showerMetadata["moliere"]["trials"] = backendStats.moliereTrials_;
    showerMetadata["moliere"]["deflections"] = backendStats.moliereDeflections_;
    showerMetadata["moliere"]["zero_deflections"] = backendStats.moliereZeroDeflections_;
    showerMetadata["moliere"]["newton_iterations"] =
        backendStats.moliereNewtonIterations_;
    showerMetadata["moliere"]["maximum_newton_iterations"] =
        backendStats.moliereMaxNewtonIterations_;
    showerMetadata["moliere"]["mean_newton_iterations"] =
        backendStats.moliereDeflections_ == 0
            ? 0.
            : static_cast<double>(backendStats.moliereNewtonIterations_) /
                  static_cast<double>(backendStats.moliereDeflections_);
    showerMetadata["queue_overflows"] = backendStats.queueOverflows_;
    showerMetadata["peak_device_bytes"] =
        static_cast<std::uint64_t>(backendStats.peakDeviceBytes_);
    showerMetadata["table_device_bytes"] =
        static_cast<std::uint64_t>(backendStats.tableDeviceBytes_);
    showerMetadata["gpu_physics_source"] = "proposal-native";
    if (backendStats.physicsSource_ == PhysicsSource::ProposalNative) {
      showerMetadata["proposal_native"]["proposal_version"] =
          backendStats.nativeProposalVersion_;
      showerMetadata["proposal_native"]["cubic_interpolation_version"] =
          backendStats.nativeCubicInterpolationVersion_;
      showerMetadata["proposal_native"]["table_sha256"] =
          to_hex(backendStats.nativeTableHash_);
      showerMetadata["proposal_native"]["node_count"] = backendStats.nativeTableNodes_;
      showerMetadata["proposal_native"]["device_bytes"] =
          static_cast<std::uint64_t>(backendStats.nativeTableDeviceBytes_);
      showerMetadata["proposal_native"]["aux_sha256"] =
          to_hex(backendStats.auxiliaryCacheHash_);
      showerMetadata["proposal_native"]["aux_cache_hit"] =
          backendStats.auxiliaryCacheHit_;
      showerMetadata["proposal_native"]["proposal_cache_table_count"] =
          backendStats.proposalCacheTableCount_;
      showerMetadata["proposal_native"]["proposal_cache_hit_count"] =
          backendStats.proposalCacheHitCount_;
      showerMetadata["proposal_native"]["proposal_cache_all_hit"] =
          backendStats.proposalCacheAllHit_;
      showerMetadata["proposal_native"]["newton_iterations"] =
          backendStats.nativeNewtonIterations_;
      showerMetadata["proposal_native"]["bisection_iterations"] =
          backendStats.nativeBisectionIterations_;
      showerMetadata["proposal_native"]["inverse_failures"] =
          backendStats.nativeInverseFailures_;
    }
    showerMetadata["workspace_bytes"] =
        static_cast<std::uint64_t>(backendStats.physicalWorkspaceBytes_);
    showerMetadata["maximum_resident_photon_batch"] =
        static_cast<std::uint64_t>(backendStats.maximumResidentPhotonBatch_);
    showerMetadata["maximum_resident_lepton_batch"] =
        static_cast<std::uint64_t>(backendStats.maximumResidentLeptonBatch_);
    if (backendStats.automaticCapacityBudgetBytes_ != 0) {
      auto capacity = showerMetadata["automatic_resident_capacity"];
      capacity["policy"] = "kokkos-budgeted-arenas-v1";
      capacity["budget_bytes"] = backendStats.automaticCapacityBudgetBytes_;
      capacity["planned_retained_bytes"] = backendStats.automaticCapacityPlannedBytes_;
      capacity["planned_peak_bytes"] = backendStats.automaticCapacityPeakBytes_;
      capacity["allocator_reserve_bytes"] =
          backendStats.automaticCapacityAllocatorReserveBytes_;
    }
    showerMetadata["cross_species"]["queue_device_bytes"] =
        static_cast<std::uint64_t>(backendStats.crossSpeciesQueueDeviceBytes_);
    showerMetadata["cross_species"]["enabled"] = gpuConfig.residentCrossSpecies_;
    showerMetadata["cross_species"]["queue_capacity_per_pid"] =
        static_cast<std::uint64_t>(backendStats.crossSpeciesQueueCapacityPerPid_);
    showerMetadata["cross_species"]["peak_pending_photons"] =
        static_cast<std::uint64_t>(backendStats.peakPendingPhotons_);
    showerMetadata["cross_species"]["peak_pending_leptons"] =
        static_cast<std::uint64_t>(backendStats.peakPendingLeptons_);
    showerMetadata["cross_species"]["particles_kept_on_device"] =
        backendStats.crossSpeciesParticlesKeptOnDevice_;
    showerMetadata["cross_species"]["device_to_device_bytes"] =
        backendStats.crossSpeciesDeviceToDeviceBytes_;
    showerMetadata["cross_species"]["host_spills"] = backendStats.crossSpeciesHostSpills_;
    showerMetadata["cross_species"]["spill_rebalances"] =
        backendStats.crossSpeciesSpillRebalances_;
    showerMetadata["cross_species"]["particles_spilled_to_cpu"] =
        backendStats.crossSpeciesParticlesSpilledToCpu_;
    showerMetadata["cross_species"]["low_energy_ordering_checks"] =
        backendStats.crossSpeciesLowEnergyOrderingChecks_;
    showerMetadata["cross_species"]["cpu_spill_steps_executed"] =
        routerStats.cpuMemorySpillStepsExecuted_;
    showerMetadata["wavefront_bucketing"]["key"] =
        "PID x medium_id x logarithmic_energy_bin";
    showerMetadata["wavefront_bucketing"]["energy_bins_per_octave"] = 16;
    showerMetadata["wavefront_bucketing"]["stable"] = true;
    showerMetadata["wavefront_bucketing"]["minimum_radix_sort_size"] =
        static_cast<std::uint64_t>(256);
    showerMetadata["wavefront_bucketing"]["batches"] =
        backendStats.wavefrontBucketingBatches_;
    showerMetadata["wavefront_bucketing"]["particles"] =
        backendStats.wavefrontBucketingParticles_;
    showerMetadata["wavefront_bucketing"]["small_batches"] =
        backendStats.wavefrontBucketingSmallBatches_;
    showerMetadata["wavefront_bucketing"]["small_particles"] =
        backendStats.wavefrontBucketingSmallParticles_;
    showerMetadata["lepton_transport"]["interaction_candidates"] =
        backendStats.leptonTransportInteractionCandidates_;
    showerMetadata["lepton_transport"]["continuous_steps"] =
        backendStats.leptonTransportContinuousSteps_;
    showerMetadata["lepton_transport"]["particle_cuts"] =
        backendStats.leptonTransportCuts_;
    showerMetadata["lepton_transport"]["layer_boundaries"] =
        backendStats.leptonTransportBoundaries_;
    showerMetadata["lepton_transport"]["observations"] =
        backendStats.leptonTransportObservations_;
    showerMetadata["lepton_transport"]["escapes"] = backendStats.leptonTransportEscapes_;
    showerMetadata["lepton_transport"]["magnetic_steps"] =
        backendStats.leptonTransportMagneticSteps_;
    showerMetadata["lepton_transport"]["decay_candidates"] =
        backendStats.leptonTransportDecayCandidates_;
    showerMetadata["cross_species"]["final_pending_photons"] =
        static_cast<std::uint64_t>(backend.pendingPhotonCount());
    showerMetadata["cross_species"]["final_pending_leptons"] =
        static_cast<std::uint64_t>(backend.pendingLeptonCount());
    showerMetadata["host_to_device_bytes"] = backendStats.physicalHostToDeviceBytes_;
    showerMetadata["device_to_host_bytes"] = backendStats.physicalDeviceToHostBytes_;
    showerMetadata["pipeline_control"]["photon_selection_transport_summary_fusions"] =
        backendStats.photonSelectionTransportSummaryFusions_;
    showerMetadata["pipeline_control"]["lepton_selection_transport_summary_fusions"] =
        backendStats.leptonSelectionTransportSummaryFusions_;
    showerMetadata["pipeline_control"]["photon_transport_final_state_summary_fusions"] =
        backendStats.photonTransportFinalStateSummaryFusions_;
    showerMetadata["pipeline_control"]["lepton_transport_vertex_summary_fusions"] =
        backendStats.leptonTransportVertexSummaryFusions_;
    showerMetadata["pipeline_control"]["lepton_vertex_final_state_summary_fusions"] =
        backendStats.leptonVertexFinalStateSummaryFusions_;
    showerMetadata["pipeline_control"]["photon_final_state_endpoint_summary_fusions"] =
        backendStats.photonFinalStateEndpointSummaryFusions_;
    showerMetadata["pipeline_control"]["lepton_final_state_endpoint_summary_fusions"] =
        backendStats.leptonFinalStateEndpointSummaryFusions_;
    showerMetadata["pipeline_control"]["host_synchronizations_eliminated"] =
        backendStats.pipelineHostSynchronizationsEliminated_;
    showerMetadata["pipeline_control"]["device_to_host_bytes_eliminated"] =
        backendStats.pipelineDeviceToHostBytesEliminated_;
    showerMetadata["lepton_pipeline_timing"]["enabled"] =
        backendStats.leptonPipelineTiming_.enabled_;
    showerMetadata["lepton_pipeline_timing"]["wavefronts"] =
        backendStats.leptonPipelineTiming_.wavefronts_;
    showerMetadata["lepton_pipeline_timing"]["selection_ms"] =
        backendStats.leptonPipelineTiming_.selectionMs_;
    showerMetadata["lepton_pipeline_timing"]["transport_ms"] =
        backendStats.leptonPipelineTiming_.transportMs_;
    showerMetadata["lepton_pipeline_timing"]["transport_physics_ms"] =
        backendStats.leptonPipelineTiming_.transportPhysicsMs_;
    showerMetadata["lepton_pipeline_timing"]["moliere_ms"] =
        backendStats.leptonPipelineTiming_.moliereMs_;
    showerMetadata["lepton_pipeline_timing"]["transport_control_ms"] =
        backendStats.leptonPipelineTiming_.transportControlMs_;
    showerMetadata["lepton_pipeline_timing"]["transport_compaction_ms"] =
        backendStats.leptonPipelineTiming_.transportCompactionMs_;
    showerMetadata["lepton_pipeline_timing"]["interaction_extraction_ms"] =
        backendStats.leptonPipelineTiming_.interactionExtractionMs_;
    showerMetadata["lepton_pipeline_timing"]["vertex_selection_ms"] =
        backendStats.leptonPipelineTiming_.vertexSelectionMs_;
    showerMetadata["lepton_pipeline_timing"]["final_state_ms"] =
        backendStats.leptonPipelineTiming_.finalStateMs_;
    showerMetadata["lepton_pipeline_timing"]["final_state_classification_ms"] =
        backendStats.leptonPipelineTiming_.finalStateClassificationMs_;
    showerMetadata["lepton_pipeline_timing"]["final_state_scan_ms"] =
        backendStats.leptonPipelineTiming_.finalStateScanMs_;
    showerMetadata["lepton_pipeline_timing"]["final_state_summary_ms"] =
        backendStats.leptonPipelineTiming_.finalStateSummaryMs_;
    showerMetadata["lepton_pipeline_timing"]["final_state_write_ms"] =
        backendStats.leptonPipelineTiming_.finalStateWriteMs_;
    showerMetadata["lepton_pipeline_timing"]["endpoint_compaction_ms"] =
        backendStats.leptonPipelineTiming_.endpointCompactionMs_;
    showerMetadata["lepton_pipeline_timing"]["post_endpoint_ms"] =
        backendStats.leptonPipelineTiming_.postEndpointMs_;
    showerMetadata["kernel_time_ms"] = backendStats.kernelTimeMs_;
    showerMetadata["transfer_time_ms"] = backendStats.transferTimeMs_;
    showerMetadata["timing_schema_version"] = 2;
    showerMetadata["timing_semantics"]["kernel_time_ms"] =
        "sum of CUDA-event durations across streams; overlapping streams "
        "must not be added to shower wall time";
    showerMetadata["timing_semantics"]["transfer_time_ms"] =
        "legacy host API wall time around selected synchronous CUDA copies";
    showerMetadata["transfer_timing"]["device_event_timing_enabled"] =
        backendStats.transferTiming_.deviceEventTimingEnabled_;
    showerMetadata["transfer_timing"]["operations"] =
        backendStats.transferTiming_.operations_;
    showerMetadata["transfer_timing"]["host_to_device_operations"] =
        backendStats.transferTiming_.hostToDeviceOperations_;
    showerMetadata["transfer_timing"]["device_to_host_operations"] =
        backendStats.transferTiming_.deviceToHostOperations_;
    showerMetadata["transfer_timing"]["device_to_device_operations"] =
        backendStats.transferTiming_.deviceToDeviceOperations_;
    showerMetadata["transfer_timing"]["host_api_time_ms"] =
        backendStats.transferTiming_.hostApiTimeMs_;
    showerMetadata["transfer_timing"]["device_copy_time_ms"] =
        backendStats.transferTiming_.deviceCopyTimeMs_;
    showerMetadata["transfer_timing"]["host_wait_upper_bound_ms"] =
        backendStats.transferTiming_.hostWaitUpperBoundMs_;
    showerMetadata["synchronization_timing"]["physical_pipeline_waits"] =
        backendStats.synchronizationTiming_.physicalPipelineWaits_;
    showerMetadata["synchronization_timing"]["physical_pipeline_wait_time_ms"] =
        backendStats.synchronizationTiming_.physicalPipelineWaitTimeMs_;
    showerMetadata["synchronization_timing"]["profile_input_waits"] =
        backendStats.synchronizationTiming_.profileInputWaits_;
    showerMetadata["synchronization_timing"]["profile_input_wait_time_ms"] =
        backendStats.synchronizationTiming_.profileInputWaitTimeMs_;
    auto scalarEmSteps = std::uint64_t{0};
    for (auto const& [pdg, count] : hybridTiming.scalarStepsByPdg_) {
      if (pdg == 22 || pdg == 11 || pdg == -11) { scalarEmSteps += count; }
    }
    auto ledgerInitialGeV = primaryTotalEnergy / 1_GeV;
#ifdef CORSIKA_KOKKOS_STATIC_MULTIGPU
    // A worker transports only its imported roots, not the global primary.
    // This is also zero for a radio-only worker with an empty frontier.
    if (auto const* source = frontier::activeFrontierInput) {
      if (source->pending())
        throw std::logic_error("Cannot report energy before the frontier is drained");
      ledgerInitialGeV = source->weightedEnergyGeV();
    }
#endif
    auto const ledgerSourceGeV = ledgerInitialGeV + routerStats.mediumRestMassInputGeV_ +
                                 routerStats.massConventionCorrectionGeV_;
    auto const ledgerTerminalGeV =
        routerStats.depositedEnergyGeV_ + routerStats.cutRestMassEnergyGeV_ +
        routerStats.observedTotalEnergyGeV_ + routerStats.escapedTotalEnergyGeV_ +
        routerStats.unwrittenPhotoelectricBindingEnergyGeV_ -
        routerStats.observationCutOverlapEnergyGeV_;
    auto const ledgerResidualGeV = ledgerSourceGeV - ledgerTerminalGeV;
    auto const ledgerRelativeError =
        std::abs(ledgerResidualGeV) /
        std::max(ledgerSourceGeV, std::numeric_limits<double>::min());
    auto const ledgerCompleteCoverage =
        is_em(beamCode) && emthinfrac <= 0. && scalarEmSteps == 0 &&
        routerStats.particlesReturnedForCpuFallback_ == 0 &&
        routerStats.particlesReturnedForCpuMemorySpill_ == 0 &&
        routerStats.specifiedCpuFinalStates_ == 0;
    auto ledger = showerMetadata["energy_ledger"];
    ledger["definition"] =
        "initial_total + medium_electron_rest + mass_convention_correction = "
        "deposit + cut_rest + "
        "observed_total + escaped_total + unwritten_photoelectric_binding - "
        "observation_cut_overlap";
    ledger["complete_coverage"] = ledgerCompleteCoverage;
    ledger["scalar_em_steps"] = scalarEmSteps;
    ledger["initial_total_GeV"] = ledgerInitialGeV;
    ledger["medium_rest_mass_input_GeV"] = routerStats.mediumRestMassInputGeV_;
    ledger["deposited_GeV"] = routerStats.depositedEnergyGeV_;
    ledger["cut_rest_mass_energy_GeV"] = routerStats.cutRestMassEnergyGeV_;
    ledger["observed_total_energy_GeV"] = routerStats.observedTotalEnergyGeV_;
    ledger["escaped_total_energy_GeV"] = routerStats.escapedTotalEnergyGeV_;
    ledger["unwritten_photoelectric_binding_energy_GeV"] =
        routerStats.unwrittenPhotoelectricBindingEnergyGeV_;
    ledger["observation_cut_overlap_energy_GeV"] =
        routerStats.observationCutOverlapEnergyGeV_;
    ledger["mass_convention_correction_GeV"] = routerStats.massConventionCorrectionGeV_;
    ledger["source_GeV"] = ledgerSourceGeV;
    ledger["terminal_GeV"] = ledgerTerminalGeV;
    ledger["residual_GeV"] = ledgerResidualGeV;
    ledger["relative_closure_error"] = ledgerRelativeError;
    ledger["acceptance_tolerance"] = 1.e-4;
    ledger["accepted"] = ledgerCompleteCoverage && ledgerRelativeError <= 1.e-4;
    CORSIKA_LOG_INFO(
        "CUDA EM energy ledger: complete={}, source={} GeV, "
        "terminal={} GeV, residual={} GeV, relative error={}",
        ledgerCompleteCoverage, ledgerSourceGeV, ledgerTerminalGeV, ledgerResidualGeV,
        ledgerRelativeError);
    if (ledgerCompleteCoverage && ledgerRelativeError > 1.e-4) {
      throw std::runtime_error("CUDA EM strict energy closure exceeded 1e-4");
    }
    if (cudaSession.getRunOutput()) {
      cudaSession.getRunOutput()->recordComplete(outputShowerId,
                                                 std::move(showerMetadata));
    }
  }
} // namespace corsika::kokkos::application
