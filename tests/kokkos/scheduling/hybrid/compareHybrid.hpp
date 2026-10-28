/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
// Generated comparison accessors for the two public interfaces.
#pragma once
#include "HybridFixture.hpp"
namespace hybrid_fixture {
  inline void snapshot(cp::HybridCascadeTimingStatistics const& s, Result& result) {
    result.counts_.push_back(s.forcedPrimaryInteractions_);
    result.counts_.push_back(s.forcedPrimaryDecays_);
    result.counts_.push_back(s.hadronicInteractionsPrepared_);
    result.counts_.push_back(s.hadronicInteractionsCommitted_);
    result.counts_.push_back(s.hadronicWorkerFlushes_);
    result.counts_.push_back(s.hadronicWorkerCostTriggeredFlushes_);
    result.counts_.push_back(s.hadronicWorkerCapacityTriggeredFlushes_);
    result.counts_.push_back(s.hadronicWorkerDrainTriggeredFlushes_);
    result.counts_.push_back(s.hadronicWorkerClassifiedBatches_);
    result.counts_.push_back(s.hadronicWorkerBatches_);
    result.counts_.push_back(s.hadronicWorkerSecondaries_);
    CHECK(std::isfinite(s.totalRunTimeMs_));
    CHECK(s.totalRunTimeMs_ >= 0.);
    CHECK(std::isfinite(s.outputStartTimeMs_));
    CHECK(s.outputStartTimeMs_ >= 0.);
    CHECK(std::isfinite(s.setNodesTimeMs_));
    CHECK(s.setNodesTimeMs_ >= 0.);
    CHECK(std::isfinite(s.routeStageTimeMs_));
    CHECK(s.routeStageTimeMs_ >= 0.);
    CHECK(std::isfinite(s.scalarStepperTimeMs_));
    CHECK(s.scalarStepperTimeMs_ >= 0.);
    CHECK(std::isfinite(s.doStackTimeMs_));
    CHECK(s.doStackTimeMs_ >= 0.);
    CHECK(std::isfinite(s.routerAdvanceTimeMs_));
    CHECK(s.routerAdvanceTimeMs_ >= 0.);
    CHECK(std::isfinite(s.initCascadeEquationsTimeMs_));
    CHECK(s.initCascadeEquationsTimeMs_ >= 0.);
    CHECK(std::isfinite(s.cascadeEquationsTimeMs_));
    CHECK(s.cascadeEquationsTimeMs_ >= 0.);
    CHECK(std::isfinite(s.outputEndTimeMs_));
    CHECK(s.outputEndTimeMs_ >= 0.);
    CHECK(std::isfinite(s.hadronicPrepareTimeMs_));
    CHECK(s.hadronicPrepareTimeMs_ >= 0.);
    CHECK(std::isfinite(s.hadronicWorkerExecuteTimeMs_));
    CHECK(s.hadronicWorkerExecuteTimeMs_ >= 0.);
    CHECK(std::isfinite(s.hadronicCommitTimeMs_));
    CHECK(s.hadronicCommitTimeMs_ >= 0.);
    result.scalarCounts_ = s.scalarStepsByPdg_;
    result.counts_.push_back(s.hadronicProcessPoolEnabled_);
    for (auto const& f : s.hadronicFlushFingerprints_)
      result.fingerprints_.push_back({f.requests_, f.requestHash_, f.responseHash_});
    for (auto const& load : s.hadronicFlushLoadRecords_) {
      result.loads_.push_back(
          {static_cast<double>(load.trigger_), static_cast<double>(load.requests_)});
      result.loads_.push_back(load.predictedCostByWorker_);
      result.loads_.push_back(load.actualFinalStateTimeByWorkerMs_);
    }
    for (auto const& item : s.hadronicWorkerBatchesByWorker_) {
      result.workerBatches_.push_back(item.first);
      result.workerBatches_.push_back(item.second);
    }
    for (auto const& item : s.hadronicWorkerRequestsByWorker_) {
      result.workerRequests_.push_back(item.first);
      result.workerRequests_.push_back(item.second);
    }
    for (auto const& item : s.hadronicWorkClasses_) {
      auto const& key = item.first;
      result.classes_.push_back(
          {static_cast<double>(key.model_), static_cast<double>(key.species_),
           static_cast<double>(key.energyBin_), static_cast<double>(key.massNumberBin_),
           static_cast<double>(item.second.steps_)});
    }
    for (auto const& item : s.hadronicWorkerFinalStateClasses_) {
      auto const& key = item.first;
      result.classes_.push_back(
          {static_cast<double>(key.model_), static_cast<double>(key.species_),
           static_cast<double>(key.energyBin_), static_cast<double>(key.massNumberBin_),
           static_cast<double>(item.second.steps_)});
    }
  }
  inline void snapshot(cp::CpuParticleSchedulerStatistics const& s, Result& result) {
    result.counts_.push_back(s.acquiredParticleSteps_);
    result.counts_.push_back(s.completedParticleSteps_);
    result.counts_.push_back(s.maxWavefrontSize_);
    result.counts_.push_back(s.suspendedParticles_);
    result.counts_.push_back(s.resumedParticles_);
    result.counts_.push_back(s.maximumSuspendedParticles_);
  }
  inline void snapshot(old::HybridCascadeTimingStatistics const& s, Result& result) {
    result.counts_.push_back(s.forced_primary_interactions);
    result.counts_.push_back(s.forced_primary_decays);
    result.counts_.push_back(s.hadronic_interactions_prepared);
    result.counts_.push_back(s.hadronic_interactions_committed);
    result.counts_.push_back(s.hadronic_worker_flushes);
    result.counts_.push_back(s.hadronic_worker_cost_triggered_flushes);
    result.counts_.push_back(s.hadronic_worker_capacity_triggered_flushes);
    result.counts_.push_back(s.hadronic_worker_drain_triggered_flushes);
    result.counts_.push_back(s.hadronic_worker_classified_batches);
    result.counts_.push_back(s.hadronic_worker_batches);
    result.counts_.push_back(s.hadronic_worker_secondaries);
    CHECK(std::isfinite(s.total_run_time_ms));
    CHECK(s.total_run_time_ms >= 0.);
    CHECK(std::isfinite(s.output_start_time_ms));
    CHECK(s.output_start_time_ms >= 0.);
    CHECK(std::isfinite(s.set_nodes_time_ms));
    CHECK(s.set_nodes_time_ms >= 0.);
    CHECK(std::isfinite(s.route_stage_time_ms));
    CHECK(s.route_stage_time_ms >= 0.);
    CHECK(std::isfinite(s.scalar_stepper_time_ms));
    CHECK(s.scalar_stepper_time_ms >= 0.);
    CHECK(std::isfinite(s.do_stack_time_ms));
    CHECK(s.do_stack_time_ms >= 0.);
    CHECK(std::isfinite(s.router_advance_time_ms));
    CHECK(s.router_advance_time_ms >= 0.);
    CHECK(std::isfinite(s.init_cascade_equations_time_ms));
    CHECK(s.init_cascade_equations_time_ms >= 0.);
    CHECK(std::isfinite(s.cascade_equations_time_ms));
    CHECK(s.cascade_equations_time_ms >= 0.);
    CHECK(std::isfinite(s.output_end_time_ms));
    CHECK(s.output_end_time_ms >= 0.);
    CHECK(std::isfinite(s.hadronic_prepare_time_ms));
    CHECK(s.hadronic_prepare_time_ms >= 0.);
    CHECK(std::isfinite(s.hadronic_worker_execute_time_ms));
    CHECK(s.hadronic_worker_execute_time_ms >= 0.);
    CHECK(std::isfinite(s.hadronic_commit_time_ms));
    CHECK(s.hadronic_commit_time_ms >= 0.);
    result.scalarCounts_ = s.scalar_steps_by_pdg;
    result.counts_.push_back(s.hadronic_process_pool_enabled);
    for (auto const& f : s.hadronic_flush_fingerprints)
      result.fingerprints_.push_back({f.requests, f.request_hash, f.response_hash});
    for (auto const& load : s.hadronic_flush_load_records) {
      result.loads_.push_back(
          {static_cast<double>(load.trigger), static_cast<double>(load.requests)});
      result.loads_.push_back(load.predicted_cost_by_worker);
      result.loads_.push_back(load.actual_final_state_time_by_worker_ms);
    }
    for (auto const& item : s.hadronic_worker_batches_by_worker) {
      result.workerBatches_.push_back(item.first);
      result.workerBatches_.push_back(item.second);
    }
    for (auto const& item : s.hadronic_worker_requests_by_worker) {
      result.workerRequests_.push_back(item.first);
      result.workerRequests_.push_back(item.second);
    }
    for (auto const& item : s.hadronic_work_classes) {
      auto const& key = item.first;
      result.classes_.push_back(
          {static_cast<double>(key.model), static_cast<double>(key.species),
           static_cast<double>(key.energy_bin), static_cast<double>(key.mass_number_bin),
           static_cast<double>(item.second.steps)});
    }
    for (auto const& item : s.hadronic_worker_final_state_classes) {
      auto const& key = item.first;
      result.classes_.push_back(
          {static_cast<double>(key.model), static_cast<double>(key.species),
           static_cast<double>(key.energy_bin), static_cast<double>(key.mass_number_bin),
           static_cast<double>(item.second.steps)});
    }
  }
  inline void snapshot(old::CpuOnlyWavefrontStatistics const& s, Result& result) {
    result.counts_.push_back(s.acquired_particle_steps);
    result.counts_.push_back(s.completed_particle_steps);
    result.counts_.push_back(s.max_wavefront_size);
    result.counts_.push_back(s.suspended_particles);
    result.counts_.push_back(s.resumed_particles);
    result.counts_.push_back(s.maximum_suspended_particles);
  }
} // namespace hybrid_fixture
