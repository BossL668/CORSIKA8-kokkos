/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline HybridCascade<TTracking, TProcessList, TOutput, TStack,
                       TEmRouter>::HybridCascade(
      Environment<medium_interface_type> const& env, TTracking& tracking,
      TProcessList& processes, TOutput& output, TStack& stack)
      : sequence_(processes)
      , output_(output)
      , stack_(stack)
      , scheduler_(stack)
      , stepper_(env, tracking, processes, stack) {
    CORSIKA_LOG_INFO(
        "HybridCascade initialized with scalar-only wavefront compatibility scheduler");
    CORSIKA_LOG_INFO("Tracking algorithm: {} (version {})", TTracking::getName(),
                     TTracking::getVersion());
    if constexpr (stack_view_type::has_event) {
      CORSIKA_LOG_INFO("Stack - with full cascade HISTORY.");
    }
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline HybridCascade<TTracking, TProcessList, TOutput, TStack,
                       TEmRouter>::HybridCascade(
      Environment<medium_interface_type> const& env, TTracking& tracking,
      TProcessList& processes, TOutput& output, TStack& stack,
      TEmRouter& em_router)
      : sequence_(processes)
      , output_(output)
      , stack_(stack)
      , em_router_(&em_router)
      , scheduler_(stack)
      , stepper_(env, tracking, processes, stack) {
    static_assert(!std::is_same_v<TEmRouter, DisabledHybridEmRouter>,
                  "The disabled router cannot be supplied to HybridCascade");
    CORSIKA_LOG_INFO(
        "HybridCascade initialized with an external EM wavefront router");
    CORSIKA_LOG_INFO("Tracking algorithm: {} (version {})", TTracking::getName(),
                     TTracking::getVersion());
    if constexpr (stack_view_type::has_event) {
      CORSIKA_LOG_INFO("Stack - with full cascade HISTORY.");
    }
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::run() {
    using Clock = std::chrono::steady_clock;
    auto elapsedMilliseconds = [](auto const start, auto const stop) {
      return std::chrono::duration<double, std::milli>(
                 stop - start)
          .count();
    };
    timing_statistics_ = {};
    timing_statistics_.hadronic_work_classifier =
        hadronic_work_classifier_;
    timing_statistics_.hadronic_process_pool_enabled =
        hadronic_process_pool_ != nullptr;
    if (!pending_hadronic_interactions_.empty() ||
        scheduler_.hasSuspendedParticles()) {
      throw std::logic_error(
          "HybridCascade started with pending hadronic interactions");
    }
    next_hadronic_sequence_id_ = 0;
    stepper_.enableDetailedPhaseTiming(
        scalar_detailed_phase_timing_enabled_);
    stepper_.resetDetailedPhaseTiming();
    auto const run_start = Clock::now();
    auto phase_start = Clock::now();
    output_.startOfShower();
    timing_statistics_.output_start_time_ms +=
        elapsedMilliseconds(phase_start, Clock::now());
    phase_start = Clock::now();
    setNodes();
    timing_statistics_.set_nodes_time_ms +=
        elapsedMilliseconds(phase_start, Clock::now());

    auto routerPending = [this]() {
      if constexpr (std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
        return false;
      } else {
        return em_router_ != nullptr && em_router_->pending();
      }
    };

    auto hadronicPending = [this]() {
      return !pending_hadronic_interactions_.empty();
    };

    auto doStackIfSafe = [this, &elapsedMilliseconds]() {
      if (scheduler_.hasSuspendedParticles()) {
        return;
      }
      auto const start = std::chrono::steady_clock::now();
      sequence_.doStack(stack_);
      timing_statistics_.do_stack_time_ms +=
          elapsedMilliseconds(
              start, std::chrono::steady_clock::now());
    };

    auto routerReadyForScalarInterleave = [this]() {
      if constexpr (
          std::is_same_v<TEmRouter, DisabledHybridEmRouter> ||
          !hybrid_detail::
              HasReadyForScalarInterleave<TEmRouter>::value) {
        return false;
      } else {
        return em_router_ != nullptr &&
               em_router_->readyForScalarInterleave();
      }
    };

    auto advanceInterleavedRouter =
        [this, &elapsedMilliseconds,
         &routerReadyForScalarInterleave]() {
          if constexpr (
              !std::is_same_v<TEmRouter,
                              DisabledHybridEmRouter> &&
              hybrid_detail::
                  HasReadyForScalarInterleave<TEmRouter>::value) {
            while (routerReadyForScalarInterleave()) {
              auto const start =
                  std::chrono::steady_clock::now();
              auto const returned =
                  em_router_->advanceOneWavefrontAndReturn(
                      stack_);
              timing_statistics_.router_advance_time_ms +=
                  elapsedMilliseconds(
                      start,
                      std::chrono::steady_clock::now());
              if (returned != 0) {
                auto const node_start =
                    std::chrono::steady_clock::now();
                setNodes();
                timing_statistics_.set_nodes_time_ms +=
                    elapsedMilliseconds(
                        node_start,
                        std::chrono::steady_clock::now());
              }
            }
          }
        };

    while (!scheduler_.empty() || routerPending() ||
           hadronicPending()) {
      phase_start = Clock::now();
      sequence_.initCascadeEquations();
      timing_statistics_.init_cascade_equations_time_ms +=
          elapsedMilliseconds(phase_start, Clock::now());

      do {
        while (!scheduler_.empty()) {
          auto scheduled = scheduler_.acquireNext();
          ++count_;

          // forceInteraction()/forceDecay() apply to the next particle selected
          // by the scalar scheduler.  Resolve that one-shot control before CUDA
          // routing so an immediately routable gamma/e/mu primary cannot bypass
          // the requested scalar vertex.  Its secondaries remain eligible for
          // normal GPU routing on subsequent scheduler iterations.
          auto const forced_scalar_action =
              pending_forced_action_ != PendingForcedAction::None;
          if (forced_scalar_action) {
            if (pending_forced_action_ ==
                PendingForcedAction::Interaction) {
              stepper_.forceInteraction();
              ++timing_statistics_.forced_primary_interactions;
            } else {
              stepper_.forceDecay();
              ++timing_statistics_.forced_primary_decays;
            }
            pending_forced_action_ = PendingForcedAction::None;
          }

          if constexpr (!std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
            phase_start = Clock::now();
            auto const route =
                !forced_scalar_action &&
                em_router_ != nullptr &&
                em_router_->canRoute(
                    scheduled.particle, scheduled.step_id);
            if (route) {
              CORSIKA_LOG_TRACE(
                  "============== hybrid GPU particle: count={}, pid={}, "
                  "history={}, parent={}, generation={}, step={}",
                  count_, scheduled.particle.getPID(), scheduled.history_id,
                  scheduled.parent_history_id, scheduled.generation,
                  scheduled.step_id);
              em_router_->stage(
                  scheduled.particle, scheduled.history_id,
                  scheduled.parent_history_id, scheduled.generation,
                  scheduled.step_id);
              timing_statistics_.route_stage_time_ms +=
                  elapsedMilliseconds(
                      phase_start, Clock::now());
              scheduled.particle.erase();
              scheduler_.completeParticleStep();
              doStackIfSafe();
              advanceInterleavedRouter();
              continue;
            }
            timing_statistics_.route_stage_time_ms +=
                elapsedMilliseconds(
                    phase_start, Clock::now());
          }

          bool force_router_decay = false;
          if constexpr (
              hybrid_detail::
                  HasConsumeForcedDecay<TEmRouter>::value) {
            force_router_decay =
                em_router_ != nullptr &&
                em_router_->consumeForcedDecay(
                    scheduled.history_id,
                    scheduled.step_id);
          }
          if (force_router_decay) {
            stepper_.forceDecay();
          }

          CORSIKA_LOG_TRACE(
              "============== hybrid scalar particle: count={}, pid={}, history={}, "
              "parent={}, generation={}, step={}, stack entries={}, stack deleted={}",
              count_, scheduled.particle.getPID(), scheduled.history_id,
              scheduled.parent_history_id, scheduled.generation,
              scheduled.step_id, stack_.getEntries(), stack_.getErased());

          auto const scalar_pdg =
              static_cast<std::int32_t>(
                  get_PDG(
                      scheduled.particle.getPID()));
          auto const hadronic_work_key =
              classifyHadronicWork(
                  scheduled.particle.getPID(),
                  scheduled.particle.getKineticEnergy(),
                  hadronic_work_classifier_);
          ++timing_statistics_
                .scalar_steps_by_pdg[scalar_pdg];
          phase_start = Clock::now();
          ScalarCascadeAdvanceResult advance_result =
              ScalarCascadeAdvanceResult::Completed;
          std::optional<
              HadronicInteractionDeferralContext>
              deferral_context;
          if (hadronic_process_pool_ != nullptr) {
            HadronicRandomKey random_key{
                hadronic_worker_config_.seed,
                hadronic_worker_config_.shower_id,
                scheduled.history_id,
                scheduled.step_id,
                1U, 0U};
            deferral_context.emplace(
                random_key,
                next_hadronic_sequence_id_);
            ScopedHadronicInteractionDeferral scope{
                *deferral_context};
            advance_result =
                stepper_.advance(scheduled.particle);
          } else {
            advance_result =
                stepper_.advance(scheduled.particle);
          }
          auto const scalar_step_time_ms =
              elapsedMilliseconds(phase_start, Clock::now());
          timing_statistics_.scalar_stepper_time_ms +=
              scalar_step_time_ms;
          timing_statistics_
              .scalar_stepper_time_by_pdg_ms[scalar_pdg] +=
              scalar_step_time_ms;
          if (hadronic_work_key) {
            timing_statistics_
                .hadronic_work_classes[*hadronic_work_key]
                .record(scalar_step_time_ms);
          }

          if (advance_result ==
              ScalarCascadeAdvanceResult::
                  DeferredHadronicInteraction) {
            if (hadronic_process_pool_ == nullptr ||
                !deferral_context ||
                !deferral_context->hasPrepared() ||
                !hadronic_work_key) {
              throw std::logic_error(
                  "Scalar step reported a deferred hadronic interaction "
                  "without a complete worker request");
            }
            auto prepared =
                deferral_context->takePrepared();
            auto const sequence_id =
                next_hadronic_sequence_id_++;
            if (prepared.request.sequence_id !=
                sequence_id) {
              throw std::logic_error(
                  "Deferred hadronic sequence ID changed during preparation");
            }
            PendingHadronicInteraction pending{
                sequence_id, scheduled.history_id,
                scheduled.particle,
                *hadronic_work_key,
                std::move(prepared.request)};
            pending_hadronic_interactions_.enqueue(
                *hadronic_work_key, sequence_id,
                estimateHadronicInteractionCost(
                    *hadronic_work_key),
                std::move(pending));
            ++timing_statistics_
                  .hadronic_interactions_prepared;
            timing_statistics_.hadronic_prepare_time_ms +=
                scalar_step_time_ms;
            scheduler_.completeParticleStep(true);

            if (auto const trigger =
                    hadronicFlushTrigger()) {
              flushHadronicInteractions(*trigger);
              doStackIfSafe();
            }
          } else {
            if (deferral_context &&
                deferral_context->hasPrepared()) {
              throw std::logic_error(
                  "Hadronic interaction was prepared but the scalar "
                  "step did not report deferral");
            }
            scheduler_.completeParticleStep();
            doStackIfSafe();
          }
        }

        if constexpr (!std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
          if (routerPending()) {
            phase_start = Clock::now();
            [[maybe_unused]] auto const returned =
                em_router_->advanceOneWavefrontAndReturn(stack_);
            timing_statistics_.router_advance_time_ms +=
                elapsedMilliseconds(
                    phase_start, Clock::now());
            CORSIKA_LOG_TRACE(
                "Hybrid EM router returned {} particles to the CPU stack",
                returned);
            if (!scheduler_.empty()) {
              phase_start = Clock::now();
              setNodes();
              timing_statistics_.set_nodes_time_ms +=
                  elapsedMilliseconds(
                      phase_start, Clock::now());
            }
          }
        }

        if (hadronicPending()) {
          auto trigger = hadronicFlushTrigger();
          if (!trigger && !routerPending() &&
              scheduler_.empty()) {
            trigger =
                HadronicQueueFlushTrigger::Drain;
          }
          if (!trigger) {
            continue;
          }
          flushHadronicInteractions(*trigger);
          doStackIfSafe();
          if (!scheduler_.empty()) {
            phase_start = Clock::now();
            setNodes();
            timing_statistics_.set_nodes_time_ms +=
                elapsedMilliseconds(
                    phase_start, Clock::now());
          }
        }
      } while (!scheduler_.empty() || routerPending() ||
               hadronicPending());

      phase_start = Clock::now();
      sequence_.doCascadeEquations(stack_);
      timing_statistics_.cascade_equations_time_ms +=
          elapsedMilliseconds(phase_start, Clock::now());
    }

    phase_start = Clock::now();
    if constexpr (!std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
      if constexpr (hybrid_detail::HasEndOfShower<TEmRouter>::value) {
        if (em_router_ != nullptr) {
          em_router_->endOfShower();
        }
      }
    }
    output_.endOfShower();
    timing_statistics_.output_end_time_ms +=
        elapsedMilliseconds(phase_start, Clock::now());
    timing_statistics_.total_run_time_ms =
        elapsedMilliseconds(run_start, Clock::now());
    timing_statistics_.scalar_phase_timing_by_pdg =
        stepper_.detailedPhaseTiming();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::setNodes() {
    stepper_.setNodes();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack,
                TEmRouter>::forceInteraction() {
    if (pending_forced_action_ ==
        PendingForcedAction::Decay) {
      CORSIKA_LOG_ERROR(
          "Cannot set forceInteraction when forceDecay is already set");
      throw std::runtime_error(
          "Cannot set forceInteraction when forceDecay is already set");
    }
    pending_forced_action_ = PendingForcedAction::Interaction;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::forceDecay() {
    if (pending_forced_action_ ==
        PendingForcedAction::Interaction) {
      CORSIKA_LOG_ERROR(
          "Cannot set forceDecay when forceInteraction is already set");
      throw std::runtime_error(
          "Cannot set forceDecay when forceInteraction is already set");
    }
    pending_forced_action_ = PendingForcedAction::Decay;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack,
                            TEmRouter>::configureHadronicWorkClassification(
      HadronicWorkClassifierConfig const& config) {
    validateHadronicWorkClassifierConfig(config);
    hadronic_work_classifier_ = config;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack,
                            TEmRouter>::configureHadronicProcessPool(
      HadronicProcessPool& pool,
      HybridHadronicWorkerConfig const& config) {
    if (pool.workerCount() == 0) {
      throw std::invalid_argument(
          "Hybrid hadronic process pool must have at least one worker");
    }
    if (config.minimum_pending_interactions == 0 ||
        config.maximum_batch_items == 0) {
      throw std::invalid_argument(
          "Hybrid hadronic batch sizes must be positive");
    }
    if (!std::isfinite(config.target_batch_cost_ms) ||
        config.target_batch_cost_ms <= 0. ||
        !std::isfinite(config.initial_interaction_cost_ms) ||
        config.initial_interaction_cost_ms <= 0.) {
      throw std::invalid_argument(
          "Hybrid hadronic cost parameters must be finite and positive");
    }
    hadronic_process_pool_ = &pool;
    hadronic_worker_config_ = config;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack,
                            TEmRouter>::enableScalarDetailedPhaseTiming(
      bool const enabled) {
    scalar_detailed_phase_timing_enabled_ = enabled;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline double HybridCascade<TTracking, TProcessList, TOutput, TStack,
                              TEmRouter>::estimateHadronicInteractionCost(
      HadronicWorkKey const& key) const {
    // Never feed measured wall-clock timing into a fixed-seed scheduling
    // decision.  The measurements remain available in output metadata for
    // profiling, while batching uses only the physical work class.
    return deterministicHadronicInteractionCost(
        key,
        hadronic_worker_config_
            .initial_interaction_cost_ms);
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline std::optional<HadronicQueueFlushTrigger>
  HybridCascade<TTracking, TProcessList, TOutput, TStack,
                TEmRouter>::hadronicFlushTrigger() const {
    if (pending_hadronic_interactions_.empty()) {
      return std::nullopt;
    }
    if (hadronic_process_pool_ == nullptr) {
      throw std::logic_error(
          "Cannot evaluate a hadronic flush without a process pool");
    }
    return decideHadronicQueueFlush(
        pending_hadronic_interactions_.size(),
        pending_hadronic_interactions_.pendingCost(),
        hadronic_process_pool_->workerCount(),
        hadronic_worker_config_
            .minimum_pending_interactions,
        hadronic_worker_config_
            .target_batch_cost_ms,
        hadronic_worker_config_
            .maximum_batch_items);
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack,
                            TEmRouter>::flushHadronicInteractions(
      HadronicQueueFlushTrigger const trigger) {
    if (pending_hadronic_interactions_.empty()) {
      return;
    }
    if (hadronic_process_pool_ == nullptr) {
      throw std::logic_error(
          "Cannot flush hadronic interactions without a process pool");
    }
    using Clock = std::chrono::steady_clock;
    auto const execute_start = Clock::now();
    auto assignments =
        planHadronicWorkerAssignments(
            pending_hadronic_interactions_,
            hadronic_process_pool_->workerCount(),
            hadronic_worker_config_
                .target_batch_cost_ms,
            hadronic_worker_config_
                .maximum_batch_items);
    for (auto const& assignment : assignments) {
      timing_statistics_
          .hadronic_worker_classified_batches +=
          assignment.batches.size();
    }
    auto super_batches =
        coalesceHadronicWorkerAssignments(assignments);
    HadronicFlushLoadRecord flush_load;
    flush_load.trigger = trigger;
    flush_load.predicted_cost_by_worker.assign(
        hadronic_process_pool_->workerCount(), 0.);
    flush_load
        .actual_final_state_time_by_worker_ms.assign(
            hadronic_process_pool_->workerCount(), 0.);
    for (auto const& batch : super_batches) {
      if (batch.worker_id >=
          flush_load.predicted_cost_by_worker.size()) {
        throw std::logic_error(
            "Hadronic super-batch worker is out of range");
      }
      flush_load.predicted_cost_by_worker[batch.worker_id] =
          batch.estimated_cost;
      flush_load.requests += batch.payloads.size();
    }

    std::vector<HadronicRequestBatch>
        request_batches;
    std::vector<
        std::vector<PendingHadronicInteraction>>
        commit_payloads;
    request_batches.reserve(super_batches.size());
    commit_payloads.reserve(super_batches.size());
    std::uint64_t batch_id = 0;
    for (auto& batch : super_batches) {
      HadronicRequestBatch request_batch;
      request_batch.batch_id =
          (hadronic_worker_config_.shower_id << 32U) |
          batch_id++;
      request_batch.worker_id =
          batch.worker_id;
      request_batch.requests.reserve(
          batch.payloads.size());
      std::vector<PendingHadronicInteraction>
          payloads;
      payloads.reserve(batch.payloads.size());
      for (auto& pending : batch.payloads) {
        request_batch.requests.push_back(
            pending.request);
        payloads.push_back(std::move(pending));
      }
      request_batches.push_back(
          std::move(request_batch));
      commit_payloads.push_back(
          std::move(payloads));
    }

    auto completed =
        hadronic_process_pool_->execute(
            request_batches);
    timing_statistics_.hadronic_worker_execute_time_ms +=
        std::chrono::duration<double, std::milli>(
            Clock::now() - execute_start)
            .count();
    ++timing_statistics_.hadronic_worker_flushes;
    switch (trigger) {
      case HadronicQueueFlushTrigger::EstimatedCost:
        ++timing_statistics_
              .hadronic_worker_cost_triggered_flushes;
        break;
      case HadronicQueueFlushTrigger::Capacity:
        ++timing_statistics_
              .hadronic_worker_capacity_triggered_flushes;
        break;
      case HadronicQueueFlushTrigger::Drain:
        ++timing_statistics_
              .hadronic_worker_drain_triggered_flushes;
        break;
    }
    timing_statistics_.hadronic_worker_batches +=
        completed.size();

    struct CommitRecord {
      PendingHadronicInteraction pending;
      HadronicInteractionResponse response;
    };
    std::vector<CommitRecord> commit_records;
    for (std::size_t batch = 0;
         batch < completed.size(); ++batch) {
      if (completed[batch].responses.size() !=
          commit_payloads[batch].size()) {
        throw std::runtime_error(
            "Hadronic worker response count does not match parked projectiles");
      }
      ++timing_statistics_
            .hadronic_worker_batches_by_worker
                [completed[batch].worker_id];
      for (std::size_t request = 0;
           request < completed[batch].responses.size();
           ++request) {
        flush_load
            .actual_final_state_time_by_worker_ms
                [completed[batch].worker_id] +=
            completed[batch]
                .responses[request]
                .header.final_state_time_ms;
        ++timing_statistics_
              .hadronic_worker_requests_by_worker
                  [completed[batch].worker_id];
        timing_statistics_
            .hadronic_worker_final_state_time_by_worker_ms
                [completed[batch].worker_id] +=
            completed[batch]
                .responses[request]
                .header.final_state_time_ms;
        commit_records.push_back(
            CommitRecord{
                std::move(
                    commit_payloads[batch][request]),
                std::move(
                    completed[batch]
                        .responses[request])});
      }
    }
    timing_statistics_
        .hadronic_flush_load_records.push_back(
            std::move(flush_load));
    std::sort(
        commit_records.begin(),
        commit_records.end(),
        [](CommitRecord const& lhs,
           CommitRecord const& rhs) {
          return lhs.pending.sequence_id <
                 rhs.pending.sequence_id;
        });

    HadronicFlushFingerprint fingerprint;
    fingerprint.request_hash =
        1469598103934665603ULL;
    fingerprint.response_hash =
        1469598103934665603ULL;
    auto appendHash = [](
                          std::uint64_t& state,
                          void const* source,
                          std::size_t const size) {
      auto const* bytes =
          static_cast<unsigned char const*>(source);
      for (std::size_t i = 0; i < size; ++i) {
        state ^= bytes[i];
        state *= 1099511628211ULL;
      }
    };
    for (auto const& record : commit_records) {
      ++fingerprint.requests;
      appendHash(
          fingerprint.request_hash,
          &record.pending.request,
          sizeof(record.pending.request));
      auto physical_header =
          record.response.header;
      physical_header.final_state_time_ms = 0.;
      appendHash(
          fingerprint.response_hash,
          &physical_header,
          sizeof(physical_header));
      if (!record.response.secondaries.empty()) {
        appendHash(
            fingerprint.response_hash,
            record.response.secondaries.data(),
            record.response.secondaries.size() *
                sizeof(record.response.secondaries.front()));
      }
    }
    timing_statistics_
        .hadronic_flush_fingerprints.push_back(
            fingerprint);

    auto const commit_start = Clock::now();
    for (auto& record : commit_records) {
      if (record.response.header.sequence_id !=
          record.pending.sequence_id) {
        throw std::runtime_error(
            "Hadronic worker response sequence ID mismatch during commit");
      }
      if (!std::isfinite(
              record.response.header
                  .final_state_time_ms) ||
          record.response.header
                  .final_state_time_ms <
              0.) {
        throw std::runtime_error(
            "Hadronic worker returned an invalid execution time");
      }
      auto particle =
          record.pending.particle;
      if (particle.isErased() ||
          particle.getHistoryId() !=
              record.pending.history_id) {
        throw std::runtime_error(
            "Parked hadronic projectile moved or was erased before commit");
      }
      stepper_.commitDeferredHadronicInteraction(
          particle, record.response.secondaries);
      scheduler_.resumeParticle(
          record.pending.history_id);
      timing_statistics_
          .hadronic_worker_final_state_classes
              [record.pending.work_key]
          .record(
              record.response.header
                  .final_state_time_ms);
      ++timing_statistics_
            .hadronic_interactions_committed;
      timing_statistics_.hadronic_worker_secondaries +=
          record.response.secondaries.size();
    }
    timing_statistics_.hadronic_commit_time_ms +=
        std::chrono::duration<double, std::milli>(
            Clock::now() - commit_start)
            .count();
  }

} // namespace corsika
