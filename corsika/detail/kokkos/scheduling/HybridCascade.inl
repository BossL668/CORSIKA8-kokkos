/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      HybridCascade(Environment<medium_interface_type> const& env, TTracking& tracking,
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
  inline HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      HybridCascade(Environment<medium_interface_type> const& env, TTracking& tracking,
                    TProcessList& processes, TOutput& output, TStack& stack,
                    TEmRouter& em_router)
      : sequence_(processes)
      , output_(output)
      , stack_(stack)
      , emRouter_(&em_router)
      , scheduler_(stack)
      , stepper_(env, tracking, processes, stack) {
    static_assert(!std::is_same_v<TEmRouter, DisabledHybridEmRouter>,
                  "The disabled router cannot be supplied to HybridCascade");
    CORSIKA_LOG_INFO("HybridCascade initialized with an external EM wavefront router");
    CORSIKA_LOG_INFO("Tracking algorithm: {} (version {})", TTracking::getName(),
                     TTracking::getVersion());
    if constexpr (stack_view_type::has_event) {
      CORSIKA_LOG_INFO("Stack - with full cascade HISTORY.");
    }
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::run() {
    using clock_type = std::chrono::steady_clock;
    auto elapsedMilliseconds = [](auto const start, auto const stop) {
      return std::chrono::duration<double, std::milli>(stop - start).count();
    };
    timingStatistics_ = {};
    timingStatistics_.hadronicWorkClassifier_ = hadronicWorkClassifier_;
    timingStatistics_.hadronicProcessPoolEnabled_ = hadronicProcessPool_ != nullptr;
    if (!pendingHadronicInteractions_.empty() || scheduler_.hasSuspendedParticles()) {
      throw std::logic_error("HybridCascade started with pending hadronic interactions");
    }
    nextHadronicSequenceId_ = 0;
    stepper_.enableDetailedPhaseTiming(scalarDetailedPhaseTimingEnabled_);
    stepper_.resetDetailedPhaseTiming();
    auto const run_start = clock_type::now();
    auto phase_start = clock_type::now();
    output_.startOfShower();
    timingStatistics_.outputStartTimeMs_ +=
        elapsedMilliseconds(phase_start, clock_type::now());
    phase_start = clock_type::now();
    setNodes();
    timingStatistics_.setNodesTimeMs_ +=
        elapsedMilliseconds(phase_start, clock_type::now());

    auto routerPending = [this]() {
      if constexpr (std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
        return false;
      } else {
        return emRouter_ != nullptr && emRouter_->pending();
      }
    };

    auto hadronicPending = [this]() { return !pendingHadronicInteractions_.empty(); };

    auto doStackIfSafe = [this, &elapsedMilliseconds]() {
      if (scheduler_.hasSuspendedParticles()) { return; }
      auto const start = std::chrono::steady_clock::now();
      sequence_.doStack(stack_);
      timingStatistics_.doStackTimeMs_ +=
          elapsedMilliseconds(start, std::chrono::steady_clock::now());
    };

    auto routerReadyForScalarInterleave = [this]() {
      if constexpr (std::is_same_v<TEmRouter, DisabledHybridEmRouter> ||
                    !hybrid_detail::HasReadyForScalarInterleave<TEmRouter>::value) {
        return false;
      } else {
        return emRouter_ != nullptr && emRouter_->readyForScalarInterleave();
      }
    };

    auto advanceInterleavedRouter = [this, &elapsedMilliseconds,
                                     &routerReadyForScalarInterleave]() {
      if constexpr (!std::is_same_v<TEmRouter, DisabledHybridEmRouter> &&
                    hybrid_detail::HasReadyForScalarInterleave<TEmRouter>::value) {
        while (routerReadyForScalarInterleave()) {
          auto const start = std::chrono::steady_clock::now();
          auto const returned = emRouter_->advanceOneWavefrontAndReturn(stack_);
          timingStatistics_.routerAdvanceTimeMs_ +=
              elapsedMilliseconds(start, std::chrono::steady_clock::now());
          if (returned != 0) {
            auto const node_start = std::chrono::steady_clock::now();
            setNodes();
            timingStatistics_.setNodesTimeMs_ +=
                elapsedMilliseconds(node_start, std::chrono::steady_clock::now());
          }
        }
      }
    };

    while (!scheduler_.empty() || routerPending() || hadronicPending()) {
      phase_start = clock_type::now();
      sequence_.initCascadeEquations();
      timingStatistics_.initCascadeEquationsTimeMs_ +=
          elapsedMilliseconds(phase_start, clock_type::now());

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
              pendingForcedAction_ != PendingForcedAction::None;
          if (forced_scalar_action) {
            if (pendingForcedAction_ == PendingForcedAction::Interaction) {
              stepper_.forceInteraction();
              ++timingStatistics_.forcedPrimaryInteractions_;
            } else {
              stepper_.forceDecay();
              ++timingStatistics_.forcedPrimaryDecays_;
            }
            pendingForcedAction_ = PendingForcedAction::None;
          }

          if constexpr (!std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
            phase_start = clock_type::now();
            auto const route =
                !forced_scalar_action && emRouter_ != nullptr &&
                emRouter_->canRoute(scheduled.particle_, scheduled.stepId_);
            if (route) {
              CORSIKA_LOG_TRACE(
                  "============== hybrid GPU particle: count={}, pid={}, "
                  "history={}, parent={}, generation={}, step={}",
                  count_, scheduled.particle_.getPID(), scheduled.historyId_,
                  scheduled.parentHistoryId_, scheduled.generation_, scheduled.stepId_);
              emRouter_->stage(scheduled.particle_, scheduled.historyId_,
                               scheduled.parentHistoryId_, scheduled.generation_,
                               scheduled.stepId_);
              timingStatistics_.routeStageTimeMs_ +=
                  elapsedMilliseconds(phase_start, clock_type::now());
              scheduled.particle_.erase();
              scheduler_.completeParticleStep();
              doStackIfSafe();
              advanceInterleavedRouter();
              continue;
            }
            timingStatistics_.routeStageTimeMs_ +=
                elapsedMilliseconds(phase_start, clock_type::now());
          }

          bool force_router_decay = false;
          if constexpr (hybrid_detail::HasConsumeForcedDecay<TEmRouter>::value) {
            force_router_decay =
                emRouter_ != nullptr &&
                emRouter_->consumeForcedDecay(scheduled.historyId_, scheduled.stepId_);
          }
          if (force_router_decay) { stepper_.forceDecay(); }

          CORSIKA_LOG_TRACE(
              "============== hybrid scalar particle: count={}, pid={}, history={}, "
              "parent={}, generation={}, step={}, stack entries={}, stack deleted={}",
              count_, scheduled.particle_.getPID(), scheduled.historyId_,
              scheduled.parentHistoryId_, scheduled.generation_, scheduled.stepId_,
              stack_.getEntries(), stack_.getErased());

          auto const scalar_pdg =
              static_cast<std::int32_t>(get_PDG(scheduled.particle_.getPID()));
          auto const hadronic_work_key = classify_hadronic_work(
              scheduled.particle_.getPID(), scheduled.particle_.getKineticEnergy(),
              hadronicWorkClassifier_);
          ++timingStatistics_.scalarStepsByPdg_[scalar_pdg];
          phase_start = clock_type::now();
          ScalarCascadeAdvanceResult advance_result =
              ScalarCascadeAdvanceResult::Completed;
          std::optional<HadronicInteractionDeferralContext> deferral_context;
          if (hadronicProcessPool_ != nullptr) {
            HadronicRandomKey random_key{hadronicWorkerConfig_.seed_,
                                         hadronicWorkerConfig_.showerId_,
                                         scheduled.historyId_,
                                         scheduled.stepId_,
                                         1U,
                                         0U};
            deferral_context.emplace(random_key, nextHadronicSequenceId_);
            ScopedHadronicInteractionDeferral scope{*deferral_context};
            advance_result = stepper_.advance(scheduled.particle_);
          } else {
            advance_result = stepper_.advance(scheduled.particle_);
          }
          auto const scalar_step_time_ms =
              elapsedMilliseconds(phase_start, clock_type::now());
          timingStatistics_.scalarStepperTimeMs_ += scalar_step_time_ms;
          timingStatistics_.scalarStepperTimeByPdgMs_[scalar_pdg] += scalar_step_time_ms;
          if (hadronic_work_key) {
            timingStatistics_.hadronicWorkClasses_[*hadronic_work_key].record(
                scalar_step_time_ms);
          }

          if (advance_result == ScalarCascadeAdvanceResult::DeferredHadronicInteraction) {
            if (hadronicProcessPool_ == nullptr || !deferral_context ||
                !deferral_context->hasPrepared() || !hadronic_work_key) {
              throw std::logic_error(
                  "Scalar step reported a deferred hadronic interaction "
                  "without a complete worker request");
            }
            auto prepared = deferral_context->takePrepared();
            auto const sequence_id = nextHadronicSequenceId_++;
            if (prepared.request_.sequenceId_ != sequence_id) {
              throw std::logic_error(
                  "Deferred hadronic sequence ID changed during preparation");
            }
            PendingHadronicInteraction pending{sequence_id, scheduled.historyId_,
                                               scheduled.particle_, *hadronic_work_key,
                                               std::move(prepared.request_)};
            pendingHadronicInteractions_.enqueue(
                *hadronic_work_key, sequence_id,
                estimateHadronicInteractionCost(*hadronic_work_key), std::move(pending));
            ++timingStatistics_.hadronicInteractionsPrepared_;
            timingStatistics_.hadronicPrepareTimeMs_ += scalar_step_time_ms;
            scheduler_.completeParticleStep(true);

            if (auto const trigger = hadronicFlushTrigger()) {
              flushHadronicInteractions(*trigger);
              doStackIfSafe();
            }
          } else {
            if (deferral_context && deferral_context->hasPrepared()) {
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
            phase_start = clock_type::now();
            [[maybe_unused]] auto const returned =
                emRouter_->advanceOneWavefrontAndReturn(stack_);
            timingStatistics_.routerAdvanceTimeMs_ +=
                elapsedMilliseconds(phase_start, clock_type::now());
            CORSIKA_LOG_TRACE("Hybrid EM router returned {} particles to the CPU stack",
                              returned);
            if (!scheduler_.empty()) {
              phase_start = clock_type::now();
              setNodes();
              timingStatistics_.setNodesTimeMs_ +=
                  elapsedMilliseconds(phase_start, clock_type::now());
            }
          }
        }

        if (hadronicPending()) {
          auto trigger = hadronicFlushTrigger();
          if (!trigger && !routerPending() && scheduler_.empty()) {
            trigger = HadronicQueueFlushTrigger::Drain;
          }
          if (!trigger) { continue; }
          flushHadronicInteractions(*trigger);
          doStackIfSafe();
          if (!scheduler_.empty()) {
            phase_start = clock_type::now();
            setNodes();
            timingStatistics_.setNodesTimeMs_ +=
                elapsedMilliseconds(phase_start, clock_type::now());
          }
        }
      } while (!scheduler_.empty() || routerPending() || hadronicPending());

      phase_start = clock_type::now();
      sequence_.doCascadeEquations(stack_);
      timingStatistics_.cascadeEquationsTimeMs_ +=
          elapsedMilliseconds(phase_start, clock_type::now());
    }

    phase_start = clock_type::now();
    if constexpr (!std::is_same_v<TEmRouter, DisabledHybridEmRouter>) {
      if constexpr (hybrid_detail::HasEndOfShower<TEmRouter>::value) {
        if (emRouter_ != nullptr) { emRouter_->endOfShower(); }
      }
    }
    output_.endOfShower();
    timingStatistics_.outputEndTimeMs_ +=
        elapsedMilliseconds(phase_start, clock_type::now());
    timingStatistics_.totalRunTimeMs_ = elapsedMilliseconds(run_start, clock_type::now());
    timingStatistics_.scalarPhaseTimingByPdg_ = stepper_.detailedPhaseTiming();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::setNodes() {
    if constexpr (hybrid_detail::HasPreserveLogicalNodes<TEmRouter>::value)
      stepper_.setNodes(emRouter_ && emRouter_->preserveLogicalNodes());
    else
      stepper_.setNodes();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::forceInteraction() {
    if (pendingForcedAction_ == PendingForcedAction::Decay) {
      CORSIKA_LOG_ERROR("Cannot set forceInteraction when forceDecay is already set");
      throw std::runtime_error(
          "Cannot set forceInteraction when forceDecay is already set");
    }
    pendingForcedAction_ = PendingForcedAction::Interaction;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::forceDecay() {
    if (pendingForcedAction_ == PendingForcedAction::Interaction) {
      CORSIKA_LOG_ERROR("Cannot set forceDecay when forceInteraction is already set");
      throw std::runtime_error(
          "Cannot set forceDecay when forceInteraction is already set");
    }
    pendingForcedAction_ = PendingForcedAction::Decay;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      configureHadronicWorkClassification(HadronicWorkClassifierConfig const& config) {
    validate_hadronic_work_classifier_config(config);
    hadronicWorkClassifier_ = config;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      configureHadronicProcessPool(HadronicProcessPool& pool,
                                   HybridHadronicWorkerConfig const& config) {
    if (pool.workerCount() == 0) {
      throw std::invalid_argument(
          "Hybrid hadronic process pool must have at least one worker");
    }
    if (config.minimumPendingInteractions_ == 0 || config.maximumBatchItems_ == 0) {
      throw std::invalid_argument("Hybrid hadronic batch sizes must be positive");
    }
    if (!std::isfinite(config.targetBatchCostMs_) || config.targetBatchCostMs_ <= 0. ||
        !std::isfinite(config.initialInteractionCostMs_) ||
        config.initialInteractionCostMs_ <= 0.) {
      throw std::invalid_argument(
          "Hybrid hadronic cost parameters must be finite and positive");
    }
    hadronicProcessPool_ = &pool;
    hadronicWorkerConfig_ = config;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void
  HybridCascade<TTracking, TProcessList, TOutput, TStack,
                TEmRouter>::enableScalarDetailedPhaseTiming(bool const enabled) {
    scalarDetailedPhaseTimingEnabled_ = enabled;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline double HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      estimateHadronicInteractionCost(HadronicWorkKey const& key) const {
    // Never feed measured wall-clock timing into a fixed-seed scheduling
    // decision.  The measurements remain available in output metadata for
    // profiling, while batching uses only the physical work class.
    return deterministic_hadronic_interaction_cost(
        key, hadronicWorkerConfig_.initialInteractionCostMs_);
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline std::optional<HadronicQueueFlushTrigger> HybridCascade<
      TTracking, TProcessList, TOutput, TStack, TEmRouter>::hadronicFlushTrigger() const {
    if (pendingHadronicInteractions_.empty()) { return std::nullopt; }
    if (hadronicProcessPool_ == nullptr) {
      throw std::logic_error("Cannot evaluate a hadronic flush without a process pool");
    }
    return decide_hadronic_queue_flush(pendingHadronicInteractions_.size(),
                                       pendingHadronicInteractions_.pendingCost(),
                                       hadronicProcessPool_->workerCount(),
                                       hadronicWorkerConfig_.minimumPendingInteractions_,
                                       hadronicWorkerConfig_.targetBatchCostMs_,
                                       hadronicWorkerConfig_.maximumBatchItems_);
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline void HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      flushHadronicInteractions(HadronicQueueFlushTrigger const trigger) {
    if (pendingHadronicInteractions_.empty()) { return; }
    if (hadronicProcessPool_ == nullptr) {
      throw std::logic_error("Cannot flush hadronic interactions without a process pool");
    }
    using clock_type = std::chrono::steady_clock;
    auto const execute_start = clock_type::now();
    auto assignments = plan_hadronic_worker_assignments(
        pendingHadronicInteractions_, hadronicProcessPool_->workerCount(),
        hadronicWorkerConfig_.targetBatchCostMs_,
        hadronicWorkerConfig_.maximumBatchItems_);
    for (auto const& assignment : assignments) {
      timingStatistics_.hadronicWorkerClassifiedBatches_ += assignment.batches_.size();
    }
    auto super_batches = coalesce_hadronic_worker_assignments(assignments);
    HadronicFlushLoadRecord flush_load;
    flush_load.trigger_ = trigger;
    flush_load.predictedCostByWorker_.assign(hadronicProcessPool_->workerCount(), 0.);
    flush_load.actualFinalStateTimeByWorkerMs_.assign(hadronicProcessPool_->workerCount(),
                                                      0.);
    for (auto const& batch : super_batches) {
      if (batch.workerId_ >= flush_load.predictedCostByWorker_.size()) {
        throw std::logic_error("Hadronic super-batch worker is out of range");
      }
      flush_load.predictedCostByWorker_[batch.workerId_] = batch.estimatedCost_;
      flush_load.requests_ += batch.payloads_.size();
    }

    std::vector<HadronicRequestBatch> request_batches;
    std::vector<std::vector<PendingHadronicInteraction>> commit_payloads;
    request_batches.reserve(super_batches.size());
    commit_payloads.reserve(super_batches.size());
    std::uint64_t batch_id = 0;
    for (auto& batch : super_batches) {
      HadronicRequestBatch request_batch;
      request_batch.batchId_ = (hadronicWorkerConfig_.showerId_ << 32U) | batch_id++;
      request_batch.workerId_ = batch.workerId_;
      request_batch.requests_.reserve(batch.payloads_.size());
      std::vector<PendingHadronicInteraction> payloads;
      payloads.reserve(batch.payloads_.size());
      for (auto& pending : batch.payloads_) {
        request_batch.requests_.push_back(pending.request_);
        payloads.push_back(std::move(pending));
      }
      request_batches.push_back(std::move(request_batch));
      commit_payloads.push_back(std::move(payloads));
    }

    auto completed = hadronicProcessPool_->execute(request_batches);
    timingStatistics_.hadronicWorkerExecuteTimeMs_ +=
        std::chrono::duration<double, std::milli>(clock_type::now() - execute_start)
            .count();
    ++timingStatistics_.hadronicWorkerFlushes_;
    switch (trigger) {
      case HadronicQueueFlushTrigger::EstimatedCost:
        ++timingStatistics_.hadronicWorkerCostTriggeredFlushes_;
        break;
      case HadronicQueueFlushTrigger::Capacity:
        ++timingStatistics_.hadronicWorkerCapacityTriggeredFlushes_;
        break;
      case HadronicQueueFlushTrigger::Drain:
        ++timingStatistics_.hadronicWorkerDrainTriggeredFlushes_;
        break;
    }
    timingStatistics_.hadronicWorkerBatches_ += completed.size();

    struct CommitRecord {
      PendingHadronicInteraction pending_;
      HadronicInteractionResponse response_;
    };
    std::vector<CommitRecord> commit_records;
    for (std::size_t batch = 0; batch < completed.size(); ++batch) {
      if (completed[batch].responses_.size() != commit_payloads[batch].size()) {
        throw std::runtime_error(
            "Hadronic worker response count does not match parked projectiles");
      }
      ++timingStatistics_.hadronicWorkerBatchesByWorker_[completed[batch].workerId_];
      for (std::size_t request = 0; request < completed[batch].responses_.size();
           ++request) {
        flush_load.actualFinalStateTimeByWorkerMs_[completed[batch].workerId_] +=
            completed[batch].responses_[request].header_.finalStateTimeMs_;
        ++timingStatistics_.hadronicWorkerRequestsByWorker_[completed[batch].workerId_];
        timingStatistics_
            .hadronicWorkerFinalStateTimeByWorkerMs_[completed[batch].workerId_] +=
            completed[batch].responses_[request].header_.finalStateTimeMs_;
        commit_records.push_back(
            CommitRecord{std::move(commit_payloads[batch][request]),
                         std::move(completed[batch].responses_[request])});
      }
    }
    timingStatistics_.hadronicFlushLoadRecords_.push_back(std::move(flush_load));
    std::sort(commit_records.begin(), commit_records.end(),
              [](CommitRecord const& lhs, CommitRecord const& rhs) {
                return lhs.pending_.sequenceId_ < rhs.pending_.sequenceId_;
              });

    HadronicFlushFingerprint fingerprint;
    fingerprint.requestHash_ = 1469598103934665603ULL;
    fingerprint.responseHash_ = 1469598103934665603ULL;
    auto appendHash = [](std::uint64_t& state, void const* source,
                         std::size_t const size) {
      auto const* bytes = static_cast<unsigned char const*>(source);
      for (std::size_t i = 0; i < size; ++i) {
        state ^= bytes[i];
        state *= 1099511628211ULL;
      }
    };
    for (auto const& record : commit_records) {
      ++fingerprint.requests_;
      appendHash(fingerprint.requestHash_, &record.pending_.request_,
                 sizeof(record.pending_.request_));
      auto physical_header = record.response_.header_;
      physical_header.finalStateTimeMs_ = 0.;
      appendHash(fingerprint.responseHash_, &physical_header, sizeof(physical_header));
      if (!record.response_.secondaries_.empty()) {
        appendHash(fingerprint.responseHash_, record.response_.secondaries_.data(),
                   record.response_.secondaries_.size() *
                       sizeof(record.response_.secondaries_.front()));
      }
    }
    timingStatistics_.hadronicFlushFingerprints_.push_back(fingerprint);

    auto const commit_start = clock_type::now();
    for (auto& record : commit_records) {
      if (record.response_.header_.sequenceId_ != record.pending_.sequenceId_) {
        throw std::runtime_error(
            "Hadronic worker response sequence ID mismatch during commit");
      }
      if (!std::isfinite(record.response_.header_.finalStateTimeMs_) ||
          record.response_.header_.finalStateTimeMs_ < 0.) {
        throw std::runtime_error("Hadronic worker returned an invalid execution time");
      }
      auto particle = record.pending_.particle_;
      if (particle.isErased() || particle.getHistoryId() != record.pending_.historyId_) {
        throw std::runtime_error(
            "Parked hadronic projectile moved or was erased before commit");
      }
      stepper_.commitDeferredHadronicInteraction(particle, record.response_.secondaries_);
      scheduler_.resumeParticle(record.pending_.historyId_);
      timingStatistics_.hadronicWorkerFinalStateClasses_[record.pending_.workKey_].record(
          record.response_.header_.finalStateTimeMs_);
      ++timingStatistics_.hadronicInteractionsCommitted_;
      timingStatistics_.hadronicWorkerSecondaries_ +=
          record.response_.secondaries_.size();
    }
    timingStatistics_.hadronicCommitTimeMs_ +=
        std::chrono::duration<double, std::milli>(clock_type::now() - commit_start)
            .count();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline CpuParticleSchedulerStatistics const& HybridCascade<
      TTracking, TProcessList, TOutput, TStack, TEmRouter>::schedulerStatistics() const {
    return scheduler_.statistics();
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline HybridCascadeTimingStatistics const& HybridCascade<
      TTracking, TProcessList, TOutput, TStack, TEmRouter>::timingStatistics() const {
    return timingStatistics_;
  }

  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter>
  inline HybridCascade<TTracking, TProcessList, TOutput, TStack, TEmRouter>::
      PendingHadronicInteraction::PendingHadronicInteraction(
          std::uint64_t const sequence, transport::HistoryId const history,
          particle_type particle_iterator, HadronicWorkKey const key,
          HadronicInteractionRequest worker_request)
      : sequenceId_(sequence)
      , historyId_(history)
      , particle_(std::move(particle_iterator))
      , workKey_(key)
      , request_(std::move(worker_request)) {}

} // namespace corsika::kokkos
