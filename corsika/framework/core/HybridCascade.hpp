/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/corsika.hpp>

#include <corsika/framework/core/CpuOnlyWavefrontScheduler.hpp>
#include <corsika/framework/core/HadronicInteractionDeferral.hpp>
#include <corsika/framework/core/HadronicProcessPool.hpp>
#include <corsika/framework/core/HadronicWorkQueue.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/ScalarCascadeStepper.hpp>

#include <corsika/media/Environment.hpp>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika {

  namespace hybrid_detail {
    template <typename T, typename = void>
    struct HasEndOfShower : std::false_type {};

    template <typename T>
    struct HasEndOfShower<
        T, std::void_t<decltype(std::declval<T&>().endOfShower())>>
        : std::true_type {};

    template <typename T, typename = void>
    struct HasConsumeForcedDecay : std::false_type {};

    template <typename T>
    struct HasConsumeForcedDecay<
        T,
        std::void_t<decltype(
            std::declval<T&>().consumeForcedDecay(
                std::declval<transport::HistoryId>(),
                std::declval<transport::StepId>()))>>
        : std::true_type {};
  } // namespace hybrid_detail

  struct DisabledHybridEmRouter {};

  struct HadronicFlushFingerprint {
    std::uint64_t requests{};
    std::uint64_t request_hash{};
    std::uint64_t response_hash{};
  };

  struct HadronicFlushLoadRecord {
    HadronicQueueFlushTrigger trigger{
        HadronicQueueFlushTrigger::Drain};
    std::uint64_t requests{};
    std::vector<double> predicted_cost_by_worker;
    std::vector<double>
        actual_final_state_time_by_worker_ms;
  };

  struct HybridCascadeTimingStatistics {
    double total_run_time_ms{};
    double output_start_time_ms{};
    double set_nodes_time_ms{};
    double route_stage_time_ms{};
    double scalar_stepper_time_ms{};
    double do_stack_time_ms{};
    double router_advance_time_ms{};
    double init_cascade_equations_time_ms{};
    double cascade_equations_time_ms{};
    double output_end_time_ms{};
    std::map<std::int32_t, std::uint64_t>
        scalar_steps_by_pdg{};
    std::map<std::int32_t, double>
        scalar_stepper_time_by_pdg_ms{};
    std::map<std::int32_t,
             ScalarCascadePhaseTimingStatistics>
        scalar_phase_timing_by_pdg{};
    HadronicWorkClassifierConfig hadronic_work_classifier{};
    std::map<HadronicWorkKey, HadronicWorkClassStatistics>
        hadronic_work_classes{};
    bool hadronic_process_pool_enabled{};
    std::uint64_t hadronic_interactions_prepared{};
    std::uint64_t hadronic_interactions_committed{};
    std::uint64_t hadronic_worker_flushes{};
    std::uint64_t hadronic_worker_cost_triggered_flushes{};
    std::uint64_t hadronic_worker_capacity_triggered_flushes{};
    std::uint64_t hadronic_worker_drain_triggered_flushes{};
    std::uint64_t hadronic_worker_classified_batches{};
    std::uint64_t hadronic_worker_batches{};
    std::uint64_t hadronic_worker_secondaries{};
    double hadronic_prepare_time_ms{};
    double hadronic_worker_execute_time_ms{};
    double hadronic_commit_time_ms{};
    std::map<std::size_t, std::uint64_t>
        hadronic_worker_batches_by_worker{};
    std::map<std::size_t, std::uint64_t>
        hadronic_worker_requests_by_worker{};
    std::map<std::size_t, double>
        hadronic_worker_final_state_time_by_worker_ms{};
    std::map<HadronicWorkKey, HadronicWorkClassStatistics>
        hadronic_worker_final_state_classes{};
    std::vector<HadronicFlushFingerprint>
        hadronic_flush_fingerprints{};
    std::vector<HadronicFlushLoadRecord>
        hadronic_flush_load_records{};
  };

  struct HybridHadronicWorkerConfig {
    std::uint64_t seed{};
    std::uint64_t shower_id{};
    std::size_t minimum_pending_interactions{64};
    double target_batch_cost_ms{5.};
    std::size_t maximum_batch_items{256};
    double initial_interaction_cost_ms{0.1};
  };

  /**
   * Hybrid scheduling shell with a scalar-only compatibility backend.
   *
   * In this phase every particle is routed through ScalarCascadeStepper. The separate
   * scheduler boundary is intentional: later phases can stage gamma/e-/e+ histories
   * into a device wavefront while retaining the scalar path for all other particles.
   */
  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter = DisabledHybridEmRouter>
  class HybridCascade {
    using stack_view_type = typename TStack::stack_view_type;
    using particle_type = typename TStack::particle_type;
    using volume_tree_node_type =
        std::remove_pointer_t<decltype(std::declval<particle_type>().getNode())>;
    using medium_interface_type = typename volume_tree_node_type::IModelProperties;
    using scheduler_type = CpuOnlyWavefrontScheduler<TStack>;

  public:
    HybridCascade() = delete;
    HybridCascade(HybridCascade const&) = default;
    HybridCascade(HybridCascade&&) = default;
    ~HybridCascade() = default;
    HybridCascade& operator=(HybridCascade const&) = default;

    HybridCascade(Environment<medium_interface_type> const& env, TTracking& tracking,
                  TProcessList& processes, TOutput& output, TStack& stack);
    HybridCascade(Environment<medium_interface_type> const& env, TTracking& tracking,
                  TProcessList& processes, TOutput& output, TStack& stack,
                  TEmRouter& em_router);

    void setNodes();
    void run();
    void forceInteraction();
    void forceDecay();
    void configureHadronicWorkClassification(
        HadronicWorkClassifierConfig const& config);
    void configureHadronicProcessPool(
        HadronicProcessPool& pool,
        HybridHadronicWorkerConfig const& config);
    void enableScalarDetailedPhaseTiming(
        bool const enabled);

    CpuOnlyWavefrontStatistics const& schedulerStatistics() const {
      return scheduler_.statistics();
    }

    HybridCascadeTimingStatistics const& timingStatistics() const {
      return timing_statistics_;
    }

  private:
    struct PendingHadronicInteraction {
      PendingHadronicInteraction(
          std::uint64_t const sequence,
          transport::HistoryId const history,
          particle_type particle_iterator,
          HadronicWorkKey const key,
          HadronicInteractionRequest worker_request)
          : sequence_id(sequence)
          , history_id(history)
          , particle(std::move(particle_iterator))
          , work_key(key)
          , request(std::move(worker_request)) {}

      std::uint64_t sequence_id;
      transport::HistoryId history_id;
      particle_type particle;
      HadronicWorkKey work_key;
      HadronicInteractionRequest request;
    };

    void flushHadronicInteractions(
        HadronicQueueFlushTrigger trigger);
    std::optional<HadronicQueueFlushTrigger>
    hadronicFlushTrigger() const;
    double estimateHadronicInteractionCost(
        HadronicWorkKey const& key) const;

    TProcessList& sequence_;
    TOutput& output_;
    TStack& stack_;
    TEmRouter* em_router_ = nullptr;
    scheduler_type scheduler_;
    ScalarCascadeStepper<TTracking, TProcessList, TStack> stepper_;
    std::uint64_t count_ = 0;
    bool scalar_detailed_phase_timing_enabled_ =
        false;
    HadronicWorkClassifierConfig hadronic_work_classifier_{};
    HadronicProcessPool* hadronic_process_pool_ =
        nullptr;
    HybridHadronicWorkerConfig
        hadronic_worker_config_{};
    ClassifiedHadronicWorkQueue<
        PendingHadronicInteraction>
        pending_hadronic_interactions_;
    std::uint64_t next_hadronic_sequence_id_{};
    HybridCascadeTimingStatistics timing_statistics_{};
  };

} // namespace corsika

#include <corsika/detail/framework/core/HybridCascade.inl>
