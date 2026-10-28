/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/corsika.hpp>

#include <corsika/kokkos/scheduling/CpuParticleScheduler.hpp>
#include <corsika/kokkos/scheduling/HadronicInteractionDeferral.hpp>
#include <corsika/kokkos/scheduling/HadronicProcessPool.hpp>
#include <corsika/kokkos/scheduling/HadronicWorkQueue.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/kokkos/scheduling/ScalarCascadeStepper.hpp>

#include <corsika/media/Environment.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::kokkos {

  namespace hybrid_detail {
    template <typename T, typename = void>
    struct HasPreserveLogicalNodes : std::false_type {};
    template <typename T>
    struct HasPreserveLogicalNodes<
        T, std::void_t<decltype(std::declval<T const&>().preserveLogicalNodes())>>
        : std::true_type {};
    template <typename T, typename = void>
    struct HasEndOfShower : std::false_type {};

    template <typename T>
    struct HasEndOfShower<T, std::void_t<decltype(std::declval<T&>().endOfShower())>>
        : std::true_type {};

    template <typename T, typename = void>
    struct HasConsumeForcedDecay : std::false_type {};

    template <typename T>
    struct HasConsumeForcedDecay<
        T, std::void_t<decltype(std::declval<T&>().consumeForcedDecay(
               std::declval<transport::HistoryId>(), std::declval<transport::StepId>()))>>
        : std::true_type {};

    template <typename T, typename = void>
    struct HasReadyForScalarInterleave : std::false_type {};

    template <typename T>
    struct HasReadyForScalarInterleave<
        T, std::void_t<decltype(std::declval<T const&>().readyForScalarInterleave())>>
        : std::true_type {};
  } // namespace hybrid_detail

  struct DisabledHybridEmRouter {};

  struct HadronicFlushFingerprint {
    std::uint64_t requests_{};
    std::uint64_t requestHash_{};
    std::uint64_t responseHash_{};
  };

  struct HadronicFlushLoadRecord {
    HadronicQueueFlushTrigger trigger_{HadronicQueueFlushTrigger::Drain};
    std::uint64_t requests_{};
    std::vector<double> predictedCostByWorker_;
    std::vector<double> actualFinalStateTimeByWorkerMs_;
  };

  struct HybridCascadeTimingStatistics {
    double totalRunTimeMs_{};
    double outputStartTimeMs_{};
    double setNodesTimeMs_{};
    double routeStageTimeMs_{};
    double scalarStepperTimeMs_{};
    double doStackTimeMs_{};
    double routerAdvanceTimeMs_{};
    double initCascadeEquationsTimeMs_{};
    double cascadeEquationsTimeMs_{};
    double outputEndTimeMs_{};
    std::uint64_t forcedPrimaryInteractions_{};
    std::uint64_t forcedPrimaryDecays_{};
    std::map<std::int32_t, std::uint64_t> scalarStepsByPdg_{};
    std::map<std::int32_t, double> scalarStepperTimeByPdgMs_{};
    std::map<std::int32_t, ScalarCascadePhaseTimingStatistics> scalarPhaseTimingByPdg_{};
    HadronicWorkClassifierConfig hadronicWorkClassifier_{};
    std::map<HadronicWorkKey, HadronicWorkClassStatistics> hadronicWorkClasses_{};
    bool hadronicProcessPoolEnabled_{};
    std::uint64_t hadronicInteractionsPrepared_{};
    std::uint64_t hadronicInteractionsCommitted_{};
    std::uint64_t hadronicWorkerFlushes_{};
    std::uint64_t hadronicWorkerCostTriggeredFlushes_{};
    std::uint64_t hadronicWorkerCapacityTriggeredFlushes_{};
    std::uint64_t hadronicWorkerDrainTriggeredFlushes_{};
    std::uint64_t hadronicWorkerClassifiedBatches_{};
    std::uint64_t hadronicWorkerBatches_{};
    std::uint64_t hadronicWorkerSecondaries_{};
    double hadronicPrepareTimeMs_{};
    double hadronicWorkerExecuteTimeMs_{};
    double hadronicCommitTimeMs_{};
    std::map<std::size_t, std::uint64_t> hadronicWorkerBatchesByWorker_{};
    std::map<std::size_t, std::uint64_t> hadronicWorkerRequestsByWorker_{};
    std::map<std::size_t, double> hadronicWorkerFinalStateTimeByWorkerMs_{};
    std::map<HadronicWorkKey, HadronicWorkClassStatistics>
        hadronicWorkerFinalStateClasses_{};
    std::vector<HadronicFlushFingerprint> hadronicFlushFingerprints_{};
    std::vector<HadronicFlushLoadRecord> hadronicFlushLoadRecords_{};
  };

  struct HybridHadronicWorkerConfig {
    std::uint64_t seed_{};
    std::uint64_t showerId_{};
    std::size_t minimumPendingInteractions_{64};
    double targetBatchCostMs_{5.};
    std::size_t maximumBatchItems_{256};
    double initialInteractionCostMs_{0.1};
  };

  /**
   * Coordinate scalar transport, an optional EM router and hadronic workers.
   *
   * The native stack owns particles throughout scheduling. An EM router may stage
   * supported histories; deferred hadronic projectiles remain suspended until their
   * ordered final states are committed through the native secondary process chain.
   * Referenced environment, processes, stack, router and pool must outlive this object.
   */
  template <typename TTracking, typename TProcessList, typename TOutput, typename TStack,
            typename TEmRouter = DisabledHybridEmRouter>
  class HybridCascade {
    using stack_view_type = typename TStack::stack_view_type;
    using particle_type = typename TStack::particle_type;
    using volume_tree_node_type =
        std::remove_pointer_t<decltype(std::declval<particle_type>().getNode())>;
    using medium_interface_type = typename volume_tree_node_type::IModelProperties;
    using scheduler_type = CpuParticleScheduler<TStack>;

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
    void configureHadronicWorkClassification(HadronicWorkClassifierConfig const& config);
    void configureHadronicProcessPool(HadronicProcessPool& pool,
                                      HybridHadronicWorkerConfig const& config);
    void enableScalarDetailedPhaseTiming(bool const enabled);

    CpuParticleSchedulerStatistics const& schedulerStatistics() const;

    HybridCascadeTimingStatistics const& timingStatistics() const;

  private:
    enum class PendingForcedAction {
      None,
      Interaction,
      Decay,
    };

    struct PendingHadronicInteraction {
      PendingHadronicInteraction(std::uint64_t const sequence,
                                 transport::HistoryId const history,
                                 particle_type particle_iterator,
                                 HadronicWorkKey const key,
                                 HadronicInteractionRequest worker_request);

      std::uint64_t sequenceId_;
      transport::HistoryId historyId_;
      particle_type particle_;
      HadronicWorkKey workKey_;
      HadronicInteractionRequest request_;
    };

    void flushHadronicInteractions(HadronicQueueFlushTrigger trigger);
    std::optional<HadronicQueueFlushTrigger> hadronicFlushTrigger() const;
    double estimateHadronicInteractionCost(HadronicWorkKey const& key) const;

    TProcessList& sequence_;
    TOutput& output_;
    TStack& stack_;
    TEmRouter* emRouter_ = nullptr;
    scheduler_type scheduler_;
    ScalarCascadeStepper<TTracking, TProcessList, TStack> stepper_;
    std::uint64_t count_ = 0;
    bool scalarDetailedPhaseTimingEnabled_ = false;
    PendingForcedAction pendingForcedAction_{PendingForcedAction::None};
    HadronicWorkClassifierConfig hadronicWorkClassifier_{};
    HadronicProcessPool* hadronicProcessPool_ = nullptr;
    HybridHadronicWorkerConfig hadronicWorkerConfig_{};
    ClassifiedHadronicWorkQueue<PendingHadronicInteraction> pendingHadronicInteractions_;
    std::uint64_t nextHadronicSequenceId_{};
    HybridCascadeTimingStatistics timingStatistics_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/HybridCascade.inl>
