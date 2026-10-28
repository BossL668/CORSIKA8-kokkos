/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/stack/history/HistoryStackExtension.hpp>

#include <corsika/media/Environment.hpp>
#include <corsika/media/NuclearComposition.hpp>

#include <type_traits>
#include <chrono>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace corsika::kokkos {

  enum class ScalarCascadeAdvanceResult { Completed, DeferredHadronicInteraction };

  struct ScalarCascadePhaseTimingStatistics {
    std::uint64_t steps_{};
    std::uint64_t geometryBoundaryCrossings_{};
    std::uint64_t geometryStepLimits_{};
    std::uint64_t interactions_{};
    std::uint64_t decays_{};
    std::uint64_t secondaryProcessingCalls_{};
    double crossSectionTimeMs_{};
    double distanceSamplingTimeMs_{};
    double trackingTimeMs_{};
    double continuousLimitTimeMs_{};
    double continuousProcessTimeMs_{};
    double geometryBoundaryTimeMs_{};
    double geometryStepLimitTimeMs_{};
    double interactionTimeMs_{};
    double decayTimeMs_{};
    double secondaryProcessingTimeMs_{};
  };

  /**
   * Advance one particle with the scalar CORSIKA transport algorithm.
   *
   * This class owns no particle data. It operates on the existing CORSIKA stack and
   * process sequence and deliberately preserves the random-number draw order and
   * physics decisions formerly implemented directly in Cascade.
   *
   * Separating one-particle transport from stack scheduling is the first boundary
   * needed by alternative schedulers such as a CPU/GPU hybrid cascade.
   */
  template <typename TTracking, typename TProcessList, typename TStack>
  class ScalarCascadeStepper {

    using stack_view_type = typename TStack::stack_view_type;
    using particle_type = typename TStack::particle_type;
    using volume_tree_node_type =
        std::remove_pointer_t<decltype(std::declval<particle_type>().getNode())>;
    using medium_interface_type = typename volume_tree_node_type::IModelProperties;

  public:
    ScalarCascadeStepper() = delete;
    ScalarCascadeStepper(ScalarCascadeStepper const&) = default;
    ScalarCascadeStepper(ScalarCascadeStepper&&) = default;
    ~ScalarCascadeStepper() = default;
    ScalarCascadeStepper& operator=(ScalarCascadeStepper const&) = default;

    ScalarCascadeStepper(Environment<medium_interface_type> const& env, TTracking& tr,
                         TProcessList& pl, TStack& stack);

    /**
     * Set the logical environment node for every particle currently on the stack.
     */
    void setNodes(bool preserve_existing = false);

    /**
     * Advance one particle until the nearest continuous, discrete, or geometric
     * limiter. The caller remains responsible for stack scheduling and doStack().
     */
    ScalarCascadeAdvanceResult advance(particle_type& particle);

    /**
     * Commit a process-isolated hadronic final state to the unique main stack.
     *
     * The projectile must still be parked at the transported interaction vertex.
     * This method reconstructs the SecondaryView, runs the unchanged
     * doSecondaries() chain, and only then erases the projectile.
     */
    void commitDeferredHadronicInteraction(
        particle_type& particle, std::vector<HadronicSecondaryRecord> const& secondaries);

    /**
     * Force the next advanced particle to interact at its current position.
     */
    void forceInteraction();

    /**
     * Force the next advanced particle to decay at its current position.
     */
    void forceDecay();

    void enableDetailedPhaseTiming(bool const enabled);

    void resetDetailedPhaseTiming();

    std::map<std::int32_t, ScalarCascadePhaseTimingStatistics> const&
    detailedPhaseTiming() const;

  private:
    ProcessReturn decay(stack_view_type& view, InverseTimeType initial_inv_decay_time);
    ProcessReturn interaction(stack_view_type& view, FourMomentum const& projectileP4,
                              NuclearComposition const& composition,
                              CrossSectionType initial_cross_section);
    void setEventType(stack_view_type& view, history::EventType eventType);

    Environment<medium_interface_type> const& environment_;
    TTracking& tracking_;
    TProcessList& sequence_;
    TStack& stack_;
    default_prng_type& rng_ = RNGManager<>::getInstance().getRandomStream("cascade");
    bool forceInteraction_ = false;
    bool forceDecay_ = false;
    bool detailedPhaseTimingEnabled_ = false;
    std::map<std::int32_t, ScalarCascadePhaseTimingStatistics> detailedPhaseTiming_;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/ScalarCascadeStepper.inl>
