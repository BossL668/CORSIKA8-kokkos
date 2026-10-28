/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/stack/TransportIdentityStackExtension.hpp>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <set>
#include <stdexcept>

namespace corsika::kokkos {

  struct CpuParticleSchedulerStatistics {
    std::uint64_t acquiredParticleSteps_ = 0;
    std::uint64_t completedParticleSteps_ = 0;
    std::uint64_t maxWavefrontSize_ = 0;
    std::uint64_t suspendedParticles_ = 0;
    std::uint64_t resumedParticles_ = 0;
    std::uint64_t maximumSuspendedParticles_ = 0;
  };

  /** LIFO host scheduler for the accelerated cascade.
   * Acquire one native-stack particle at a time, preserving production order.
   * Suspended histories remain in the stack while a CPU worker owns their
   * interaction. Do not compact or reorder that stack until they are resumed.
   * The stack must outlive the scheduler; only one step may be in flight.
   * This class neither chooses physics processes nor creates worker threads.
   */
  template <typename TStack>
  class CpuParticleScheduler {
  public:
    using particle_type = typename TStack::particle_type;

    struct ScheduledParticle {
      particle_type particle_;
      transport::HistoryId historyId_;
      transport::HistoryId parentHistoryId_;
      transport::Generation generation_;
      transport::StepId stepId_;
    };

    static_assert(transport::has_transport_identity_v<particle_type>,
                  "CpuOnlyWavefrontScheduler requires TransportIdentityStackExtension");

    explicit CpuParticleScheduler(TStack& stack);

    bool empty();

    ScheduledParticle acquireNext();

    void completeParticleStep(bool const suspendParticle = false);

    void resumeParticle(transport::HistoryId const historyId);

    bool hasSuspendedParticles() const noexcept;

    std::size_t suspendedParticleCount() const noexcept;

    CpuParticleSchedulerStatistics const& statistics() const;

  private:
    TStack& stack_;
    CpuParticleSchedulerStatistics statistics_;
    bool particleStepInFlight_ = false;
    transport::HistoryId inFlightHistoryId_ = transport::NoParentHistoryId;
    std::set<transport::HistoryId> suspendedHistoryIds_;
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/CpuParticleScheduler.inl>
