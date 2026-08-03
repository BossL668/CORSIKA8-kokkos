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
#include <set>
#include <stdexcept>

namespace corsika {

  struct CpuOnlyWavefrontStatistics {
    std::uint64_t acquired_particle_steps = 0;
    std::uint64_t completed_particle_steps = 0;
    std::uint64_t max_wavefront_size = 0;
    std::uint64_t suspended_particles = 0;
    std::uint64_t resumed_particles = 0;
    std::uint64_t maximum_suspended_particles = 0;
  };

  /**
   * Compatibility scheduler for HybridCascade.
   *
   * It deliberately exposes a wavefront-shaped scheduling boundary while acquiring
   * exactly one LIFO particle at a time. Consequently it preserves the existing scalar
   * Cascade order and random-number draw order. A later GPU scheduler can replace this
   * class without changing ScalarCascadeStepper.
   */
  template <typename TStack>
  class CpuOnlyWavefrontScheduler {
  public:
    using particle_type = typename TStack::particle_type;

    struct ScheduledParticle {
      particle_type particle;
      transport::HistoryId history_id;
      transport::HistoryId parent_history_id;
      transport::Generation generation;
      transport::StepId step_id;
    };

    static_assert(
        transport::has_transport_identity_v<particle_type>,
        "CpuOnlyWavefrontScheduler requires TransportIdentityStackExtension");

    explicit CpuOnlyWavefrontScheduler(TStack& stack)
        : stack_(stack) {}

    bool empty() {
      while (!stack_.isEmpty() &&
             stack_.purgeLastIfDeleted()) {}
      for (unsigned int offset = 0;
           offset < stack_.getSize(); ++offset) {
        auto particle =
            stack_.at(stack_.getSize() - 1 - offset);
        if (!particle.isErased() &&
            suspended_history_ids_.count(
                particle.getHistoryId()) == 0) {
          return false;
        }
      }
      return true;
    }

    ScheduledParticle acquireNext() {
      if (particle_step_in_flight_) {
        throw std::logic_error(
            "CPU-only scheduler already has a particle step in flight");
      }
      if (empty()) {
        throw std::logic_error("Cannot acquire a particle from an empty stack");
      }

      particle_type particle = stack_.last();
      bool found = false;
      for (unsigned int offset = 0;
           offset < stack_.getSize(); ++offset) {
        auto candidate =
            stack_.at(stack_.getSize() - 1 - offset);
        if (!candidate.isErased() &&
            suspended_history_ids_.count(
                candidate.getHistoryId()) == 0) {
          particle = candidate;
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::logic_error(
            "No runnable particle exists outside the suspended set");
      }
      ScheduledParticle scheduled{particle,
                                  particle.getHistoryId(),
                                  particle.getParentHistoryId(),
                                  particle.getGeneration(),
                                  particle.beginTransportStep()};

      ++statistics_.acquired_particle_steps;
      statistics_.max_wavefront_size =
          std::max<std::uint64_t>(statistics_.max_wavefront_size, 1);
      particle_step_in_flight_ = true;
      in_flight_history_id_ = scheduled.history_id;
      return scheduled;
    }

    void completeParticleStep(
        bool const suspend_particle = false) {
      if (!particle_step_in_flight_) {
        throw std::logic_error("No acquired particle step is awaiting completion");
      }
      if (suspend_particle) {
        if (!suspended_history_ids_
                 .insert(in_flight_history_id_)
                 .second) {
          throw std::logic_error(
              "Particle history is already suspended");
        }
        ++statistics_.suspended_particles;
        statistics_.maximum_suspended_particles =
            std::max<std::uint64_t>(
                statistics_.maximum_suspended_particles,
                suspended_history_ids_.size());
      }
      ++statistics_.completed_particle_steps;
      particle_step_in_flight_ = false;
      in_flight_history_id_ =
          transport::NoParentHistoryId;
    }

    void resumeParticle(
        transport::HistoryId const history_id) {
      if (particle_step_in_flight_) {
        throw std::logic_error(
            "Cannot resume a particle while another step is in flight");
      }
      if (suspended_history_ids_.erase(history_id) !=
          1) {
        throw std::logic_error(
            "Cannot resume a particle history that is not suspended");
      }
      ++statistics_.resumed_particles;
    }

    bool hasSuspendedParticles() const noexcept {
      return !suspended_history_ids_.empty();
    }

    std::size_t suspendedParticleCount() const noexcept {
      return suspended_history_ids_.size();
    }

    CpuOnlyWavefrontStatistics const& statistics() const { return statistics_; }

  private:
    TStack& stack_;
    CpuOnlyWavefrontStatistics statistics_;
    bool particle_step_in_flight_ = false;
    transport::HistoryId in_flight_history_id_ =
        transport::NoParentHistoryId;
    std::set<transport::HistoryId>
        suspended_history_ids_;
  };

} // namespace corsika
