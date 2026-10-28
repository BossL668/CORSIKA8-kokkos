/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  template <typename TStack>
  CpuParticleScheduler<TStack>::CpuParticleScheduler(TStack& stack)
      : stack_(stack) {}
  template <typename TStack>
  bool CpuParticleScheduler<TStack>::empty() {
    while (!stack_.isEmpty() && stack_.purgeLastIfDeleted()) {}
    for (unsigned int offset = 0; offset < stack_.getSize(); ++offset) {
      auto particle = stack_.at(stack_.getSize() - 1 - offset);
      if (!particle.isErased() &&
          suspendedHistoryIds_.count(particle.getHistoryId()) == 0) {
        return false;
      }
    }
    return true;
  }
  template <typename TStack>
  typename CpuParticleScheduler<TStack>::ScheduledParticle
  CpuParticleScheduler<TStack>::acquireNext() {
    if (particleStepInFlight_) {
      throw std::logic_error("CPU-only scheduler already has a particle step in flight");
    }
    if (empty()) {
      throw std::logic_error("Cannot acquire a particle from an empty stack");
    }

    particle_type particle = stack_.last();
    bool found = false;
    for (unsigned int offset = 0; offset < stack_.getSize(); ++offset) {
      auto candidate = stack_.at(stack_.getSize() - 1 - offset);
      if (!candidate.isErased() &&
          suspendedHistoryIds_.count(candidate.getHistoryId()) == 0) {
        particle = candidate;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::logic_error("No runnable particle exists outside the suspended set");
    }
    ScheduledParticle scheduled{particle, particle.getHistoryId(),
                                particle.getParentHistoryId(), particle.getGeneration(),
                                particle.beginTransportStep()};

    ++statistics_.acquiredParticleSteps_;
    statistics_.maxWavefrontSize_ =
        std::max<std::uint64_t>(statistics_.maxWavefrontSize_, 1);
    particleStepInFlight_ = true;
    inFlightHistoryId_ = scheduled.historyId_;
    return scheduled;
  }
  template <typename TStack>
  void CpuParticleScheduler<TStack>::completeParticleStep(bool const suspendParticle) {
    if (!particleStepInFlight_) {
      throw std::logic_error("No acquired particle step is awaiting completion");
    }
    if (suspendParticle) {
      if (!suspendedHistoryIds_.insert(inFlightHistoryId_).second) {
        throw std::logic_error("Particle history is already suspended");
      }
      ++statistics_.suspendedParticles_;
      statistics_.maximumSuspendedParticles_ = std::max<std::uint64_t>(
          statistics_.maximumSuspendedParticles_, suspendedHistoryIds_.size());
    }
    ++statistics_.completedParticleSteps_;
    particleStepInFlight_ = false;
    inFlightHistoryId_ = transport::NoParentHistoryId;
  }
  template <typename TStack>
  void CpuParticleScheduler<TStack>::resumeParticle(
      transport::HistoryId const historyId) {
    if (particleStepInFlight_) {
      throw std::logic_error("Cannot resume a particle while another step is in flight");
    }
    if (suspendedHistoryIds_.erase(historyId) != 1) {
      throw std::logic_error("Cannot resume a particle history that is not suspended");
    }
    ++statistics_.resumedParticles_;
  }
  template <typename TStack>
  bool CpuParticleScheduler<TStack>::hasSuspendedParticles() const noexcept {
    return !suspendedHistoryIds_.empty();
  }
  template <typename TStack>
  std::size_t CpuParticleScheduler<TStack>::suspendedParticleCount() const noexcept {
    return suspendedHistoryIds_.size();
  }
  template <typename TStack>
  CpuParticleSchedulerStatistics const& CpuParticleScheduler<TStack>::statistics() const {
    return statistics_;
  }
} // namespace corsika::kokkos
