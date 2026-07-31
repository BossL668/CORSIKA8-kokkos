/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/InteractionHistogram.hpp>

namespace corsika {

  template <class TCountedProcess>
  inline InteractionCounter<TCountedProcess>::InteractionCounter(TCountedProcess& process)
      : process_(process) {}

  template <class TCountedProcess>
  template <typename TSecondaryView>
  inline void InteractionCounter<TCountedProcess>::doInteraction(
      TSecondaryView& view, Code const projectileId, Code const targetId,
      FourMomentum const& projectileP4, FourMomentum const& targetP4) {
    size_t const massNumber = is_nucleus(targetId) ? get_nucleus_A(targetId) : 1;
    auto const massTarget = massNumber * constants::nucleonMass;
    histogram_.fill(projectileId, projectileP4.getTimeLikeComponent(), massTarget);
    if constexpr (
        has_hadronic_worker_model_v<TCountedProcess>) {
      if (auto* context =
              HadronicInteractionDeferralContext::current();
          context != nullptr &&
          context->tryPrepare(
              TCountedProcess::hadronic_worker_model,
              projectileId, targetId, projectileP4,
              targetP4)) {
        timing_samples_.push_back(
            InteractionTimingSample{
                count_, projectileId, targetId,
                projectileP4.getTimeLikeComponent() -
                    get_mass(projectileId),
                0., true});
        ++count_;
        return;
      }
    }
    auto const start = std::chrono::steady_clock::now();
    process_.doInteraction(view, projectileId, targetId, projectileP4, targetP4);
    auto const elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    total_final_state_time_ms_ += elapsed_ms;
    timing_samples_.push_back(
        InteractionTimingSample{
            count_, projectileId, targetId,
            projectileP4.getTimeLikeComponent() -
                get_mass(projectileId),
            elapsed_ms, false});
    ++count_;
  }

  template <class TCountedProcess>
  inline CrossSectionType InteractionCounter<TCountedProcess>::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {
    return process_.getCrossSection(projectileId, targetId, projectileP4, targetP4);
  }

  template <class TCountedProcess>
  inline InteractionHistogram const& InteractionCounter<TCountedProcess>::getHistogram()
      const {
    return histogram_;
  }

  template <class TCountedProcess>
  inline std::uint64_t InteractionCounter<TCountedProcess>::getCount() const {
    return count_;
  }

  template <class TCountedProcess>
  inline std::vector<InteractionTimingSample> const&
  InteractionCounter<TCountedProcess>::getTimingSamples() const {
    return timing_samples_;
  }

  template <class TCountedProcess>
  inline double
  InteractionCounter<TCountedProcess>::getTotalFinalStateTimeMs() const {
    return total_final_state_time_ms_;
  }

} // namespace corsika
