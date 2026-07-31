/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

namespace corsika::proposal {

  /**
   * Delay construction of a heavyweight fallback generator until its first
   * validity query or interaction. The enclosing fallback adapter short
   * circuits on the preferred model, so normal SIBYLL-supported events do not
   * pay QGSJet-II initialization cost.
   *
   * CORSIKA's cascade scheduler invokes process models serially. This wrapper
   * is intentionally not a concurrent lazy-initialization primitive.
   */
  template <typename TModel>
  class LazyHadronicInteractionModel {
  public:
    LazyHadronicInteractionModel() = default;

    LazyHadronicInteractionModel(
        LazyHadronicInteractionModel const&) = delete;
    LazyHadronicInteractionModel& operator=(
        LazyHadronicInteractionModel const&) = delete;

    bool isValid(
        Code projectile, Code target,
        HEPEnergyType sqrt_s_per_nucleon) const {
      return model().isValid(
          projectile, target,
          sqrt_s_per_nucleon);
    }

    template <typename TSecondaryView>
    void doInteraction(
        TSecondaryView& view, Code projectile,
        Code target, FourMomentum const& projectile_p4,
        FourMomentum const& target_p4) {
      model().doInteraction(
          view, projectile, target,
          projectile_p4, target_p4);
    }

    bool initialized() const noexcept {
      return static_cast<bool>(model_);
    }

  private:
    TModel& model() const {
      if (!model_) {
        model_ = std::make_unique<TModel>();
      }
      return *model_;
    }

    mutable std::unique_ptr<TModel> model_;
  };

  struct HadronicInteractionModelFallbackStatistics {
    std::uint64_t preferred_interactions{};
    std::uint64_t fallback_interactions{};
  };

  /**
   * Preserve a preferred hadronic generator while providing a complete,
   * explicitly counted fallback for configurations it cannot represent.
   *
   * This adapter is primarily used by PROPOSAL photoproduction.  SIBYLL does
   * not accept every atmospheric nucleus (notably argon), while QGSJet-II can
   * generate the corresponding rho0-nucleus final state.  Returning an empty
   * final state would destroy the selected photon energy, so a configuration
   * unsupported by both models is a hard error.
   */
  template <typename TPreferredModel, typename TFallbackModel>
  class HadronicInteractionModelFallback {
  public:
    HadronicInteractionModelFallback(
        TPreferredModel& preferred,
        TFallbackModel& fallback)
        : preferred_(preferred), fallback_(fallback) {}

    bool isValid(
        Code projectile, Code target,
        HEPEnergyType sqrt_s_per_nucleon) const {
      return preferred_.isValid(
                 projectile, target,
                 sqrt_s_per_nucleon) ||
             fallback_.isValid(
                 projectile, target,
                 sqrt_s_per_nucleon);
    }

    template <typename TSecondaryView>
    void doInteraction(
        TSecondaryView& view, Code projectile,
        Code target, FourMomentum const& projectile_p4,
        FourMomentum const& target_p4) {
      auto const projectile_a =
          is_nucleus(projectile)
              ? get_nucleus_A(projectile)
              : 1;
      auto const target_a =
          is_nucleus(target) ? get_nucleus_A(target) : 1;
      auto const sqrt_s_per_nucleon =
          (projectile_p4 / projectile_a +
           target_p4 / target_a)
              .getNorm();
      if (preferred_.isValid(
              projectile, target,
              sqrt_s_per_nucleon)) {
        ++statistics_.preferred_interactions;
        preferred_.doInteraction(
            view, projectile, target,
            projectile_p4, target_p4);
        return;
      }
      if (fallback_.isValid(
              projectile, target,
              sqrt_s_per_nucleon)) {
        ++statistics_.fallback_interactions;
        CORSIKA_LOG_WARN(
            "PROPOSAL photo-hadronic final state uses the configured "
            "fallback model: projectile={}, target={}, sqrt(S)/nucleon={} GeV, "
            "fallback count={}",
            projectile, target,
            sqrt_s_per_nucleon / 1_GeV,
            statistics_.fallback_interactions);
        fallback_.doInteraction(
            view, projectile, target,
            projectile_p4, target_p4);
        return;
      }
      throw std::runtime_error(
          "no configured hadronic model can generate the selected "
          "PROPOSAL photo-hadronic final state");
    }

    HadronicInteractionModelFallbackStatistics const&
    statistics() const noexcept {
      return statistics_;
    }

  private:
    TPreferredModel& preferred_;
    TFallbackModel& fallback_;
    HadronicInteractionModelFallbackStatistics statistics_{};
  };

} // namespace corsika::proposal
