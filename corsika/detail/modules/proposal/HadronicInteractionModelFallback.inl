/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::proposal {
  template <typename TModel>
  bool LazyHadronicInteractionModel<TModel>::isValid(
      Code projectile, Code target, HEPEnergyType sqrt_s_per_nucleon) const {
    return model().isValid(projectile, target, sqrt_s_per_nucleon);
  }

  template <typename TModel>
  template <typename TSecondaryView>
  void LazyHadronicInteractionModel<TModel>::doInteraction(
      TSecondaryView& view, Code projectile, Code target,
      FourMomentum const& projectile_p4, FourMomentum const& target_p4) {
    model().doInteraction(view, projectile, target, projectile_p4, target_p4);
  }

  template <typename TModel>
  bool LazyHadronicInteractionModel<TModel>::initialized() const noexcept {
    return static_cast<bool>(model_);
  }

  template <typename TModel>
  TModel& LazyHadronicInteractionModel<TModel>::model() const {
    if (!model_) { model_ = std::make_unique<TModel>(); }
    return *model_;
  }

  template <typename TPreferredModel, typename TFallbackModel>
  HadronicInteractionModelFallback<TPreferredModel, TFallbackModel>::
      HadronicInteractionModelFallback(TPreferredModel& preferred,
                                       TFallbackModel& fallback)
      : preferred_(preferred)
      , fallback_(fallback) {}

  template <typename TPreferredModel, typename TFallbackModel>
  bool HadronicInteractionModelFallback<TPreferredModel, TFallbackModel>::isValid(
      Code projectile, Code target, HEPEnergyType sqrt_s_per_nucleon) const {
    return preferred_.isValid(projectile, target, sqrt_s_per_nucleon) ||
           fallback_.isValid(projectile, target, sqrt_s_per_nucleon);
  }

  template <typename TPreferredModel, typename TFallbackModel>
  template <typename TSecondaryView>
  void HadronicInteractionModelFallback<TPreferredModel, TFallbackModel>::doInteraction(
      TSecondaryView& view, Code projectile, Code target,
      FourMomentum const& projectile_p4, FourMomentum const& target_p4) {
    auto const projectile_a = is_nucleus(projectile) ? get_nucleus_A(projectile) : 1;
    auto const target_a = is_nucleus(target) ? get_nucleus_A(target) : 1;
    auto const sqrt_s_per_nucleon =
        (projectile_p4 / projectile_a + target_p4 / target_a).getNorm();
    if (preferred_.isValid(projectile, target, sqrt_s_per_nucleon)) {
      ++statistics_.preferredInteractions_;
      preferred_.doInteraction(view, projectile, target, projectile_p4, target_p4);
      return;
    }
    if (fallback_.isValid(projectile, target, sqrt_s_per_nucleon)) {
      ++statistics_.fallbackInteractions_;
      CORSIKA_LOG_WARN(
          "PROPOSAL photo-hadronic final state uses the configured "
          "fallback model: projectile={}, target={}, sqrt(S)/nucleon={} GeV, "
          "fallback count={}",
          projectile, target, sqrt_s_per_nucleon / 1_GeV,
          statistics_.fallbackInteractions_);
      fallback_.doInteraction(view, projectile, target, projectile_p4, target_p4);
      return;
    }
    throw std::runtime_error(
        "no configured hadronic model can generate the selected "
        "PROPOSAL photo-hadronic final state");
  }

  template <typename TPreferredModel, typename TFallbackModel>
  HadronicInteractionModelFallbackStatistics const&
  HadronicInteractionModelFallback<TPreferredModel, TFallbackModel>::statistics() const
      noexcept {
    return statistics_;
  }

} // namespace corsika::proposal
