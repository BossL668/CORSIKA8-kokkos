/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace corsika::proposal {
  namespace detail {
    template <typename TModel>
    CountedPhotonModel<TModel>::CountedPhotonModel(TModel& model)
        : model_(model) {}

    template <typename TModel>
    bool CountedPhotonModel<TModel>::isValid(Code projectile, Code target,
                                             HEPEnergyType energy) const {
      return model_.isValid(projectile, target, energy);
    }

    template <typename TModel>
    template <typename TView>
    void CountedPhotonModel<TModel>::doInteraction(TView& view, Code projectile,
                                                   Code target,
                                                   FourMomentum const& projectileMomentum,
                                                   FourMomentum const& targetMomentum) {
      ++statistics_.interactions_;
      model_.doInteraction(view, projectile, target, projectileMomentum, targetMomentum);
    }

    template <typename TModel>
    auto CountedPhotonModel<TModel>::statistics() const -> Statistics const& {
      return statistics_;
    }

    template <typename TL, typename TH>
    AirShowerPhotonModels<TL, TH>::AirShowerPhotonModels(TL& low, TH& high)
        : low_(low)
        , high_(high) {}
  } // namespace detail

  template <typename TL, typename TH>
  template <typename TE>
  AirShowerInteraction<TL, TH>::AirShowerInteraction(TE const& environment, TL& low,
                                                     TH& high,
                                                     HEPEnergyType const& threshold)
      : models_type(low, high)
      , process_type(environment, models_type::low_, models_type::high_, threshold) {}

  template <typename TL, typename TH>
  auto& AirShowerInteraction<TL, TH>::lowEnergyModel() {
    return models_type::low_;
  }
  template <typename TL, typename TH>
  auto& AirShowerInteraction<TL, TH>::highEnergyModel() {
    return models_type::high_;
  }
} // namespace corsika::proposal
