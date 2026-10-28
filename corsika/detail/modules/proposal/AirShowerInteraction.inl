/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace corsika::proposal {
  namespace detail {
    template <typename TL, typename TH>
    AirShowerPhotonModels<TL, TH>::AirShowerPhotonModels(TL& low, TH& high)
        : low_(low, threshold_)
        , high_(high, fallback_) {}
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
  template <typename TL, typename TH>
  auto const& AirShowerInteraction<TL, TH>::fallbackModel() const {
    return models_type::fallback_;
  }
} // namespace corsika::proposal
