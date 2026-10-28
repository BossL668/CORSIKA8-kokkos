/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/modules/QGSJetII.hpp>
#include <corsika/modules/proposal/HadronicInteractionModelFallback.hpp>
#include <corsika/modules/proposal/ThresholdPhotoproductionModel.hpp>

namespace corsika::proposal {
  namespace detail {
    /** Own the existing fallback models before the referencing PROPOSAL process. */
    template <typename TLow, typename THigh>
    struct AirShowerPhotonModels {
      using low_type =
          HadronicInteractionModelFallback<TLow, ThresholdPhotoproductionModel>;
      using fallback_type = LazyHadronicInteractionModel<qgsjetII::InteractionModel>;
      using high_type = HadronicInteractionModelFallback<THigh, fallback_type>;
      ThresholdPhotoproductionModel threshold_;
      low_type low_;
      fallback_type fallback_;
      high_type high_;

      AirShowerPhotonModels(TLow&, THigh&);
    };
  } // namespace detail

  /** Package the established photoproduction policy without changing its models.
   * Preferred generators are borrowed; threshold and lazy fallback lifetimes are
   * internal. This host physics adapter has no dependency on Kokkos.
   */
  template <typename TLow, typename THigh>
  class AirShowerInteraction
      : private detail::AirShowerPhotonModels<TLow, THigh>,
        public InteractionModel<
            typename detail::AirShowerPhotonModels<TLow, THigh>::low_type,
            typename detail::AirShowerPhotonModels<TLow, THigh>::high_type>,
        public InteractionProcess<AirShowerInteraction<TLow, THigh>> {
    using models_type = detail::AirShowerPhotonModels<TLow, THigh>;
    using process_type =
        InteractionModel<typename models_type::low_type, typename models_type::high_type>;

  public:
    template <typename TEnvironment>
    AirShowerInteraction(TEnvironment const&, TLow&, THigh&, HEPEnergyType const&);
    AirShowerInteraction(AirShowerInteraction const&) = delete;
    AirShowerInteraction& operator=(AirShowerInteraction const&) = delete;
    auto& lowEnergyModel();
    auto& highEnergyModel();
    auto const& fallbackModel() const;
  };
  template <typename TE, typename TL, typename TH>
  AirShowerInteraction(TE const&, TL&, TH&, HEPEnergyType const&)
      ->AirShowerInteraction<TL, TH>;
} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/AirShowerInteraction.inl>
