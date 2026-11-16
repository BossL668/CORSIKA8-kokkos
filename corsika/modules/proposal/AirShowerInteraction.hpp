/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/PROPOSAL.hpp>
#include <cstdint>

namespace corsika::proposal {
  namespace detail {
    /** Count generator invocations without changing validity checks, random draws,
     * or final states. The application retains ownership of the selected model.
     */
    template <typename TModel>
    class CountedPhotonModel {
    public:
      struct Statistics {
        std::uint64_t interactions_{};
      };

      explicit CountedPhotonModel(TModel& model);
      bool isValid(Code projectile, Code target, HEPEnergyType energy) const;
      template <typename TView>
      void doInteraction(TView& view, Code projectile, Code target,
                         FourMomentum const& projectileMomentum,
                         FourMomentum const& targetMomentum);
      Statistics const& statistics() const;

    private:
      TModel& model_;
      Statistics statistics_;
    };

    /** Borrow the models selected by the application, with no substitute model. */
    template <typename TLow, typename THigh>
    struct AirShowerPhotonModels {
      using low_type = CountedPhotonModel<TLow>;
      using high_type = CountedPhotonModel<THigh>;
      low_type low_;
      high_type high_;

      AirShowerPhotonModels(TLow&, THigh&);
    };
  } // namespace detail

  /** Expose the application's photoproduction models to run diagnostics.
   * The PROPOSAL process uses the official model selection and final states.
   * This host adapter has no dependency on Kokkos.
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
  };
  template <typename TE, typename TL, typename TH>
  AirShowerInteraction(TE const&, TL&, TH&, HEPEnergyType const&)
      -> AirShowerInteraction<TL, TH>;
} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/AirShowerInteraction.inl>
