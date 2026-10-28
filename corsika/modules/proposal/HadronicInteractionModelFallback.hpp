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

    LazyHadronicInteractionModel(LazyHadronicInteractionModel const&) = delete;
    LazyHadronicInteractionModel& operator=(LazyHadronicInteractionModel const&) = delete;

    bool isValid(Code projectile, Code target, HEPEnergyType sqrt_s_per_nucleon) const;

    template <typename TSecondaryView>
    void doInteraction(TSecondaryView& view, Code projectile, Code target,
                       FourMomentum const& projectile_p4, FourMomentum const& target_p4);

    bool initialized() const noexcept;

  private:
    TModel& model() const;

    mutable std::unique_ptr<TModel> model_;
  };

  struct HadronicInteractionModelFallbackStatistics {
    std::uint64_t preferredInteractions_{};
    std::uint64_t fallbackInteractions_{};
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
    HadronicInteractionModelFallback(TPreferredModel& preferred,
                                     TFallbackModel& fallback);

    bool isValid(Code projectile, Code target, HEPEnergyType sqrt_s_per_nucleon) const;

    template <typename TSecondaryView>
    void doInteraction(TSecondaryView& view, Code projectile, Code target,
                       FourMomentum const& projectile_p4, FourMomentum const& target_p4);

    HadronicInteractionModelFallbackStatistics const& statistics() const noexcept;

  private:
    TPreferredModel& preferred_;
    TFallbackModel& fallback_;
    HadronicInteractionModelFallbackStatistics statistics_{};
  };

} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/HadronicInteractionModelFallback.inl>
