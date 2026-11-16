/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/geometry/FourVector.hpp>

#include <tuple>

namespace corsika::sophia {

  /**
   * @brief Provides the SOPHIA photon-nucleon interaction model.
   *
   * This is a TModel argument for InteractionProcess<TModel>.
   */

  class InteractionModel {

  public:
    InteractionModel();
    ~InteractionModel();

    /**
     * @brief Set the Verbose flag.
     *
     * If flag is true, SOPHIA will printout additional secondary particle information
     * lists, etc.
     *
     * @param flag to switch.
     */
    void setVerbose(bool const flag);

    /**
     * @brief evaluated validity of collision system.
     *
     * SOPHIA only accepts nucleons as targets, that is protons (Hydrogen) or
     * neutrons.
     */
    bool constexpr isValid(Code const projectileId, Code const targetId,
                           HEPEnergyType const sqrtSnn) const;

    /**
     * Returns inelastic (production) cross section.
     *
     * This cross section must correspond to the process described in doInteraction.
     * Allowed targets are: nuclei or single nucleons (p,n,hydrogen).
     *
     * @param projectile is the Code of the projectile
     * @param target is the Code of the target
     * @param projectileP4: four-momentum of projectile
     * @param targetP4: four-momentum of target
     *
     * @return inelastic cross section
     * elastic cross section
     */
    CrossSectionType getCrossSection(
        [[maybe_unused]] Code const projectile, [[maybe_unused]] Code const target,
        [[maybe_unused]] FourMomentum const& projectileP4,
        [[maybe_unused]] FourMomentum const& targetP4) const {
      CORSIKA_LOGGER_ERROR(logger_, "cross section not implemented in SOPHIA!");
      return CrossSectionType::zero();
    }
    /**
     * In this function SOPHIA is called to produce one event. The
     * event is copied (and boosted) into the frame of the incoming particles.
     *
     * @param view is the stack object for the secondaries
     * @param projectile is the Code of the projectile
     * @param target is the Code of the target
     * @param projectileP4: four-momentum of projectile
     * @param targetP4: four-momentum of target
     */

    template <typename TSecondaries>
    void doInteraction(TSecondaries& view, Code const projectile, Code const target,
                       FourMomentum const& projectileP4, FourMomentum const& targetP4);

  private:
    HEPEnergyType constexpr getMinEnergyCoM() const { return minEnergyCoM_; }
    HEPEnergyType constexpr getMaxEnergyCoM() const { return maxEnergyCoM_; }

    // hard model limits
    static HEPEnergyType constexpr minEnergyCoM_ = 1.079166345 * 1e9 * electronvolt;
    static HEPEnergyType constexpr maxEnergyCoM_ = 1.e6 * 1e9 * electronvolt;

    default_prng_type& RNG_ = RNGManager<>::getInstance().getRandomStream("sophia");

    // data members
    int count_ = 0;
    bool sophia_listing_;

    std::shared_ptr<spdlog::logger> logger_ =
        get_logger("corsika_sophia_InteractionModel");
  };

} // namespace corsika::sophia

#include <corsika/detail/modules/sophia/InteractionModel.inl>