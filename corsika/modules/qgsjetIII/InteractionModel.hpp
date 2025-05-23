/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/qgsjetIII/ParticleConversion.hpp>
#include <qgsjet-III-04.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/utility/CorsikaData.hpp>

#include <boost/filesystem/path.hpp>

#include <random>

namespace corsika::qgsjetIII {

  class InteractionModel {

  public:
    InteractionModel(boost::filesystem::path dataPath = corsika_data("QGSJetIII"));
    ~InteractionModel();

    /**
     * Throws exception if invalid system is passed.
     *
     * @param beamId
     * @param targetId
     */
    bool isValid(Code const beamId, Code const targetId, HEPEnergyType const sqrtS) const;

    /**
     * Return the QGSJETII inelastic/production cross section.
     *
     * This cross section must correspond to the process described in doInteraction.
     * Allowed targets are: nuclei or single nucleons (p,n,hydrogen).
     *
     * @param projectile is the Code of the projectile
     * @param target is the Code of the target
     * @param sqrtSnn is the center-of-mass energy (per nucleon pair)
     * @param Aprojectile is the mass number of the projectils, if it is a nucleus
     * @param Atarget is the mass number of the target, if it is a nucleus
     *
     * @return inelastic cross section.
     */
    CrossSectionType getCrossSection(Code const projectile, Code const target,
                                     FourMomentum const& projectileP4,
                                     FourMomentum const& targetP4) const;

    /**
     * Returns inelastic AND elastic cross sections.
     *
     * These cross sections must correspond to the process described in doInteraction
     * AND elastic scattering (sigma_tot = sigma_inel + sigma_el).
     *
     * @param projectile is the Code of the projectile
     * @param target is the Code of the target
     * @param projectileP4: four-momentum of projectile
     * @param targetP4: four-momentum of target
     *
     * @return a tuple of: inelastic cross section, elastic cross section
     */
    std::tuple<CrossSectionType, CrossSectionType> getCrossSectionInelEla(
        Code const projectile, Code const target, FourMomentum const& projectileP4,
        FourMomentum const& targetP4) const;

    /**
     * In this function QGSJETIII is called to produce one event.
     *
     * The event is copied (and boosted) into the shower lab frame.
     */
    template <typename TSecondaries>
    void doInteraction(TSecondaries&, Code const projectile, Code const target,
                       FourMomentum const& projectileP4, FourMomentum const& targetP4);

  private:
    int count_ = 0;
    QgsjetIIIHadronType alternate_ =
        QgsjetIIIHadronType::PiPlusType; // for pi0, rho0 projectiles

    corsika::default_prng_type& rng_ =
        corsika::RNGManager<>::getInstance().getRandomStream("qgsjetIII");
    std::bernoulli_distribution bernoulli_;
    static size_t constexpr maxMassNumber_ = 208;
    static HEPEnergyType constexpr sqrtSmin_ = 10_GeV;
  };

} // namespace corsika::qgsjetIII

#include <corsika/detail/modules/qgsjetIII/InteractionModel.inl>
