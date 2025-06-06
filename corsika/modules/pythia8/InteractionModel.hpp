/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <tuple>
#include <unordered_map>
#include <boost/filesystem/path.hpp>

#include <corsika/framework/utility/CorsikaData.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/utility/CrossSectionTable.hpp>
#include <corsika/framework/process/InteractionProcess.hpp>
#include <corsika/modules/pythia8/Pythia8.hpp>

namespace corsika::pythia8 {

  /**
   * @brief Defines the interface to the PYTHIA8 interaction model. Configured for
   * hadron-nucleus interactions using Angantyr.
   *
   * This is a TModel argument for InteractionProcess<TModel>.
   */

  class InteractionModel {

  public:
    /**
     * Constructs the interface for PYTHIA8
     *
     * @param stableParticles - set of particle ids to be treated as stable inside
     * pythia. if empty then all particles are treated as stable (no decays!)
     * @param dataPath path to the pythia tables
     * @param printListing switch on/off the pythia printout
     */
    InteractionModel(std::set<Code> const& stableParticles = {},
                     boost::filesystem::path const& dataPath = corsika_data("Pythia"),
                     bool const printListing = false);
    ~InteractionModel();

    bool canInteract(Code const) const;

    /**
     * Check if configuration of projectile and target is valid.
     *
     * @param projectileId is the Code of the projectile
     * @param targetId is the Code of the target
     * @param sqrtS is the squared sum of the 4-momenta of projectile and target
     */
    bool isValid(Code const projectileId, Code const targetId,
                 HEPEnergyType const sqrtS) const;

    /**
     * Returns inelastic AND elastic cross sections.
     *
     * These cross sections must correspond to the process described in doInteraction
     * (sigma_tot = sigma_inel + sigma_el). Allowed targets are:
     * nuclei or single nucleons (p,n,hydrogen).
     *
     * @param projectile is the Code of the projectile
     * @param target is the Code of the target
     * @param projectileP4 is the 4-momentum of the projectile
     * @param targetP4 is the 4-momentum of the target
     *
     * @return a tuple of: inelastic cross section, elastic cross section
     */
    std::tuple<CrossSectionType, CrossSectionType> getCrossSectionInelEla(
        Code const projectile, Code const target, FourMomentum const& projectileP4,
        FourMomentum const& targetP4) const;

    /**
     * Returns inelastic (production) cross section.
     *
     * This cross section must correspond to the process described in doInteraction.
     * Allowed targets are: nuclei or single nucleons (p,n,hydrogen).
     *
     * @param projectile is the Code of the projectile
     * @param target is the Code of the target
     * @param projectileP4 is the 4-momentum of the projectile
     * @param targetP4 is the 4-momentum of the target
     *
     * @return inelastic cross section
     * elastic cross section
     */
    CrossSectionType getCrossSection(Code const projectile, Code const target,
                                     FourMomentum const& projectileP4,
                                     FourMomentum const& targetP4) const;

    /**
     * In this function PYTHIA8 is called to produce one event. The
     * event is copied (and boosted) into the reference frame defined by the input
     * 4-momenta.
     */
    template <typename TView>
    void doInteraction(TView& output, Code const projectileId, Code const targetId,
                       FourMomentum const& projectileP4, FourMomentum const& targetP4);

    using key_type = std::pair<corsika::Code, corsika::Code>;

  private:
    static auto constexpr GeV_mult = [](float x) { return x * 1_GeV; };
    static auto constexpr millibarn_mult = [](float x) { return x * 1_mb; };

    static CrossSectionTable<InterpolationTransforms::Log> loadPPTable(
        boost::filesystem::path const& dataPath, char const* key);

    default_prng_type& RNG_ = RNGManager<>::getInstance().getRandomStream("pythia");

    /**
     * All particles that we enable as projectiles. Code::Nucleus is not listed as it is
     * handled separately. The list is unlikely to be exhaustive, but should cover the
     * mainstream use-cases sufficiently well.
     */

    static std::array constexpr validProjectiles_ = {
        Code::PiPlus,       Code::PiMinus,   Code::Pi0,           Code::Proton,
        Code::AntiProton,   Code::Neutron,   Code::AntiNeutron,   Code::KPlus,
        Code::KMinus,       Code::K0Long,    Code::K0Short,       Code::SigmaMinus,
        Code::SigmaPlusBar, Code::SigmaPlus, Code::SigmaMinusBar, Code::Xi0,
        Code::Xi0Bar};

    static std::array constexpr validTargets_ = {
        Code::Proton, Code::Carbon, Code::Nitrogen, Code::Oxygen, Code::Argon};

    std::unordered_map<corsika::Code, corsika::Code> const xs_map_ = {
        {Code::SigmaMinus, Code::SigmaPlus},
        {Code::SigmaMinusBar, Code::SigmaPlus},
        {Code::SigmaPlusBar, Code::SigmaPlus},
        {Code::Xi0Bar, Code::Xi0}};

    std::unordered_map<int, int> const xs_nuc_map_ = {
        {2, 1},   {3, 2},   {4, 2},   {6, 3},   {7, 3},   {9, 4},   {10, 5},  {11, 5},
        {12, 6},  {13, 6},  {14, 7},  {15, 7},  {16, 8},  {17, 8},  {18, 8},  {19, 9},
        {20, 10}, {21, 10}, {22, 10}, {23, 11}, {24, 12}, {25, 12}, {26, 12}, {27, 13},
        {28, 14}, {29, 14}, {30, 14}, {31, 15}, {32, 16}, {33, 16}, {34, 16}, {35, 17},
        {37, 17}, {36, 18}, {38, 18}, {39, 19}, {41, 19}, {40, 20}, {42, 20}, {43, 20},
        {44, 20}, {45, 21}, {46, 22}, {47, 22}, {48, 22}, {49, 22}, {50, 23}, {51, 23},
        {52, 24}, {53, 24}, {55, 25}, {54, 26}, {56, 26}};

    Pythia8::Pythia pythia_;

    std::unordered_map<key_type, CrossSectionTable<InterpolationTransforms::Log>>
        crossSectionTables_;

    CrossSectionTable<InterpolationTransforms::Log> crossSectionPPElastic_,
        crossSectionPPInelastic_;

    bool const print_listing_ = false;
    HEPEnergyType const eMaxLab_ =
        1e21_eV; // Cross-section tables tabulated up to 10^21 eV
    HEPEnergyType const eKinMinLab_ = 100_GeV;
  };

} // namespace corsika::pythia8

#include <corsika/detail/modules/pythia8/InteractionModel.inl>
