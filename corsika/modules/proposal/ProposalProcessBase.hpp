/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <PROPOSAL/PROPOSAL.h>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/random/RNGManager.hpp>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace corsika::proposal {

  //!
  //! Particles which can be handled by proposal. That means they can be
  //! propagated and decayed if they decays.
  //!
  static constexpr std::array<Code, 7> tracked{
      Code::Photon, Code::Electron, Code::Positron, Code::MuMinus,
      Code::MuPlus, Code::TauPlus,  Code::TauMinus,
  };

  //!
  //! Relative EnergyCut value for PROPOSAL. Is defines that PROPOSAL will treat energy
  //! losses with a relative energy loss larger than v_cut stochastically.
  //!
  static constexpr double v_cut = 0.01;

  //!
  //! List of EnergyCut values for which CORSIKA will, by default, create and provide
  //! PROPOSAL tables.
  //!
  static constexpr std::array<HEPEnergyType, 10> energycut_table_values{
      1000_MeV, 100_MeV, 20_MeV,   10_MeV,   3_MeV,
      1_MeV,    0.4_MeV, 0.25_MeV, 0.15_MeV, 0.05_MeV};

  //!
  //! Internal map from particle codes to particle properties required for
  //! crosssections, decay and scattering algorithms. In the future the
  //! particles may be created by reading out the Corsica constants.
  //!
  static std::map<Code, PROPOSAL::ParticleDef> particle = {
      {Code::Photon, PROPOSAL::GammaDef()},   {Code::Electron, PROPOSAL::EMinusDef()},
      {Code::Positron, PROPOSAL::EPlusDef()}, {Code::MuMinus, PROPOSAL::MuMinusDef()},
      {Code::MuPlus, PROPOSAL::MuPlusDef()},  {Code::TauMinus, PROPOSAL::TauMinusDef()},
      {Code::TauPlus, PROPOSAL::TauPlusDef()}};

  //!
  //! Crosssection factories for different particle types.
  //!

  // Cross sections for muons and taus
  template <typename T>
  auto cross_builder = [](PROPOSAL::Medium& medium,
                          corsika::units::si::HEPEnergyType emCut,
                          bool const interpolate) {
    // Stochastic losses smaller than the given cut are handled continuously.
    auto particle_def = T();
    CORSIKA_LOG_DEBUG("PROPOSAL: Generate general cross sections for particle type {} ",
                      particle_def.name);
    auto p_cut =
        std::make_shared<const PROPOSAL::EnergyCutSettings>(emCut / 1_MeV, v_cut, false);

    // do not use v_cut for ionization due to numerical problems
    auto p_cut_no_vcut =
        std::make_shared<const PROPOSAL::EnergyCutSettings>(emCut / 1_MeV, 1, false);

    auto cross_vec = std::vector<std::shared_ptr<PROPOSAL::CrossSectionBase>>();
    cross_vec.push_back(PROPOSAL::make_crosssection(
        PROPOSAL::crosssection::BremsKelnerKokoulinPetrukhin{false}, particle_def, medium,
        p_cut, interpolate));
    cross_vec.push_back(PROPOSAL::make_crosssection(
        PROPOSAL::crosssection::EpairKelnerKokoulinPetrukhin{false}, particle_def, medium,
        p_cut, interpolate));
    cross_vec.push_back(PROPOSAL::make_crosssection(
        PROPOSAL::crosssection::IonizBetheBlochRossi{*p_cut_no_vcut}, particle_def,
        medium, p_cut_no_vcut, interpolate));
    cross_vec.push_back(PROPOSAL::make_crosssection(
        PROPOSAL::crosssection::PhotoAbramowiczLevinLevyMaor97{
            std::make_unique<PROPOSAL::crosssection::ShadowButkevichMikheyev>()},
        particle_def, medium, p_cut, interpolate));

    return cross_vec;
  };

  // cross sections for photons
  template <>
  auto cross_builder<PROPOSAL::GammaDef> =
      [](PROPOSAL::Medium& medium,
         corsika::units::si::HEPEnergyType,
         bool const interpolate) { //!< Only-stochastic propagation
        auto particle_def = PROPOSAL::GammaDef();
        CORSIKA_LOG_DEBUG(
            "PROPOSAL: Generate photon cross sections for particle type {} ",
            particle_def.name);

        auto cross_vec = std::vector<std::shared_ptr<PROPOSAL::CrossSectionBase>>();
        auto photopair =
            PROPOSAL::make_crosssection(PROPOSAL::crosssection::PhotoPairKochMotz{false},
                                        particle_def, medium, nullptr, interpolate);
        cross_vec.push_back(std::move(photopair));
        auto compton =
            PROPOSAL::make_crosssection(PROPOSAL::crosssection::ComptonKleinNishina{},
                                        particle_def, medium, nullptr, interpolate);
        cross_vec.push_back(std::move(compton));
        auto photoproduction = PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::PhotoproductionHeckC7Shadowing{}, particle_def,
            medium, nullptr, interpolate);
        cross_vec.push_back(std::move(photoproduction));
        auto photoeffect =
            PROPOSAL::make_crosssection(PROPOSAL::crosssection::PhotoeffectSauter{},
                                        particle_def, medium, nullptr, interpolate);
        cross_vec.push_back(std::move(photoeffect));
        auto muonpair = PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::PhotoMuPairBurkhardtKelnerKokoulin{}, particle_def,
            medium, nullptr, interpolate);
        cross_vec.push_back(std::move(muonpair));
        return cross_vec;
      };

  // cross sections for electrons
  template <>
  auto cross_builder<PROPOSAL::EMinusDef> =
      [](PROPOSAL::Medium& medium, corsika::units::si::HEPEnergyType emCut,
         bool const interpolate) {
        // Stochastic losses smaller than the given cut are handled continuously.
        auto particle_def = PROPOSAL::EMinusDef();
        CORSIKA_LOG_DEBUG(
            "PROPOSAL: Generate electron cross sections for particle type {} ",
            particle_def.name);
        auto p_cut = std::make_shared<const PROPOSAL::EnergyCutSettings>(emCut / 1_MeV,
                                                                         v_cut, false);

        // do not use v_cut for ionization due to numerical problems
        auto p_cut_no_vcut =
            std::make_shared<const PROPOSAL::EnergyCutSettings>(emCut / 1_MeV, 1, false);

        auto cross_vec = std::vector<std::shared_ptr<PROPOSAL::CrossSectionBase>>();
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::BremsElectronScreening{false}, particle_def, medium,
            p_cut, interpolate));
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::EpairForElectronPositron{false}, particle_def, medium,
            p_cut, interpolate));
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::IonizBergerSeltzerMoller{*p_cut_no_vcut},
            particle_def, medium, p_cut_no_vcut, interpolate));
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::PhotoAbramowiczLevinLevyMaor97{
                std::make_unique<PROPOSAL::crosssection::ShadowButkevichMikheyev>()},
            particle_def, medium, p_cut, interpolate));

        return cross_vec;
      };

  // cross sections for positrons
  template <>
  auto cross_builder<PROPOSAL::EPlusDef> =
      [](PROPOSAL::Medium& medium, corsika::units::si::HEPEnergyType emCut,
         bool const interpolate) {
        // Stochastic losses smaller than the given cut are handled continuously.
        auto particle_def = PROPOSAL::EPlusDef();
        CORSIKA_LOG_DEBUG(
            "PROPOSAL: Generate positron cross sections for particle type {} ",
            particle_def.name);
        auto p_cut = std::make_shared<const PROPOSAL::EnergyCutSettings>(emCut / 1_MeV,
                                                                         v_cut, false);

        // do not use v_cut for ionization due to numerical problems
        auto p_cut_no_vcut =
            std::make_shared<const PROPOSAL::EnergyCutSettings>(emCut / 1_MeV, 1, false);

        auto cross_vec = std::vector<std::shared_ptr<PROPOSAL::CrossSectionBase>>();
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::BremsElectronScreening{false}, particle_def, medium,
            p_cut, interpolate));
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::EpairForElectronPositron{false}, particle_def, medium,
            p_cut, interpolate));
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::IonizBergerSeltzerBhabha{*p_cut_no_vcut},
            particle_def, medium, p_cut_no_vcut, interpolate));
        cross_vec.push_back(PROPOSAL::make_crosssection(
            PROPOSAL::crosssection::PhotoAbramowiczLevinLevyMaor97{
                std::make_unique<PROPOSAL::crosssection::ShadowButkevichMikheyev>()},
            particle_def, medium, p_cut, interpolate));
        cross_vec.push_back(
            PROPOSAL::make_crosssection(PROPOSAL::crosssection::AnnihilationHeitler{},
                                        particle_def, medium, nullptr, interpolate));

        return cross_vec;
      };

  //!
  //! PROPOSAL default crosssections are maped to corresponding corsika particle
  //! code.
  //!
  inline PROPOSAL::crosssection_list_t make_cross_sections(
      Code const code, PROPOSAL::Medium& medium,
      corsika::units::si::HEPEnergyType const emCut,
      bool const interpolate) {
    switch (code) {
    case Code::Photon:
      return cross_builder<PROPOSAL::GammaDef>(medium, emCut, interpolate);
    case Code::Electron:
      return cross_builder<PROPOSAL::EMinusDef>(medium, emCut, interpolate);
    case Code::Positron:
      return cross_builder<PROPOSAL::EPlusDef>(medium, emCut, interpolate);
    case Code::MuMinus:
      return cross_builder<PROPOSAL::MuMinusDef>(medium, emCut, interpolate);
    case Code::MuPlus:
      return cross_builder<PROPOSAL::MuPlusDef>(medium, emCut, interpolate);
    case Code::TauMinus:
      return cross_builder<PROPOSAL::TauMinusDef>(medium, emCut, interpolate);
    case Code::TauPlus:
      return cross_builder<PROPOSAL::TauPlusDef>(medium, emCut, interpolate);
    default:
      throw std::invalid_argument(
          "PROPOSAL cannot build cross sections for this particle code");
    }
  }

  static std::map<Code, std::function<PROPOSAL::crosssection_list_t(
                            PROPOSAL::Medium&, corsika::units::si::HEPEnergyType)>>
      cross = {
          {Code::Photon,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::Photon, medium, cut, true);
           }},
          {Code::Electron,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::Electron, medium, cut, true);
           }},
          {Code::Positron,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::Positron, medium, cut, true);
           }},
          {Code::MuMinus,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::MuMinus, medium, cut, true);
           }},
          {Code::MuPlus,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::MuPlus, medium, cut, true);
           }},
          {Code::TauMinus,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::TauMinus, medium, cut, true);
           }},
          {Code::TauPlus,
           [](auto& medium, auto const cut) {
             return make_cross_sections(Code::TauPlus, medium, cut, true);
           }}};

  //!
  //! PROPOSAL base process which handels mapping of particle codes to
  //! stored interpolation tables.
  //!
  class ProposalProcessBase {
  protected:
    default_prng_type& RNG_ = RNGManager<>::getInstance().getRandomStream("proposal");

    std::unordered_map<std::size_t, PROPOSAL::Medium>
        media; //!< maps nuclear composition from univers to media to produce
               //!< crosssections, which requires further ionization constants.

    //!< save emcut for tracked particles
    std::unordered_map<Code, corsika::units::si::HEPEnergyType>
        proposal_energycutsettings;

    //!
    //! Store cut and  nuclear composition of the whole universe in media which are
    //! required for creating crosssections by proposal.
    //!
    template <typename TEnvironment>
    ProposalProcessBase(TEnvironment const& _env);

    //!
    //! Checks if a particle can be processed by proposal
    //!
    bool canInteract(Code pcode) const;

    //!
    //! Finds the optimal EnergyCut for which PROPOSAL tables should (by default) be
    //! available.
    //!
    HEPEnergyType getOptimizedEmCut(Code code) const;

    using calc_key_t = std::pair<std::size_t, Code>;

    //!
    //! Hash to store interpolation tables related to a pair of particle and nuclear
    //! composition.
    //!
    struct hash {
      size_t operator()(const calc_key_t& p) const noexcept;
    };

    //!
    //! Builds the calculator to the corresponding class
    //!
    virtual void buildCalculator(Code, size_t const&) = 0;

    //!
    //! Initialize PROPOSAL tables for given medium, code, and energy cut
    //!
    void buildTables(PROPOSAL::Medium, Code, HEPEnergyType);

    //!
    //! Searches the particle dependet calculator dependent of actuall medium composition
    //! and particle type. If no calculator is found, the corresponding new calculator is
    //! built and then returned.
    //!
    template <typename Particle, typename Calculators>
    auto getCalculator(Particle& vP, Calculators& calc) {
      const auto& comp = vP.getNode()->getModelProperties().getNuclearComposition();
      auto calc_it = calc.find(std::make_pair(comp.getHash(), vP.getPID()));
      if (calc_it != calc.end()) return calc_it;
      buildCalculator(vP.getPID(), comp.getHash());
      return getCalculator(vP, calc);
    }
  };
} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/ProposalProcessBase.inl>
