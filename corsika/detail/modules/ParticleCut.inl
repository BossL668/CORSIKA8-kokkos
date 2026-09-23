/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/utility/PositronAtRest.hpp>
#include <random>

namespace corsika {

  template <typename TOutput>
  template <typename... TArgs>
  inline ParticleCut<TOutput>::ParticleCut(HEPEnergyType const eEleCut,
                                           HEPEnergyType const ePhoCut,
                                           HEPEnergyType const eHadCut,
                                           HEPEnergyType const eMuCut,
                                           HEPEnergyType const eTauCut, bool const inv,
                                           TArgs&&... outputArgs)
      : TOutput(std::forward<TArgs>(outputArgs)...)
      , cut_electrons_(eEleCut)
      , cut_photons_(ePhoCut)
      , cut_hadrons_(eHadCut)
      , cut_muons_(eMuCut)
      , cut_tau_(eTauCut)
      , doCutInv_(inv) {
    for (auto const p : get_all_particles()) {
      if (is_hadron(p)) // nuclei are also hadrons
        set_kinetic_energy_propagation_threshold(p, eHadCut);
      else if (is_muon(p))
        set_kinetic_energy_propagation_threshold(p, eMuCut);
      else if (p == Code::TauMinus || p == Code::TauPlus)
        set_kinetic_energy_propagation_threshold(p, eTauCut);
      else if (p == Code::Electron || p == Code::Positron)
        set_kinetic_energy_propagation_threshold(p, eEleCut);
      else if (p == Code::Photon)
        set_kinetic_energy_propagation_threshold(p, ePhoCut);
    }
    set_kinetic_energy_propagation_threshold(Code::Nucleus, eHadCut);
    CORSIKA_LOG_DEBUG(
        "setting kinetic energy thresholds: electrons = {} GeV, photons = {} GeV, "
        "hadrons = {} GeV, "
        "muons = {} GeV",
        "tau = {} GeV", eEleCut / 1_GeV, ePhoCut / 1_GeV, eHadCut / 1_GeV, eMuCut / 1_GeV,
        eTauCut / 1_GeV);
  }

  template <typename TOutput>
  template <typename... TArgs>
  inline ParticleCut<TOutput>::ParticleCut(HEPEnergyType const eCut, bool const inv,
                                           TArgs&&... outputArgs)
      : TOutput(std::forward<TArgs>(outputArgs)...)
      , doCutInv_(inv) {
    for (auto p : get_all_particles()) {
      set_kinetic_energy_propagation_threshold(p, eCut);
    }
    set_kinetic_energy_propagation_threshold(Code::Nucleus, eCut);
    CORSIKA_LOG_DEBUG("setting kinetic energy threshold {} GeV", eCut / 1_GeV);
  }

  template <typename TOutput>
  template <typename... TArgs>
  inline ParticleCut<TOutput>::ParticleCut(
      std::unordered_map<Code const, HEPEnergyType const> const& eCuts, bool const inv,
      TArgs&&... args)
      : TOutput(std::forward<TArgs>(args)...)
      , doCutInv_(inv) {
    for (auto const& cut : eCuts) {
      set_kinetic_energy_propagation_threshold(cut.first, cut.second);
    }
    CORSIKA_LOG_DEBUG("setting threshold particles individually");
  }

  template <typename TOutput>
  inline bool ParticleCut<TOutput>::isBelowEnergyCut(
      Code const pid, HEPEnergyType const energyLab) const {
    // nuclei
    if (is_nucleus(pid)) {
      // calculate energy per nucleon
      auto const ElabNuc = energyLab / get_nucleus_A(pid);
      return (ElabNuc < get_kinetic_energy_propagation_threshold(pid));
    } else {
      return (energyLab < get_kinetic_energy_propagation_threshold(pid));
    }
  }

  template <typename TOutput>
  inline bool ParticleCut<TOutput>::checkCutParticle(Code const pid,
                                                     HEPEnergyType const kine_energy,
                                                     TimeType const timePost) const {

    HEPEnergyType const energy = kine_energy + get_mass(pid);
    CORSIKA_LOG_DEBUG(
        "ParticleCut: checking {} ({}), E_kin= {} GeV, E={} GeV, m={} "
        "GeV",
        pid, get_PDG(pid), kine_energy / 1_GeV, energy / 1_GeV, get_mass(pid) / 1_GeV);
    if (doCutInv_ && is_neutrino(pid)) {
      CORSIKA_LOG_DEBUG("removing inv. particle...");
      return true;
    } else if (isBelowEnergyCut(pid, kine_energy)) {
      CORSIKA_LOG_DEBUG("removing low en. particle...");
      return true;
    } else if (timePost > 10_ms) {
      CORSIKA_LOG_DEBUG("removing OLD particle...");
      return true;
    } else {
      for (auto const& cut : cuts_) {
        if (pid == cut.first && kine_energy < cut.second) { return true; }
      }
    }
    return false; // this particle will not be removed/cut
  }

  template <typename TOutput>
  inline void ParticleCut<TOutput>::recordCut(
      Code const pid, HEPEnergyType const kinetic_energy,
      double const weight, bool const rest_mass_converted) {
    auto const weighted_kinetic_GeV =
        weight * kinetic_energy / 1_GeV;
    auto const weighted_rest_GeV =
        rest_mass_converted ? 0. : weight * get_mass(pid) / 1_GeV;
    ++statistics_.particles;
    statistics_.weighted_kinetic_energy_GeV +=
        weighted_kinetic_GeV;
    statistics_.weighted_rest_mass_energy_GeV +=
        weighted_rest_GeV;
    if (is_neutrino(pid)) {
      ++statistics_.invisible_particles;
      statistics_.weighted_invisible_kinetic_energy_GeV +=
          weighted_kinetic_GeV;
    }
    auto& species = statistics_.by_pdg[
        static_cast<std::int32_t>(get_PDG(pid))];
    ++species.particles;
    species.weighted_kinetic_energy_GeV +=
        weighted_kinetic_GeV;
    species.weighted_rest_mass_energy_GeV +=
        weighted_rest_GeV;
  }

  template <typename TOutput>
  template <typename TParticle>
  inline bool ParticleCut<TOutput>::annihilateStoppedPositron(
      TParticle parent, HEPEnergyType kinetic, Point const& position, TimeType time) {
    // Time/geometry termination is not stopping physics. In particular, a
    // late positron must not start another cascade after the time limit.
    if (parent.getPID() != Code::Positron || time > 10_ms ||
        !isBelowEnergyCut(Code::Positron, kinetic)) return false;
    auto const mass = get_mass(Code::Electron);
    auto const weight = parent.getWeight();
    ++statistics_.stopped_positron_annihilations;
    statistics_.weighted_medium_rest_mass_input_GeV += weight * mass / 1_GeV;
    if (isBelowEnergyCut(Code::Photon, mass)) {
      for (int i = 0; i < 2; ++i) {
        recordCut(Code::Photon, mass, weight);
        this->write(position, Code::Photon, weight * mass);
      }
      return true;
    }
    auto& rng = RNGManager<>::getInstance().getRandomStream("cascade");
    std::uniform_real_distribution<double> uniform(0., 1.);
    // Separate statements make random consumption order explicit in C++17.
    auto const u = uniform(rng);
    auto const phi = uniform(rng);
    double d[3];
    positronAtRestDirection(u, phi, d);
    // Copy all endpoint data before append: underlying SoA storage may grow.
    auto const endpoint = position;
    for (double sign : {1., -1.}) {
      auto photon = parent.addSecondary(std::make_tuple(Code::Photon, mass,
          DirectionVector(endpoint.getCoordinateSystem(), {sign*d[0], sign*d[1], sign*d[2]})));
      photon.setPosition(endpoint);
      photon.setTime(time);
      photon.setWeight(weight); // no second thinning of a stopped parent
    }
    return true;
  }

  template <typename TOutput>
  template <typename TStackView>
  inline void ParticleCut<TOutput>::doSecondaries(TStackView& vS) {
    HEPEnergyType energy_event = 0_GeV; // per event counting for printout
    auto particle = vS.begin();
    while (particle != vS.end()) {
      Code pid = particle.getPID();
      HEPEnergyType Ekin = particle.getKineticEnergy();
      if (checkCutParticle(pid, Ekin, particle.getTime())) {
        auto const annihilated = annihilateStoppedPositron(
            particle, Ekin, particle.getPosition(), particle.getTime());
        recordCut(pid, Ekin, particle.getWeight(), annihilated);
        this->write(particle.getPosition(), pid, particle.getWeight() * Ekin);
        particle.erase();
      }
      ++particle; // next entry in SecondaryView
    }
    CORSIKA_LOG_DEBUG("Event cut: {} GeV", energy_event / 1_GeV);
  }

  template <typename TOutput>
  template <typename TParticle>
  inline ProcessReturn ParticleCut<TOutput>::doContinuous(Step<TParticle>& step,
                                                          bool const) {
    if (checkCutParticle(step.getParticlePre().getPID(), step.getEkinPost(),
                         step.getTimePost())) {
      auto const annihilated = annihilateStoppedPositron(
          step.getParticlePre(), step.getEkinPost(), step.getPositionPost(), step.getTimePost());
      recordCut(
          step.getParticlePre().getPID(), step.getEkinPost(),
          step.getParticlePre().getWeight(), annihilated);
      this->write(
          step.getPositionPost(), step.getParticlePre().getPID(),
          step.getParticlePre().getWeight() *
              step.getEkinPost()); // ToDO: should the cut happen at the start of the
                                   // track? For now, I set it to happen at the start
      CORSIKA_LOG_TRACE("removing during continuous");
      // signal to upstream code that this particle was deleted
      return ProcessReturn::ParticleAbsorbed;
    }
    return ProcessReturn::Ok;
  }

  template <typename TOutput>
  inline void ParticleCut<TOutput>::printThresholds() const {

    CORSIKA_LOG_DEBUG("kinetic energy threshold for electrons is {} GeV",
                      cut_electrons_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for photons is {} GeV",
                      cut_photons_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for muons is {} GeV", cut_muons_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for tau is {} GeV", cut_tau_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for hadrons is {} GeV",
                      cut_hadrons_ / 1_GeV);

    for (auto const& cut : cuts_) {
      CORSIKA_LOG_DEBUG("kinetic energy threshold for particle {} is {} GeV", cut.first,
                        cut.second / 1_GeV);
    }
  }

  template <typename TOutput>
  inline YAML::Node ParticleCut<TOutput>::getConfig() const {

    YAML::Node node;
    node["type"] = "ParticleCut";
    node["units"]["energy"] = "GeV";
    node["cut_electrons"] = cut_electrons_ / 1_GeV;
    node["cut_photons"] = cut_photons_ / 1_GeV;
    node["cut_muons"] = cut_muons_ / 1_GeV;
    node["cut_hadrons"] = cut_hadrons_ / 1_GeV;
    node["cut_tau"] = cut_tau_ / 1_GeV;
    node["cut_invisibles"] = doCutInv_;
    node["stopped_positron_annihilation"] = "two_photon_at_rest_v1";
    for (auto const& cut : cuts_) {
      node[fmt::format("cut_{}", cut.first)] = cut.second / 1_GeV;
    }
    return node;
  }

} // namespace corsika
