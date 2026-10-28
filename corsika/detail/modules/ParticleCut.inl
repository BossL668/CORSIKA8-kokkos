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
      , cutElectrons_(eEleCut)
      , cutPhotons_(ePhoCut)
      , cutHadrons_(eHadCut)
      , cutMuons_(eMuCut)
      , cutTau_(eTauCut)
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
  inline void ParticleCut<TOutput>::recordCut(Code const pid,
                                              HEPEnergyType const kineticEnergy,
                                              double const weight,
                                              bool const restMassConverted) {
    auto const weightedKineticGeV = weight * kineticEnergy / 1_GeV;
    auto const weightedRestGeV = restMassConverted ? 0. : weight * get_mass(pid) / 1_GeV;
    ++statistics_.particles_;
    statistics_.weightedKineticEnergyGeV_ += weightedKineticGeV;
    statistics_.weightedRestMassEnergyGeV_ += weightedRestGeV;
    if (is_neutrino(pid)) {
      ++statistics_.invisibleParticles_;
      statistics_.weightedInvisibleKineticEnergyGeV_ += weightedKineticGeV;
    }
    auto& species = statistics_.byPdg_[static_cast<std::int32_t>(get_PDG(pid))];
    ++species.particles_;
    species.weightedKineticEnergyGeV_ += weightedKineticGeV;
    species.weightedRestMassEnergyGeV_ += weightedRestGeV;
  }

  template <typename TOutput>
  template <typename TParticle>
  inline bool ParticleCut<TOutput>::annihilateStoppedPositron(TParticle parent,
                                                              HEPEnergyType kinetic,
                                                              Point const& position,
                                                              TimeType time) {
    if constexpr (!EnableStoppedPositronAnnihilation) {
      return false;
    } else {
      // Discard the entire implementation when disabled: read-only particle
      // adapters used by output/cut tests do not provide addSecondary().
      // Time/geometry termination is not stopping physics. In particular, a
      // late positron must not start another cascade after the time limit.
      if (parent.getPID() != Code::Positron || time > 10_ms ||
          !isBelowEnergyCut(Code::Positron, kinetic))
        return false;
      auto const mass = get_mass(Code::Electron);
      auto const weight = parent.getWeight();
      ++statistics_.stoppedPositronAnnihilations_;
      statistics_.weightedMediumRestMassInputGeV_ += weight * mass / 1_GeV;
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
      positron_at_rest_direction(u, phi, d);
      // Copy all endpoint data before append: underlying SoA storage may grow.
      auto const endpoint = position;
      for (double sign : {1., -1.}) {
        auto photon = parent.addSecondary(
            std::make_tuple(Code::Photon, mass,
                            DirectionVector(endpoint.getCoordinateSystem(),
                                            {sign * d[0], sign * d[1], sign * d[2]})));
        photon.setPosition(endpoint);
        photon.setTime(time);
        photon.setWeight(weight); // no second thinning of a stopped parent
      }
      return true;
    }
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
      auto const annihilated =
          annihilateStoppedPositron(step.getParticlePre(), step.getEkinPost(),
                                    step.getPositionPost(), step.getTimePost());
      recordCut(step.getParticlePre().getPID(), step.getEkinPost(),
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
                      cutElectrons_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for photons is {} GeV",
                      cutPhotons_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for muons is {} GeV", cutMuons_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for tau is {} GeV", cutTau_ / 1_GeV);
    CORSIKA_LOG_DEBUG("kinetic energy threshold for hadrons is {} GeV",
                      cutHadrons_ / 1_GeV);

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
    node["cut_electrons"] = cutElectrons_ / 1_GeV;
    node["cut_photons"] = cutPhotons_ / 1_GeV;
    node["cut_muons"] = cutMuons_ / 1_GeV;
    node["cut_hadrons"] = cutHadrons_ / 1_GeV;
    node["cut_tau"] = cutTau_ / 1_GeV;
    node["cut_invisibles"] = doCutInv_;
    node["stopped_positron_annihilation"] = EnableStoppedPositronAnnihilation
                                                ? "two_photon_at_rest_v1"
                                                : "disabled_original_c8_cut";
    for (auto const& cut : cuts_) {
      node[fmt::format("cut_{}", cut.first)] = cut.second / 1_GeV;
    }
    return node;
  }

  template <typename TOutput>
  template <typename TParticle, typename TTrajectory>
  inline LengthType ParticleCut<TOutput>::getMaxStepLength(TParticle const&,
                                                           TTrajectory const&) {
    return meter * std::numeric_limits<double>::infinity();
  }

  template <typename TOutput>
  inline HEPEnergyType ParticleCut<TOutput>::getElectronKineticECut() const {
    return cutElectrons_;
  }

  template <typename TOutput>
  inline HEPEnergyType ParticleCut<TOutput>::getPhotonKineticECut() const {
    return cutPhotons_;
  }

  template <typename TOutput>
  inline HEPEnergyType ParticleCut<TOutput>::getMuonKineticECut() const {
    return cutMuons_;
  }

  template <typename TOutput>
  inline HEPEnergyType ParticleCut<TOutput>::getTauKineticECut() const {
    return cutTau_;
  }

  template <typename TOutput>
  inline HEPEnergyType ParticleCut<TOutput>::getHadronKineticECut() const {
    return cutHadrons_;
  }

  template <typename TOutput>
  inline ParticleCutStatistics const& ParticleCut<TOutput>::statistics() const noexcept {
    return statistics_;
  }

} // namespace corsika
