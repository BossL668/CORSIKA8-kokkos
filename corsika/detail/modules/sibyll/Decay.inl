/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/sibyll/Decay.hpp>
#include <corsika/modules/sibyll/ParticleConversion.hpp>
#include <corsika/modules/sibyll/SibStack.hpp>
#include <corsika/modules/Random.hpp>

#include <iostream>
#include <vector>

namespace corsika::sibyll {

  inline Decay::Decay(const bool sibyll_printout_on)
      : sibyll_listing_(sibyll_printout_on) {
    corsika::connect_random_stream("sibyll", ::sibyll23d::set_rng_function);
    // switch off decays to avoid internal decay chains
    setAllStable();
    // handle all decays by default
    handleAllDecays_ = true;
  }

  inline Decay::Decay(std::set<Code> const& vHandled)
      : handleAllDecays_(false)
      , handledDecays_(vHandled) {
    setAllStable();
  }

  inline Decay::~Decay() {
    CORSIKA_LOGGER_TRACE(logger_, "Total number of Sibyll decays n={}", count_);
  }

  inline bool Decay::canHandleDecay(const Code vParticleCode) {
    // if known to sibyll and not proton or neutrino it can decay
    if (is_nucleus(vParticleCode) || vParticleCode == Code::Proton ||
        vParticleCode == Code::AntiProton || vParticleCode == Code::NuE ||
        vParticleCode == Code::NuMu || vParticleCode == Code::NuTau ||
        vParticleCode == Code::NuEBar || vParticleCode == Code::NuMuBar ||
        vParticleCode == Code::NuTauBar || vParticleCode == Code::Electron ||
        vParticleCode == Code::Positron)
      return false;
    else if (corsika::sibyll::convertToSibyllRaw(
                 vParticleCode)) // non-zero for particles known to sibyll
      return true;
    else
      return false;
  }

  inline void Decay::setHandleDecay(const Code vParticleCode) {
    handleAllDecays_ = false;
    CORSIKA_LOGGER_DEBUG(logger_, "set to handle decay of {}", vParticleCode);
    if (Decay::canHandleDecay(vParticleCode))
      handledDecays_.insert(vParticleCode);
    else
      throw std::runtime_error("this decay can not be handled by sibyll!");
  }

  inline void Decay::setHandleDecay(std::vector<Code> const& vParticleList) {
    handleAllDecays_ = false;
    for (auto const p : vParticleList) Decay::setHandleDecay(p);
  }

  inline bool Decay::isDecayHandled(corsika::Code const vParticleCode) {
    if (handleAllDecays_ && Decay::canHandleDecay(vParticleCode))
      return true;
    else
      return Decay::handledDecays_.find(vParticleCode) != Decay::handledDecays_.end()
                 ? true
                 : false;
  }

  inline void Decay::setStable(std::vector<Code> const& vParticleList) {
    for (auto const p : vParticleList) Decay::setStable(p);
  }

  inline void Decay::setUnstable(std::vector<Code> const& vParticleList) {
    for (auto const p : vParticleList) Decay::setUnstable(p);
  }

  inline bool Decay::isStable(Code const vCode) {
    return abs(sibyll::convertToSibyllRaw(vCode)) <= 0 ? true : false;
  }

  inline bool Decay::isUnstable(Code const vCode) {
    return abs(sibyll::convertToSibyllRaw(vCode)) > 0 ? true : false;
  }

  inline void Decay::setDecay(const Code vCode, const bool vMakeUnstable) {
    vMakeUnstable ? setUnstable(vCode) : setStable(vCode);
  }

  inline void Decay::setUnstable(Code const vCode) {
    CORSIKA_LOGGER_DEBUG(logger_, "setting {} as \"unstable\". ", vCode);

    const int s_id = abs(sibyll::convertToSibyllRaw(vCode));
    ::sibyll23d::s_csydec_->idb[s_id - 1] = abs(::sibyll23d::s_csydec_->idb[s_id - 1]);
  }

  inline void Decay::setStable(Code const vCode) {
    CORSIKA_LOGGER_DEBUG(logger_, "setting {} as \"stable\". ", vCode);

    const int s_id = abs(sibyll::convertToSibyllRaw(vCode));
    ::sibyll23d::s_csydec_->idb[s_id - 1] =
        (-1) * abs(::sibyll23d::s_csydec_->idb[s_id - 1]);
  }

  inline void Decay::setAllStable() {
    for (int i = 0; i < 99; ++i)
      ::sibyll23d::s_csydec_->idb[i] = -1 * abs(::sibyll23d::s_csydec_->idb[i]);
  }

  inline void Decay::setAllUnstable() {
    for (int i = 0; i < 99; ++i)
      ::sibyll23d::s_csydec_->idb[i] = abs(::sibyll23d::s_csydec_->idb[i]);
  }

  inline void Decay::printDecayConfig([[maybe_unused]] const Code vCode) {
    [[maybe_unused]] const int sibCode = corsika::sibyll::convertToSibyllRaw(vCode);
    [[maybe_unused]] const int absSibCode = abs(sibCode);
    CORSIKA_LOGGER_DEBUG(
        logger_, "decay configuration: {} is \"{}\"", vCode,
        (::sibyll23d::s_csydec_->idb[absSibCode - 1] <= 0) ? "stable" : "unstable");
  }
  inline void Decay::printDecayConfig() {
    CORSIKA_LOGGER_DEBUG(logger_, "decay configuration:");
    if (handleAllDecays_) {
      CORSIKA_LOGGER_DEBUG(
          logger_, "     all particles known to Sibyll are handled by Sibyll::Decay!");

    } else {
      for ([[maybe_unused]] auto const pCode : handledDecays_) {
        CORSIKA_LOGGER_DEBUG(logger_, "      Decay of {} is handled by Sibyll!", pCode);
      }
    }
  }

  template <typename TParticle>
  inline TimeType Decay::getLifetime(TParticle const& projectile) {

    const Code pid = projectile.getPID();
    if (Decay::isDecayHandled(pid)) {
      HEPEnergyType const E = projectile.getEnergy();
      HEPMassType const m = projectile.getMass();
      const double gamma = E / m;
      const TimeType t0 = get_lifetime(projectile.getPID());
      auto const lifetime = gamma * t0;
      [[maybe_unused]] const auto mkin =
          +(E * E - projectile.getMomentum()
                        .getSquaredNorm()); // delta_mass(projectile.getMomentum(), E, m);
      CORSIKA_LOGGER_DEBUG(logger_, "code: {} ", projectile.getPID());
      CORSIKA_LOGGER_DEBUG(logger_, "MinStep: t0: {} ", t0);
      CORSIKA_LOGGER_DEBUG(logger_, "MinStep: energy: {} GeV ", E / 1_GeV);
      CORSIKA_LOGGER_DEBUG(logger_, "momentum: {} GeV ",
                           projectile.getMomentum().getComponents() / 1_GeV);
      CORSIKA_LOGGER_DEBUG(logger_, "momentum: shell mass-kin. inv. mass {} {}",
                           mkin / 1_GeV / 1_GeV, m / 1_GeV * m / 1_GeV);
      [[maybe_unused]] auto sib_id =
          corsika::sibyll::convertToSibyllRaw(projectile.getPID());
      CORSIKA_LOGGER_DEBUG(logger_, "sib mass: {}",
                           ::sibyll23d::get_sibyll_mass2(sib_id));
      CORSIKA_LOGGER_DEBUG(logger_, "MinStep: gamma:  {}", gamma);
      CORSIKA_LOGGER_DEBUG(logger_, "MinStep: tau {} s: ", lifetime / 1_s);
      return lifetime;
    }
    return std::numeric_limits<double>::infinity() * 1_s;
  } // namespace corsika::sibyll

  template <typename TSecondaryView>
  inline void Decay::doDecay(TSecondaryView& view) {

    auto projectile = view.getProjectile();
    const Code pCode = projectile.getPID();
    // check if sibyll is configured to handle this decay!
    if (!isDecayHandled(pCode))
      throw std::runtime_error("STOP! Sibyll not configured to execute this decay!");

    count_++;

    // switch on decay for this particle
    setUnstable(pCode);
    printDecayConfig(pCode);

    // call sibyll decay
    CORSIKA_LOGGER_DEBUG(logger_, "calling Sibyll decay routine..");

    // particle to pass to sibyll decay
    int inputSibPID = sibyll::convertToSibyllRaw(pCode);

    // particle momentum format: px, py, pz, e, mass. units: GeV
    double inputMomentum[5];
    QuantityVector<hepmomentum_d> input_components =
        projectile.getMomentum().getComponents();
    for (int idx = 0; idx < 3; ++idx) inputMomentum[idx] = input_components[idx] / 1_GeV;

    inputMomentum[3] = projectile.getEnergy() / 1_GeV;
    inputMomentum[4] = get_mass(pCode) / 1_GeV;
    int nFinalParticles;

    double outputMomentum[5 * 10];
    int outputSibPID[10];

    // run decay routine
    ::sibyll23d::decpar_sib_(inputSibPID, inputMomentum, nFinalParticles, outputSibPID,
                             &outputMomentum[0]);

    CORSIKA_LOGGER_TRACE(logger_, "number of final state particles: {}", nFinalParticles);

    // reset to stable
    setStable(pCode);

    CoordinateSystemPtr const& rootCS = get_root_CoordinateSystem();

    // copy particles from sibyll ministack to corsika
    for (int i = 0; i < nFinalParticles; ++i) {
      QuantityVector<hepmomentum_d> components = {outputMomentum[10 * 0 + i] * 1_GeV,
                                                  outputMomentum[10 * 1 + i] * 1_GeV,
                                                  outputMomentum[10 * 2 + i] * 1_GeV};

      auto const pid = sibyll::convertFromSibyll(
          static_cast<corsika::sibyll::SibyllCode>(outputSibPID[i]));

      CORSIKA_LOGGER_TRACE(logger_, "final particle i={} id={} p={} GeV", i, pid,
                           components / 1_GeV);

      auto const p3 = MomentumVector(rootCS, components);
      HEPEnergyType const mass = get_mass(pid);
      HEPEnergyType const Ekin = sqrt(p3.getSquaredNorm() + mass * mass) - mass;

      projectile.addSecondary(std::make_tuple(pid, Ekin, p3.normalized()));
    }
  }

} // namespace corsika::sibyll
