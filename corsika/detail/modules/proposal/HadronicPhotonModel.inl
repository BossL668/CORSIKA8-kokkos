/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>

#include <tuple>
#include <random>

namespace corsika::proposal {

  template <typename THadronicLEModel, typename THadronicHEModel>
  inline HadronicPhotonModel<THadronicLEModel, THadronicHEModel>::HadronicPhotonModel(
      THadronicLEModel& _hadintLE, THadronicHEModel& _hadintHE,
      HEPEnergyType const& _heenthresholdNN)
      : leHadronicInteraction_(_hadintLE)
      , heHadronicInteraction_(_hadintHE)
      , heHadronicModelThresholdLabNN_(_heenthresholdNN) {
    // check validity of threshold assuming photon-nucleon
    // sqrtS per target nucleon
    HEPEnergyType const sqrtS =
        calculate_com_energy(_heenthresholdNN, Rho0::mass, Proton::mass);
    if (!heHadronicInteraction_.isValid(Code::Rho0, Code::Proton, sqrtS)) {
      CORSIKA_LOGGER_CRITICAL(
          logger_,
          "Invalid energy threshold for hadron interaction model. threshold_lab= {} GeV, "
          "threshold_com={} GeV",
          _heenthresholdNN / 1_GeV, sqrtS / 1_GeV);
      throw std::runtime_error("Configuration error!");
    }
    CORSIKA_LOGGER_DEBUG(
        logger_, "Threshold for HE hadronic interactions in proposal set to Elab={} GeV",
        _heenthresholdNN / 1_GeV);
  }

  template <typename THadronicLEModel, typename THadronicHEModel>
  template <typename TStackView>
  inline ProcessReturn
  HadronicPhotonModel<THadronicLEModel, THadronicHEModel>::doHadronicPhotonInteraction(
      TStackView& view, CoordinateSystemPtr const& labCS, FourMomentum const& photonP4,
      Code const& targetId) {

    //  temporarily add to stack, will be removed after interaction in DoInteraction
    typename TStackView::inner_stack_value_type photonStack;
    Point const pDummy(labCS, {0_m, 0_m, 0_m});
    TimeType const tDummy = 0_ns;
    // target at rest
    FourMomentum const targetP4(get_mass(targetId),
                                MomentumVector(labCS, {0_GeV, 0_GeV, 0_GeV}));
    auto hadronicPhoton = photonStack.addParticle(
        std::make_tuple(Code::Photon, photonP4.getTimeLikeComponent(),
                        photonP4.getSpaceLikeComponents().normalized(), pDummy, tDummy));
    hadronicPhoton.setNode(view.getProjectile().getNode());
    // create inelastic interaction of the hadronic photon
    // create new StackView for the photon
    TStackView photon_secondaries(hadronicPhoton);

    // call inner hadronic event generator
    CORSIKA_LOGGER_TRACE(logger_, "{} + {} interaction. Ekinlab = {} GeV", Code::Photon,
                         targetId, photonP4.getTimeLikeComponent() / 1_GeV);
    // check if had. model can handle configuration

    if (photonP4.getTimeLikeComponent() > heHadronicModelThresholdLabNN_) {
      CORSIKA_LOGGER_TRACE(logger_, "HE photo-hadronic interaction!");
      auto const sqrtSNN = (photonP4 + targetP4 / get_nucleus_A(targetId)).getNorm();
      CORSIKA_LOGGER_DEBUG(logger_, "sqrtS={} GeV", sqrtSNN / 1_GeV);
      // when Sibyll is used for hadronic interactions Argon cannot be used as target
      // nucleus. Since PROPOSAL has a non-zero cross section for Argon
      // targets we have to check here if the model can handle Argon (see Issue #498)
      if (!heHadronicInteraction_.isValid(Code::Rho0, targetId, sqrtSNN)) {
        CORSIKA_LOGGER_WARN(
            logger_,
            "HE interaction model cannot handle configuration in photo-hadronic "
            "interaction! projectile={}, target={} (A={}, Z={}), sqrt(S) per "
            "nuc.={:8.2f} "
            "GeV. Skipping secondary production!",
            Code::Rho0, targetId, get_nucleus_A(targetId), get_nucleus_Z(targetId),
            sqrtSNN / 1_GeV);
        return ProcessReturn::Ok;
      }
      heHadronicInteraction_.doInteraction(photon_secondaries, Code::Rho0, targetId,
                                           photonP4, targetP4);
    } else {
      CORSIKA_LOGGER_TRACE(logger_,
                           "LE photo-hadronic interaction! implemented via SOPHIA "
                           "assuming a single nucleon as target");
      // sample nucleon from nucleus A,Z
      double const fProtons = get_nucleus_Z(targetId) / double(get_nucleus_A(targetId));
      double const fNeutrons = 1. - fProtons;
      std::discrete_distribution<int> nucleonChannelDist{fProtons, fNeutrons};
      corsika::default_prng_type& rng =
          corsika::RNGManager<>::getInstance().getRandomStream("proposal");
      Code const nucleonId = (nucleonChannelDist(rng) ? Code::Neutron : Code::Proton);
      // target passed to SOPHIA needs to be exactly on-shell!
      FourMomentum const nucleonP4(get_mass(nucleonId),
                                   MomentumVector(labCS, {0_GeV, 0_GeV, 0_GeV}));
      CORSIKA_LOGGER_DEBUG(logger_,
                           "selected {} as target nucleon (f_proton, f_neutron)={},{}",
                           nucleonId, fProtons, fNeutrons);
      auto const sqrtSNN = (photonP4 + nucleonP4).getNorm();
      CORSIKA_LOGGER_DEBUG(logger_, "sqrtS={} GeV", sqrtSNN / 1_GeV);

      if (!leHadronicInteraction_.isValid(Code::Photon, nucleonId, sqrtSNN)) {
        CORSIKA_LOGGER_WARN(
            logger_,
            "LE interaction model cannot handle configuration in photo-hadronic "
            "interaction! projectile={}, target={} (A={}, Z={}), sqrt(S) per "
            "nuc.={:8.2f} "
            "GeV. Skipping secondary production!",
            Code::Photon, targetId, get_nucleus_A(targetId), get_nucleus_Z(targetId),
            sqrtSNN / 1_GeV);
        return ProcessReturn::Ok;
      }
      leHadronicInteraction_.doInteraction(photon_secondaries, Code::Photon, nucleonId,
                                           photonP4, nucleonP4);
    }
    for (const auto& pSec : photon_secondaries) {
      auto const p3lab = pSec.getMomentum();
      Code const pid = pSec.getPID();
      HEPEnergyType const secEkin =
          calculate_kinetic_energy(p3lab.getNorm(), get_mass(pid));
      view.addSecondary(std::make_tuple(pid, secEkin, p3lab.normalized()));
    }
    return ProcessReturn::Ok;
  }
} // namespace corsika::proposal
