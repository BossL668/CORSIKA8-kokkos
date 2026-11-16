/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/modules/sophia/ParticleConversion.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/modules/Random.hpp>
#include <corsika/modules/sophia/SophiaStack.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>

#include <sophia.hpp>

namespace corsika::sophia {

  inline void InteractionModel::setVerbose(bool const flag) { sophia_listing_ = flag; }

  inline InteractionModel::InteractionModel()
      : sophia_listing_(false) {
    corsika::connect_random_stream(RNG_, ::sophia::set_rng_function);
    // set all particles stable in SOPHIA
    for (int i = 0; i < 49; ++i) so_csydec_.idb[i] = -abs(so_csydec_.idb[i]);
  }

  inline InteractionModel::~InteractionModel() {
    CORSIKA_LOG_DEBUG("Sophia::Model n={}", count_);
  }

  inline bool constexpr InteractionModel::isValid(Code const projectileId,
                                                  Code const targetId,
                                                  HEPEnergyType const sqrtSnn) const {
    if ((minEnergyCoM_ > sqrtSnn) || (sqrtSnn > maxEnergyCoM_)) { return false; }

    if (!(targetId == Code::Proton || targetId == Code::Neutron ||
          targetId == Code::Hydrogen))
      return false;

    if (projectileId != Code::Photon) return false;

    return true;
  }

  template <typename TSecondaryView>
  inline void InteractionModel::doInteraction(TSecondaryView& secondaries,
                                              Code const projectileId,
                                              Code const targetId,
                                              FourMomentum const& projectileP4,
                                              FourMomentum const& targetP4) {

    CORSIKA_LOGGER_DEBUG(logger_, "projectile: Id={}, E={} GeV, p3={} GeV", projectileId,
                         projectileP4.getTimeLikeComponent() / 1_GeV,
                         projectileP4.getSpaceLikeComponents().getComponents() / 1_GeV);
    CORSIKA_LOGGER_DEBUG(logger_, "target: Id={}, E={} GeV, p3={} GeV", targetId,
                         targetP4.getTimeLikeComponent() / 1_GeV,
                         targetP4.getSpaceLikeComponents().getComponents() / 1_GeV);

    // sqrtS per target nucleon
    HEPEnergyType const sqrtS = (projectileP4 + targetP4).getNorm();

    CORSIKA_LOGGER_DEBUG(logger_, "sqrtS={}GeV", sqrtS / 1_GeV);

    // accepts only photon-nucleon interactions
    if (!isValid(projectileId, targetId, sqrtS)) {
      CORSIKA_LOGGER_ERROR(logger_,
                           "Invalid target/projectile/energy combination: {},{},{} GeV",
                           projectileId, targetId, sqrtS / 1_GeV);
      throw std::runtime_error("SOPHIA: Invalid target/projectile/energy combination");
    }

    COMBoost const boost(projectileP4, targetP4);

    int nucleonSophiaCode = convertToSophiaRaw(targetId); // either proton or neutron
    // initialize resonance spectrum
    initial_(nucleonSophiaCode);
    // Sophia does sqrt(1 - mass_sophia / mass_c8), so we need to make sure that E0 >=
    // m_sophia
    double Enucleon = std::max(corsika::sophia::getSophiaMass(targetId) / 1_GeV,
                               targetP4.getTimeLikeComponent() / 1_GeV);
    double Ephoton = projectileP4.getTimeLikeComponent() / 1_GeV;
    double theta = 0.0; // set nucleon at rest in collision
    int Imode = -1;     // overwritten inside SOPHIA
    CORSIKA_LOGGER_DEBUG(logger_,
                         "calling SOPHIA eventgen with L0={}, E0={}, eps={},theta={}",
                         nucleonSophiaCode, Enucleon, Ephoton, theta);
    count_++;
    // call sophia
    eventgen_(nucleonSophiaCode, Enucleon, Ephoton, theta, Imode);

    if (sophia_listing_) {
      int arg = 3;
      print_event_(arg);
    }

    auto const& originalCS = boost.getOriginalCS();
    // SOPHIA has photon along -z  and nucleon along +z (GZK calc..)
    COMBoost const boostInternal(targetP4, projectileP4);
    auto const& csPrime = boost.getRotatedCS();
    CoordinateSystemPtr csPrimePrime =
        make_rotation(csPrime, QuantityVector<length_d>{1_m, 0_m, 0_m}, M_PI);

    SophiaStack ss;

    MomentumVector P_final(originalCS, {0.0_GeV, 0.0_GeV, 0.0_GeV});
    HEPEnergyType E_final = 0_GeV;
    for (auto& psop : ss) {
      // abort on particles that have decayed in SOPHIA. Should not happen!
      if (psop.hasDecayed()) { // LCOV_EXCL_START
        throw std::runtime_error("found particle that decayed in SOPHIA!");
      } // LCOV_EXCL_STOP

      auto momentumSophia = psop.getMomentum(csPrimePrime);
      momentumSophia.rebase(csPrime);
      auto const energySophia = psop.getEnergy();
      auto const P4com = boostInternal.toCoM(FourVector{energySophia, momentumSophia});
      auto const P4lab = boost.fromCoM(P4com);
      SophiaCode const pidSophia = psop.getPID();
      Code const pid = convertFromSophia(pidSophia);
      auto momentum = P4lab.getSpaceLikeComponents();
      momentum.rebase(originalCS);
      HEPEnergyType const Ekin =
          calculate_kinetic_energy(momentum.getNorm(), get_mass(pid));

      CORSIKA_LOGGER_TRACE(logger_, "SOPHIA: pid={}, p={} GeV", pidSophia,
                           momentumSophia.getComponents() / 1_GeV);

      CORSIKA_LOGGER_TRACE(logger_, "CORSIKA: pid={}, p={} GeV", pid,
                           momentum.getComponents() / 1_GeV);

      auto pnew =
          secondaries.addSecondary(std::make_tuple(pid, Ekin, momentum.normalized()));

      P_final += pnew.getMomentum();
      E_final += pnew.getEnergy();
    }
    CORSIKA_LOGGER_TRACE(logger_, "Efinal={} GeV,Pfinal={} GeV", E_final / 1_GeV,
                         P_final.getComponents() / 1_GeV);
  }

} // namespace corsika::sophia
