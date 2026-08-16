/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/modules/sophia/ParticleConversion.hpp>
#include <corsika/modules/sophia/ThresholdKinematics.hpp>
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
    // Broad resonances are generated with their sampled off-shell mass.  They
    // must decay before conversion to the on-shell CORSIKA stack; otherwise
    // rebuilding E from the sampled momentum and the nominal CORSIKA mass can
    // create O(10--100 MeV) at a single photo-hadronic vertex.  Keep long-lived
    // particles stable for normal CORSIKA transport, but restore SOPHIA decays
    // for rho/omega/phi-like states and the four Delta charge states.  The
    // indices below are C++ (zero-based) forms of SOPHIA IDB(25,26,27,32,33)
    // and IDB(40:43).
    for (int const i : {24, 25, 26, 31, 32, 39, 40, 41, 42}) {
      so_csydec_.idb[i] = abs(so_csydec_.idb[i]);
    }
  }

  inline InteractionModel::~InteractionModel() {
    CORSIKA_LOG_DEBUG("Sophia::Model n={}", count_);
  }

  inline bool InteractionModel::isValid(
      Code const projectileId, Code const targetId,
      HEPEnergyType const sqrtSnn) const {
    if (!(targetId == Code::Proton || targetId == Code::Neutron ||
          targetId == Code::Hydrogen))
      return false;

    if (projectileId != Code::Photon) return false;

    auto const internal_sqrt_s =
        internalComEnergy(targetId, sqrtSnn);
    return internal_sqrt_s >=
               sqrt(InternalPhotopionThresholdSGeV2) *
                   1_GeV &&
           internal_sqrt_s <= maxEnergyCoM_;
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

    auto const internal_target =
        internalNucleonCode(targetId);
    int nucleonSophiaCode =
        convertToSophiaRaw(internal_target);
    // initialize resonance spectrum
    initial_(nucleonSophiaCode);
    // Sophia does sqrt(1 - mass_sophia / mass_c8), so we need to make sure that E0 >=
    // m_sophia
    double Enucleon = std::max(corsika::sophia::getSophiaMass(internal_target) / 1_GeV,
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
    if (ss.getSize() == 0) {
      throw std::runtime_error(
          "SOPHIA returned an empty final state for a collision accepted by "
          "its C++ capability check");
    }

    std::vector<std::pair<Code, MomentumVector>> on_shell_candidates;
    on_shell_candidates.reserve(ss.getSize());
    for (auto& psop : ss) {
      // Decayed broad resonances remain in SOPHIA's event record with a
      // status offset.  Only their on-shell decay products belong on the
      // CORSIKA secondary stack.
      if (psop.hasDecayed()) { continue; }

      auto momentumSophia = psop.getMomentum(csPrimePrime);
      momentumSophia.rebase(csPrime);
      auto const energySophia = psop.getEnergy();
      auto const P4com = boostInternal.toCoM(FourVector{energySophia, momentumSophia});
      SophiaCode const pidSophia = psop.getPID();
      Code const pid = convertFromSophia(pidSophia);
      on_shell_candidates.emplace_back(
          pid, P4com.getSpaceLikeComponents());

      CORSIKA_LOGGER_TRACE(logger_, "SOPHIA: pid={}, p={} GeV", pidSophia,
                           momentumSophia.getComponents() / 1_GeV);

    }

    // Remove the small legacy-generator COM momentum residual with the
    // minimum equal-share correction before applying an energy scale.  This
    // prevents near-threshold scale factors from amplifying an O(MeV)
    // imbalance while preserving relative momenta as closely as possible.
    if (on_shell_candidates.empty()) {
      throw std::runtime_error(
          "SOPHIA returned no undecayed final-state particles");
    }
    MomentumVector mean_com_momentum(
        csPrime, {0.0_GeV, 0.0_GeV, 0.0_GeV});
    for (auto const& [pid, momentum] : on_shell_candidates) {
      static_cast<void>(pid);
      mean_com_momentum += momentum;
    }
    mean_com_momentum *=
        1. / static_cast<double>(on_shell_candidates.size());
    for (auto& [pid, momentum] : on_shell_candidates) {
      static_cast<void>(pid);
      momentum -= mean_com_momentum;
    }

    // SOPHIA samples broad-resonance masses.  After those resonances decay,
    // project the stable final state onto the CORSIKA nominal mass shell in
    // the collision COM frame.  A common momentum scale preserves the
    // generated directions and COM momentum closure while enforcing the
    // exact incoming sqrt(s).
    HEPEnergyType rest_mass_sum = 0_GeV;
    for (auto const& [pid, momentum] : on_shell_candidates) {
      static_cast<void>(momentum);
      rest_mass_sum += get_mass(pid);
    }
    if (rest_mass_sum > sqrtS + 1.e-9 * 1_GeV) {
      throw std::runtime_error(
          "SOPHIA final-state nominal masses exceed available COM energy");
    }
    auto const total_com_energy = [&](double const scale) {
      HEPEnergyType total = 0_GeV;
      for (auto const& [pid, momentum] : on_shell_candidates) {
        total += calculate_total_energy(
            momentum.getNorm() * scale, get_mass(pid));
      }
      return total;
    };
    double lower_scale = 0.;
    double upper_scale = 1.;
    while (total_com_energy(upper_scale) < sqrtS) {
      upper_scale *= 2.;
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
      auto const middle = 0.5 * (lower_scale + upper_scale);
      if (total_com_energy(middle) < sqrtS) {
        lower_scale = middle;
      } else {
        upper_scale = middle;
      }
    }
    auto const on_shell_scale =
        0.5 * (lower_scale + upper_scale);

    MomentumVector P_final(originalCS, {0.0_GeV, 0.0_GeV, 0.0_GeV});
    HEPEnergyType E_final = 0_GeV;
    for (auto const& [pid, raw_com_momentum] :
         on_shell_candidates) {
      auto const com_momentum =
          raw_com_momentum * on_shell_scale;
      auto const com_energy = calculate_total_energy(
          com_momentum.getNorm(), get_mass(pid));
      auto const P4lab = boost.fromCoM(
          FourVector{com_energy, com_momentum});
      auto momentum = P4lab.getSpaceLikeComponents();
      momentum.rebase(originalCS);
      HEPEnergyType const Ekin =
          calculate_kinetic_energy(
              momentum.getNorm(), get_mass(pid));
      CORSIKA_LOGGER_TRACE(logger_, "CORSIKA: pid={}, p={} GeV", pid,
                           momentum.getComponents() / 1_GeV);
      auto pnew = secondaries.addSecondary(
          std::make_tuple(pid, Ekin, momentum.normalized()));
      P_final += pnew.getMomentum();
      E_final += pnew.getEnergy();
    }
    CORSIKA_LOGGER_TRACE(logger_, "Efinal={} GeV,Pfinal={} GeV", E_final / 1_GeV,
                         P_final.getComponents() / 1_GeV);
  }

} // namespace corsika::sophia
