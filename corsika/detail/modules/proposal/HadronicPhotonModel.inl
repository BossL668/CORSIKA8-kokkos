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
#include <algorithm>
#include <cmath>

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
    auto actual_target_energy = targetP4.getTimeLikeComponent();
    Code actual_target_id = targetId;
    bool const high_energy_interaction =
        photonP4.getTimeLikeComponent() >
        heHadronicModelThresholdLabNN_;
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

    if (high_energy_interaction) {
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
            "GeV. Aborting instead of discarding the selected final state!",
            Code::Rho0, targetId, get_nucleus_A(targetId), get_nucleus_Z(targetId),
            sqrtSNN / 1_GeV);
        throw std::runtime_error(
            "high-energy photo-hadronic final-state model rejected the "
            "selected projectile/target/energy configuration");
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
      actual_target_id = nucleonId;
      // target passed to SOPHIA needs to be exactly on-shell!
      FourMomentum const nucleonP4(get_mass(nucleonId),
                                   MomentumVector(labCS, {0_GeV, 0_GeV, 0_GeV}));
      actual_target_energy = nucleonP4.getTimeLikeComponent();
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
            "GeV. Aborting instead of discarding the selected final state!",
            Code::Photon, targetId, get_nucleus_A(targetId), get_nucleus_Z(targetId),
            sqrtSNN / 1_GeV);
        throw std::runtime_error(
            "low-energy photo-hadronic final-state model rejected the "
            "selected projectile/target/energy configuration");
      }
      leHadronicInteraction_.doInteraction(photon_secondaries, Code::Photon, nucleonId,
                                           photonP4, nucleonP4);
    }
    double weighted_secondary_energy_GeV = 0.;
    double weighted_secondary_baryon_number = 0.;
    double weighted_secondary_charge_number = 0.;
    std::vector<int> secondary_pdgs;
    MomentumVector secondary_momentum(
        labCS, {0_GeV, 0_GeV, 0_GeV});
    for (const auto& pSec : photon_secondaries) {
      auto const p3lab = pSec.getMomentum();
      secondary_momentum += p3lab;
      Code const pid = pSec.getPID();
      HEPEnergyType const secEkin =
          calculate_kinetic_energy(p3lab.getNorm(), get_mass(pid));
      auto added = view.addSecondary(
          std::make_tuple(pid, secEkin, p3lab.normalized()));
      secondary_pdgs.push_back(
          static_cast<int>(get_PDG(pid)));
      auto const pdg = static_cast<int>(get_PDG(pid));
      auto const absolute_pdg = std::abs(pdg);
      auto const baryon_number = is_nucleus(pid)
          ? static_cast<int>(get_nucleus_A(pid))
          : (absolute_pdg < 1000000000 &&
                     ((absolute_pdg / 1000) % 10) != 0
                 ? (pdg < 0 ? -1 : 1)
                 : 0);
      weighted_secondary_baryon_number +=
          baryon_number * added.getWeight();
      weighted_secondary_charge_number +=
          get_charge_number(pid) * added.getWeight();
      weighted_secondary_energy_GeV +=
          added.getEnergy() / 1_GeV * added.getWeight();
    }
    auto const projectile_weight =
        view.getProjectile().getWeight();
    auto const weighted_projectile_energy_GeV =
        photonP4.getTimeLikeComponent() / 1_GeV *
        projectile_weight;
    auto const weighted_target_energy_GeV =
        actual_target_energy / 1_GeV * projectile_weight;
    auto const residual_GeV =
        weighted_projectile_energy_GeV +
        weighted_target_energy_GeV -
        weighted_secondary_energy_GeV;
    auto const momentum_residual =
        photonP4.getSpaceLikeComponents() -
        secondary_momentum;
    auto const weighted_momentum_residual_norm_GeV =
        momentum_residual.getNorm() / 1_GeV *
        projectile_weight;
    auto const residual_mass_squared_GeV2 =
        residual_GeV * residual_GeV -
        weighted_momentum_residual_norm_GeV *
            weighted_momentum_residual_norm_GeV;
    constexpr double residual_tolerance_GeV = 1.e-9;
    auto const projectile_baryon_number = 0.;
    auto const target_baryon_number = is_nucleus(actual_target_id)
        ? static_cast<double>(get_nucleus_A(actual_target_id))
        : 1.;
    auto const weighted_baryon_number_residual =
        (projectile_baryon_number + target_baryon_number) *
            projectile_weight -
        weighted_secondary_baryon_number;
    auto const weighted_charge_number_residual =
        get_charge_number(actual_target_id) * projectile_weight -
        weighted_secondary_charge_number;
    ++energy_ledger_statistics_.interactions;
    if (high_energy_interaction) {
      ++energy_ledger_statistics_.high_energy_interactions;
    } else {
      ++energy_ledger_statistics_.low_energy_interactions;
    }
    energy_ledger_statistics_
        .weighted_projectile_total_energy_GeV +=
        weighted_projectile_energy_GeV;
    energy_ledger_statistics_
        .weighted_target_total_energy_GeV +=
        weighted_target_energy_GeV;
    energy_ledger_statistics_
        .weighted_secondary_total_energy_GeV +=
        weighted_secondary_energy_GeV;
    energy_ledger_statistics_.weighted_energy_residual_GeV +=
        residual_GeV;
    energy_ledger_statistics_
        .weighted_absolute_energy_residual_GeV +=
        std::abs(residual_GeV);
    energy_ledger_statistics_.maximum_absolute_residual_GeV =
        std::max(
            energy_ledger_statistics_.maximum_absolute_residual_GeV,
            std::abs(residual_GeV));
    energy_ledger_statistics_.maximum_target_total_energy_GeV =
        std::max(
            energy_ledger_statistics_.maximum_target_total_energy_GeV,
            weighted_target_energy_GeV);
    energy_ledger_statistics_
        .weighted_momentum_residual_norm_GeV +=
        weighted_momentum_residual_norm_GeV;
    energy_ledger_statistics_
        .maximum_momentum_residual_norm_GeV =
        std::max(
            energy_ledger_statistics_
                .maximum_momentum_residual_norm_GeV,
            weighted_momentum_residual_norm_GeV);
    if (residual_GeV < -residual_tolerance_GeV) {
      ++energy_ledger_statistics_
            .negative_energy_residual_interactions;
      if (-residual_GeV >
          energy_ledger_statistics_
              .maximum_negative_energy_residual_GeV) {
        energy_ledger_statistics_
            .maximum_negative_energy_residual_GeV =
            -residual_GeV;
        energy_ledger_statistics_
            .maximum_negative_projectile_pdg = 22;
        energy_ledger_statistics_
            .maximum_negative_target_pdg =
            static_cast<int>(get_PDG(actual_target_id));
        energy_ledger_statistics_
            .maximum_negative_projectile_total_energy_GeV =
            weighted_projectile_energy_GeV;
        energy_ledger_statistics_
            .maximum_negative_target_total_energy_GeV =
            weighted_target_energy_GeV;
        energy_ledger_statistics_
            .maximum_negative_secondary_total_energy_GeV =
            weighted_secondary_energy_GeV;
        energy_ledger_statistics_
            .maximum_negative_secondary_count =
            secondary_pdgs.size();
        energy_ledger_statistics_
            .maximum_negative_baryon_number_residual =
            weighted_baryon_number_residual;
        energy_ledger_statistics_
            .maximum_negative_charge_number_residual =
            weighted_charge_number_residual;
        energy_ledger_statistics_
            .maximum_negative_secondary_pdgs =
            std::move(secondary_pdgs);
      }
    } else {
      if (weighted_momentum_residual_norm_GeV >
          std::max(0., residual_GeV) +
              residual_tolerance_GeV) {
        ++energy_ledger_statistics_
              .spacelike_residual_interactions;
        energy_ledger_statistics_
            .maximum_spacelike_excess_GeV =
            std::max(
                energy_ledger_statistics_
                    .maximum_spacelike_excess_GeV,
                weighted_momentum_residual_norm_GeV -
                    residual_GeV);
      }
      if (residual_GeV > 0.) {
        energy_ledger_statistics_
            .maximum_positive_residual_beta =
            std::max(
                energy_ledger_statistics_
                    .maximum_positive_residual_beta,
                weighted_momentum_residual_norm_GeV /
                    residual_GeV);
      }
    }
    energy_ledger_statistics_
        .minimum_residual_mass_squared_GeV2 =
        std::min(
            energy_ledger_statistics_
                .minimum_residual_mass_squared_GeV2,
            residual_mass_squared_GeV2);
    if (residual_mass_squared_GeV2 >= 0. &&
        residual_GeV >= 0.) {
      energy_ledger_statistics_
          .maximum_timelike_residual_mass_GeV =
          std::max(
              energy_ledger_statistics_
                  .maximum_timelike_residual_mass_GeV,
              std::sqrt(residual_mass_squared_GeV2));
    }
    return ProcessReturn::Ok;
  }
} // namespace corsika::proposal
