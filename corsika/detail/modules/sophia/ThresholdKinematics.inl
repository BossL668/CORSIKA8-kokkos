/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::sophia {
  inline Code internal_nucleon_code(Code target) {
    if (target == Code::Hydrogen) { return Code::Proton; }
    if (target == Code::Proton || target == Code::Neutron) { return target; }
    throw std::invalid_argument(
        "SOPHIA threshold kinematics requires a proton or neutron target");
  }

  inline double calculate_eventgen_target_beta(HEPEnergyType target_energy,
                                               HEPMassType sophia_mass) {
    auto const gamma = target_energy / sophia_mass;
    if (!(gamma >= 1.)) { return 0.; }
    auto const inverse_gamma = 1. / gamma;
    if (gamma > 1000.) {
      auto const inverse_gamma_squared = inverse_gamma * inverse_gamma;
      return 1. - 0.5 * inverse_gamma_squared -
             0.125 * inverse_gamma_squared * inverse_gamma_squared;
    }
    return std::sqrt(1. - inverse_gamma) * std::sqrt(1. + inverse_gamma);
  }

  inline HEPEnergyType internal_com_energy(Code target, HEPEnergyType corsika_sqrt_s) {
    auto const corsika_mass = get_mass(target);
    auto const sophia_mass = getSophiaMass(internal_nucleon_code(target));
    if (!(corsika_mass > 0_eV) || corsika_sqrt_s < corsika_mass) { return 0_eV; }
    auto const photon_energy =
        (corsika_sqrt_s * corsika_sqrt_s - corsika_mass * corsika_mass) /
        (2. * corsika_mass);
    auto const eventgen_target_energy = std::max(sophia_mass, corsika_mass);
    auto const eventgen_target_beta =
        calculate_eventgen_target_beta(eventgen_target_energy, sophia_mass);
    auto const internal_s = sophia_mass * sophia_mass + 2. * photon_energy *
                                                            eventgen_target_energy *
                                                            (1. - eventgen_target_beta);
    if (!(internal_s > 0_eV * 0_eV)) { return 0_eV; }
    return sqrt(internal_s);
  }

  inline HEPEnergyType minimum_corsika_com_energy(Code target) {
    auto const corsika_mass = get_mass(target);
    auto const sophia_mass = getSophiaMass(internal_nucleon_code(target));
    auto const eventgen_target_energy = std::max(sophia_mass, corsika_mass);
    auto const eventgen_target_beta =
        calculate_eventgen_target_beta(eventgen_target_energy, sophia_mass);
    auto const threshold_s = InternalPhotopionThresholdSGeV2 * 1_GeV * 1_GeV;
    auto const photon_threshold =
        (threshold_s - sophia_mass * sophia_mass) /
        (2. * eventgen_target_energy * (1. - eventgen_target_beta));
    return sqrt(corsika_mass * corsika_mass + 2. * corsika_mass * photon_threshold);
  }

} // namespace corsika::sophia
