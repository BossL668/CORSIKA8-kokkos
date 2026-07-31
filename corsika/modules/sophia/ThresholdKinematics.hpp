/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/modules/sophia/ParticleConversion.hpp>

namespace corsika::sophia {

  // eventgen.f rejects s < 1.1646 GeV^2 after recomputing the collision with
  // SOPHIA's own nucleon mass table.
  inline constexpr double InternalPhotopionThresholdSGeV2 =
      1.1646;

  inline Code internalNucleonCode(Code target) {
    if (target == Code::Hydrogen) {
      return Code::Proton;
    }
    if (target == Code::Proton ||
        target == Code::Neutron) {
      return target;
    }
    throw std::invalid_argument(
        "SOPHIA threshold kinematics requires a proton or neutron target");
  }

  inline double eventgenTargetBeta(
      HEPEnergyType target_energy,
      HEPMassType sophia_mass) {
    auto const gamma = target_energy / sophia_mass;
    if (!(gamma >= 1.)) {
      return 0.;
    }
    auto const inverse_gamma = 1. / gamma;
    if (gamma > 1000.) {
      auto const inverse_gamma_squared =
          inverse_gamma * inverse_gamma;
      return 1. -
             0.5 * inverse_gamma_squared -
             0.125 * inverse_gamma_squared *
                 inverse_gamma_squared;
    }
    return std::sqrt(1. - inverse_gamma) *
           std::sqrt(1. + inverse_gamma);
  }

  /**
   * Reproduce the centre-of-mass energy calculated inside eventgen.f.
   *
   * The public SOPHIA wrapper receives CORSIKA four-vectors, but eventgen
   * rebuilds the target at rest with the mass stored in its Fortran AM table.
   * Around threshold, checking only the norm of the CORSIKA four-vectors can
   * therefore accept a collision for which eventgen sets NP=0.
   */
  inline HEPEnergyType internalComEnergy(
      Code target, HEPEnergyType corsika_sqrt_s) {
    auto const corsika_mass = get_mass(target);
    auto const sophia_mass =
        getSophiaMass(internalNucleonCode(target));
    if (!(corsika_mass > 0_eV) ||
        corsika_sqrt_s < corsika_mass) {
      return 0_eV;
    }
    auto const photon_energy =
        (corsika_sqrt_s * corsika_sqrt_s -
         corsika_mass * corsika_mass) /
        (2. * corsika_mass);
    auto const eventgen_target_energy =
        std::max(sophia_mass, corsika_mass);
    auto const eventgen_target_beta =
        eventgenTargetBeta(
            eventgen_target_energy, sophia_mass);
    auto const internal_s =
        sophia_mass * sophia_mass +
        2. * photon_energy * eventgen_target_energy *
            (1. - eventgen_target_beta);
    if (!(internal_s > 0_eV * 0_eV)) {
      return 0_eV;
    }
    return sqrt(internal_s);
  }

  /**
   * Express eventgen.f's exact threshold in the CORSIKA sqrt(s) coordinate
   * used by the hadronic-model dispatcher.
   */
  inline HEPEnergyType minimumCorsikaComEnergy(Code target) {
    auto const corsika_mass = get_mass(target);
    auto const sophia_mass =
        getSophiaMass(internalNucleonCode(target));
    auto const eventgen_target_energy =
        std::max(sophia_mass, corsika_mass);
    auto const eventgen_target_beta =
        eventgenTargetBeta(
            eventgen_target_energy, sophia_mass);
    auto const threshold_s =
        InternalPhotopionThresholdSGeV2 *
        1_GeV * 1_GeV;
    auto const photon_threshold =
        (threshold_s - sophia_mass * sophia_mass) /
        (2. * eventgen_target_energy *
         (1. - eventgen_target_beta));
    return sqrt(
        corsika_mass * corsika_mass +
        2. * corsika_mass * photon_threshold);
  }

} // namespace corsika::sophia
