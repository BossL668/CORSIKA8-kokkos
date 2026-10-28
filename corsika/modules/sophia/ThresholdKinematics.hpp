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
  inline constexpr double InternalPhotopionThresholdSGeV2 = 1.1646;

  Code internal_nucleon_code(Code target);

  double calculate_eventgen_target_beta(HEPEnergyType target_energy,
                                        HEPMassType sophia_mass);

  /**
   * Reproduce the centre-of-mass energy calculated inside eventgen.f.
   *
   * The public SOPHIA wrapper receives CORSIKA four-vectors, but eventgen
   * rebuilds the target at rest with the mass stored in its Fortran AM table.
   * Around threshold, checking only the norm of the CORSIKA four-vectors can
   * therefore accept a collision for which eventgen sets NP=0.
   */
  HEPEnergyType internal_com_energy(Code target, HEPEnergyType corsika_sqrt_s);

  /**
   * Express eventgen.f's exact threshold in the CORSIKA sqrt(s) coordinate
   * used by the hadronic-model dispatcher.
   */
  HEPEnergyType minimum_corsika_com_energy(Code target);

} // namespace corsika::sophia

#include <corsika/detail/modules/sophia/ThresholdKinematics.inl>
