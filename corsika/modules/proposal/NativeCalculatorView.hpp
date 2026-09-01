/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

namespace PROPOSAL {
  class Displacement;
  class Interaction;
  class Medium;
  namespace crosssection {
    class PhotoPairLPM;
    class BremsLPM;
  }
}

namespace corsika::proposal {

  /**
   * Non-owning, read-only access to one calculator already constructed by the
   * scalar InteractionModel.  The pointers are valid for the lifetime of that
   * model and are never exposed to device code.
   */
  struct NativeInteractionCalculatorView {
    Code projectile{Code::Unknown};
    std::size_t medium_hash{};
    PROPOSAL::Medium const* medium{};
    PROPOSAL::Interaction const* interaction{};
    PROPOSAL::crosssection::PhotoPairLPM const* photon_pair_lpm{};
    PROPOSAL::crosssection::BremsLPM const* brems_lpm{};
    HEPEnergyType stochastic_energy_cut{};
    double particle_mass_MeV{};
  };

  /** Read-only view of the scalar continuous-loss displacement calculator. */
  struct NativeContinuousCalculatorView {
    Code projectile{Code::Unknown};
    std::size_t medium_hash{};
    PROPOSAL::Medium const* medium{};
    PROPOSAL::Displacement const* displacement{};
    HEPEnergyType stochastic_energy_cut{};
    double particle_mass_MeV{};
  };

} // namespace corsika::proposal
