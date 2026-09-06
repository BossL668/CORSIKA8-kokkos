/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/accelerator/em/common/tables/Sha256.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace corsika::gpu::em::tables {

  // Matches scalar ContinuousProcess stopping just below ParticleCut.
  inline constexpr double ContinuousCutSafetyFactor = 0.9999;

  /**
   * PROPOSAL PhotoPairLPM parameters frozen into the physics cache.
   *
   * These values are deliberately stored instead of recomputed in CUDA code:
   * e_lpm_MeV depends on the exact PROPOSAL parametrization and a numerical
   * integral over the complete medium. The component data are the two target
   * quantities used by PhotoPairLPM::suppression_factor().
   */
  struct PhotonPairLpmComponent {
    std::uint64_t proposal_hash{};
    double nuclear_charge{};
    double radiation_log_constant{};
  };

  struct PhotonPairLpmMetadata {
    double baseline_mass_density_g_per_cm3{};
    double molecular_density_per_cm3{};
    double sum_charge{};
    double e_lpm_MeV{};
    double classical_electron_radius_cm{};
    double fine_structure_constant{};
    std::vector<PhotonPairLpmComponent> components;
  };

  /**
   * PROPOSAL BremsLPM parameters frozen into the physics cache.
   *
   * BremsLPM::eLpm_ is obtained from a numerical integral over the complete
   * medium and has no public accessor in PROPOSAL 7.6.2.  The remaining
   * fields are kept explicitly so the CUDA implementation is a direct
   * translation of the reference formula rather than a fit.
   */
  struct BremsLpmComponent {
    std::uint64_t proposal_hash{};
    double nuclear_charge{};
    // PROPOSAL::Component::GetAtomicNum(); despite its historical name this
    // is the target atomic mass used in D_n = 1.54 A^0.27.
    double atomic_mass_number{};
    double radiation_log_constant{};
  };

  struct BremsLpmMetadata {
    double baseline_mass_density_g_per_cm3{};
    double molecular_density_per_cm3{};
    double sum_charge{};
    double e_lpm_MeV{};
    double lepton_mass_MeV{};
    double electron_mass_MeV{};
    double muon_mass_MeV{};
    double classical_electron_radius_cm{};
    double fine_structure_constant{};
    std::vector<BremsLpmComponent> components;
  };

  struct MoliereComponentMetadata {
    std::uint64_t proposal_hash{};
    double nuclear_charge{};
    double atomic_mass_number{};
    double atoms_in_molecule{};
  };

  /**
   * Versioned numerical state for PROPOSAL 7.6.2 Moliere scattering.
   *
   * Synthetic/unit-test tables may set enabled=false and leave all remaining
   * fields empty. Production CUDA capability checks must reject such a table
   * before selecting charged transport with multiple scattering.
   */
  struct MoliereMetadata {
    bool enabled{};
    std::string reference_mode;
    double particle_mass_MeV{};
    double electron_mass_MeV{};
    double fine_structure_constant{};
    double avogadro_per_mol{};
    double hbar_MeV_s{};
    double speed_of_light_cm_per_s{};
    double euler_mascheroni{};
    std::vector<MoliereComponentMetadata> components;
    std::vector<double> c1;
    std::vector<double> c2;
    std::vector<double> c2_large;
    std::vector<double> s2_large;
    std::vector<double> C1_large;
  };

} // namespace corsika::gpu::em::tables
