/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/gpu/em/tables/Sha256.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace corsika::gpu::em::tables {

  inline constexpr std::uint32_t MediumConfigSchemaVersion = 1;
  inline constexpr char TableGeneratorContractVersion[] =
      "c8-gpu-em-tablegen-0.18";

  struct MediumComponentConfig {
    std::string name;
    std::int32_t corsika_pid{};
    double nuclear_charge{};
    double atomic_mass_g_per_mol{};
    double number_fraction{};
  };

  /**
   * Complete material state required by PROPOSAL::Medium.
   *
   * Field names retain the PROPOSAL/Sternheimer convention. In particular,
   * density_correction_C is the signed C value passed directly to PROPOSAL
   * (CORSIKA's positive Cbar therefore appears as a negative value here).
   */
  struct MediumConfig {
    std::uint32_t schema_version{MediumConfigSchemaVersion};
    std::string name;
    double mean_excitation_energy_eV{};
    double density_correction_C{};
    double density_correction_a{};
    double density_correction_m{};
    double density_correction_x0{};
    double density_correction_x1{};
    double density_correction_delta0{};
    double reference_mass_density_g_per_cm3{};
    std::vector<MediumComponentConfig> components;
  };

  /**
   * Return the exact dry-air contract historically used by gpu_em_tablegen.
   */
  MediumConfig standardDryAirMediumConfig();

  /**
   * Load, validate, normalize and component-sort a schema-v1 YAML file.
   *
   * Unknown fields are rejected. Number fractions must sum to one within
   * 1e-8 and are then normalized exactly so harmless decimal roundoff does
   * not create a distinct content address.
   */
  MediumConfig loadMediumConfig(std::filesystem::path const& path);

  /**
   * Validate and normalize an in-memory configuration.
   */
  MediumConfig normalizeMediumConfig(MediumConfig config);

  /**
   * Stable, human-readable canonical serialization used as the hash input.
   */
  std::string canonicalMediumYaml(MediumConfig const& config);

  Sha256Digest mediumConfigHash(MediumConfig const& config);
  std::string mediumConfigHashHex(MediumConfig const& config);

  /**
   * Atomically write the canonical YAML representation.
   */
  void writeCanonicalMediumYaml(std::filesystem::path const& path,
                                MediumConfig const& config);

} // namespace corsika::gpu::em::tables
