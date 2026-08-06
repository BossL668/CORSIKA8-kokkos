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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace corsika::gpu::em::tables {

  inline constexpr std::uint32_t RateTableFormatVersion = 10;
  // PROPOSAL/LPM validation in gpu_em_tablegen is contracted through 1e14 MeV
  // (1e20 eV). Preparation requests above this bound are rejected rather than
  // extrapolating validation metadata.
  inline constexpr double MaximumGeneratedTableEnergyMeV = 1.e14;
  inline constexpr double ProposalRelativeVCut = 0.01;
  inline constexpr char RateUnit[] = "cm2/g";
  inline constexpr char EnergyUnit[] = "MeV";
  // An explicit, serialized capability marker for a process/component whose
  // rate remains device-selectable but whose PROPOSAL inverse CDF cannot be
  // represented monotonically within the requested table tolerance.  The
  // selector preserves process, component and loss quantile and routes only
  // the selected-loss evaluation/final state through the CPU fallback.
  inline constexpr char SelectedLossCpuFallbackReferenceMode[] =
      "proposal_selected_loss_cpu_fallback";
  inline constexpr std::uint64_t LossQuantileCount =
      std::uint64_t{1} << 32;
  inline constexpr double lossQuantileFromPhiloxWord(
      std::uint32_t word) {
    return (static_cast<double>(word) + 0.5) /
           static_cast<double>(LossQuantileCount);
  }
  inline constexpr double PhiloxLossQuantileMinimum =
      lossQuantileFromPhiloxWord(0);
  inline constexpr double PhiloxLossQuantileMaximum =
      lossQuantileFromPhiloxWord(std::uint32_t{0xffffffffu});
  inline constexpr double LossQuantileMinimum =
      PhiloxLossQuantileMinimum;
  inline constexpr double LossQuantileMaximum =
      PhiloxLossQuantileMaximum;
  // PROPOSAL 7.6.2's interpolated Epair inverse has a target-dependent,
  // discontinuous Newton-root branch below u ~= 1.2e-4 around 145 MeV when
  // the absolute cut is at least 5 MeV. A continuous two-dimensional table
  // cannot represent that numerical discontinuity to 1e-3 accuracy. Keep a
  // conservative capability boundary at 2e-4: the lower 0.02% of selected
  // Epair vertices is returned through the explicit, counted PROPOSAL
  // fallback instead of silently smoothing the reference discontinuity.
  // PROPOSAL direct integration is also unreliable in the uppermost tail.
  inline constexpr double EpairLossQuantileMinimum = 2.e-4;
  inline constexpr double EpairLossQuantileMaximum = 0.98;
  inline constexpr double EpairRhoQuantileMinimum = 1.e-4;
  inline constexpr double EpairRhoQuantileMaximum = 1. - 1.e-4;
  inline constexpr double EpairLossMinimumEnergyMeV = 20.;
  inline constexpr double BremsLossMinimumEnergyMeV = 1.;
  // The auxiliary photon-pair final-state column stores the normalized
  // kinematic coordinate s=(rho-m/E)/(1-2m/E), not rho itself.  This remains
  // smooth down to the physical pair threshold.
  // The adaptive PROPOSAL inverse-CDF table is well behaved above 10 GeV.
  // Lower-energy pair splits use the device Koch--Motz analytical sampler.
  inline constexpr double PhotonPairFinalStateMinimumEnergyMeV = 1.e4;
  inline constexpr double PhotonPairFinalStateQuantileMinimum = 1.e-4;
  inline constexpr double PhotonPairFinalStateQuantileMaximum =
      1. - 1.e-4;
  inline constexpr double IonizationLossQuantileMaximum =
      1. - 1.e-5;
  inline constexpr double LossFractionRelativeScaleFloor = 1.e-6;
  // Mirrors proposal::ContinuousProcess::getMaxStepLength(): step slightly
  // below the kinetic cut so the following ParticleCut comparison (< cut)
  // absorbs the particle.
  inline constexpr double ContinuousCutSafetyFactor = 0.9999;

  struct MediumComponent {
    std::int32_t corsika_pid{};
    std::uint64_t proposal_hash{};
    double number_fraction{};
    std::string name;
  };

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

  struct InverseCdfTable {
    // "proposal_interpolated" follows the production CPU interpolant,
    // "proposal_interpolated_monotone" projects its narrow local reversals,
    // and "proposal_direct" uses interpolate=false integration/root finding.
    std::string reference_mode;
    std::vector<double> energies_MeV;
    // Row i occupies [quantile_offsets[i], quantile_offsets[i + 1]).
    std::vector<std::uint64_t> quantile_offsets;
    std::vector<double> quantiles;
    // Same ragged indexing as quantiles.
    std::vector<double> v_loss;
  };

  struct RateColumn {
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    std::string process_name;
    std::string parameterization;
    std::string target_name;
    std::vector<double> rates_cm2_per_g;
    InverseCdfTable inverse_cdf;
  };

  struct ParticleRateTable {
    std::int32_t pdg_id{};
    std::string particle_name;
    std::uint64_t interaction_hash{};
    std::vector<double> energies_MeV;
    std::vector<RateColumn> columns;
  };

  /**
   * Continuous stopping-power/range data for one charged particle.
   *
   * dEdX uses MeV/(g/cm^2). range_g_per_cm2 is the PROPOSAL interpolated
   * displacement integral from minimum_total_energy_MeV to the corresponding
   * energy. It is strictly increasing and therefore also serves as the
   * inverse-range coordinate used after a traversed grammage.
   */
  struct ContinuousEnergyTable {
    std::int32_t pdg_id{};
    std::string particle_name;
    std::string reference_mode;
    double mass_MeV{};
    double minimum_total_energy_MeV{};
    double measured_max_relative_error{};
    std::vector<double> energies_MeV;
    std::vector<double> dEdX_MeV_cm2_per_g;
    std::vector<double> range_g_per_cm2;
  };

  /**
   * Dense inverse CDF for the KKP electron-pair final-state asymmetry.
   *
   * The independent coordinates are parent energy, the positive threshold
   * excess s_v=E*v/(4*m_e)-1, and the independent rho quantile. Values are
   * |rho|/rho_max in component-major row-major order:
   *
   *   (((component * E_count + E) * v_count + v) * rho_count + rho).
   *
   * log(s_v) fixes the moving E*v=4*m_e pair threshold to one coordinate and
   * is substantially smoother than either v or its stochastic random number.
   */
  struct EpairRhoInverseCdfTable {
    bool enabled{};
    std::string reference_mode;
    double requested_max_normalized_error{};
    double measured_max_normalized_error{};
    std::vector<std::uint64_t> component_hashes;
    std::vector<double> energies_MeV;
    std::vector<double> v_coordinates;
    std::vector<double> rho_quantiles;
    std::vector<double> normalized_rho;
  };

  struct RateTableMetadata {
    std::string proposal_version;
    std::string generator_version;
    std::string medium_name;
    std::uint64_t proposal_medium_hash{};
    double energy_cut_MeV{};
    double relative_v_cut{};
    double energy_min_MeV{};
    double energy_max_MeV{};
    double requested_relative_tolerance{};
    double measured_max_relative_error{};
    double requested_loss_relative_tolerance{};
    double measured_max_loss_relative_error{};
    std::vector<MediumComponent> components;
    PhotonPairLpmMetadata photon_pair_lpm;
    BremsLpmMetadata brems_lpm;
    MoliereMetadata moliere;
  };

  struct RateTableSet {
    RateTableMetadata metadata;
    std::vector<ParticleRateTable> particles;
    std::vector<ContinuousEnergyTable> continuous_energy_tables;
    EpairRhoInverseCdfTable epair_rho;
    Sha256Digest content_hash{};
  };

  struct RateTableRequirements {
    std::string proposal_version;
    std::string medium_name;
    std::uint64_t proposal_medium_hash{};
    double energy_cut_MeV{};
    double relative_v_cut{};
    double required_energy_min_MeV{};
    double required_energy_max_MeV{};
    double maximum_relative_error{};
    double maximum_loss_relative_error{};
    std::vector<MediumComponent> components;
  };

  inline std::uint32_t rateTableProcessCount(RateTableSet const& table) {
    std::size_t count = 0;
    for (auto const& particle : table.particles) {
      if (particle.columns.size() >
          std::numeric_limits<std::size_t>::max() - count) {
        throw std::overflow_error("GPU EM rate-table process count overflow");
      }
      count += particle.columns.size();
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "GPU EM rate-table process count exceeds descriptor capacity");
    }
    return static_cast<std::uint32_t>(count);
  }

  /**
   * Validate all invariants needed by the host reader and future CUDA upload.
   * Throws std::invalid_argument for malformed in-memory data.
   */
  void validateRateTable(RateTableSet const& table);

  /**
   * Reject a valid but physically incompatible cache (wrong PROPOSAL version,
   * medium, cuts, domain, tolerance, or composition).
   */
  void validateCompatibility(RateTableSet const& table,
                             RateTableRequirements const& requirements);

  /**
   * Interpolate one dN/dX column in log(E). Positive endpoints are interpolated
   * logarithmically in rate; a threshold interval containing zero is interpolated
   * linearly in rate. Extrapolation is deliberately forbidden.
   */
  double interpolateRate(ParticleRateTable const& particle,
                         std::int32_t process_id,
                         std::uint64_t component_hash,
                         double energy_MeV);

  double interpolateTotalRate(ParticleRateTable const& particle,
                              double energy_MeV);

  /**
   * Interpolate the conditional inverse CDF v(E,u) for one already selected
   * process/component. energy is interpolated in log(E), v in log(v), and u
   * in logit(u). Quantiles or energies outside the stored capability domain
   * throw std::out_of_range so the hybrid scheduler can count and route an
   * explicit CPU fallback.
   */
  double interpolateLossFraction(ParticleRateTable const& particle,
                                 std::int32_t process_id,
                                 std::uint64_t component_hash,
                                 double energy_MeV, double quantile);

  ParticleRateTable const& findParticle(RateTableSet const& table,
                                        std::int32_t pdg_id);

  ContinuousEnergyTable const& findContinuousEnergyTable(
      RateTableSet const& table, std::int32_t pdg_id);

  double interpolateContinuousDedx(
      ContinuousEnergyTable const&, double energy_MeV);

  double interpolateContinuousRange(
      ContinuousEnergyTable const&, double energy_MeV);

  double interpolateContinuousEnergy(
      ContinuousEnergyTable const&, double range_g_per_cm2);

  /**
   * Return the final total energy after continuous loss through grammage.
   * Throws std::out_of_range if the requested grammage reaches/breaches the
   * configured transport cut.
   */
  double energyAfterContinuousLoss(
      ContinuousEnergyTable const&, double initial_energy_MeV,
      double grammage_g_per_cm2);

  /**
   * Deterministic little-endian binary I/O. The SHA-256 covers exactly the
   * payload bytes, not the fixed file envelope containing the digest.
   */
  Sha256Digest writeRateTable(std::filesystem::path const& path,
                              RateTableSet const& table);
  RateTableSet readRateTable(std::filesystem::path const& path);
  Sha256Digest calculateContentHash(RateTableSet const& table);

} // namespace corsika::gpu::em::tables
