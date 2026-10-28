/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  inline constexpr std::size_t MaxBremsLpmComponents = 16;

  /** One PROPOSAL bremsstrahlung LPM component, without host pointers. */
  struct BremsLpmComponentSnapshot {
    std::uint64_t componentHash_{};
    double nuclearCharge_{};
    double atomicMassNumber_{};
    double radiationLogConstant_{};
  };

  /** Pointer-free PROPOSAL bremsstrahlung LPM state.
   * Energies and masses are in MeV; density and length units are explicit below.
   * The muon extension is exported from the live calculator, not inferred by scaling.
   * Member order and initialization retain the production auxiliary-cache layout. */
  struct BremsLpmSnapshot {
    double baselineMassDensityGPerCm3_{};
    double molecularDensityPerCm3_{};
    double sumCharge_{};
    double eLpmMeV_{};
    double leptonMassMeV_{};
    double electronMassMeV_{};
    double muonMassMeV_{};
    double classicalElectronRadiusCm_{};
    double fineStructureConstant_{};
    std::uint32_t componentCount_{};
    std::uint32_t reserved_{};
    BremsLpmComponentSnapshot components_[MaxBremsLpmComponents]{};
    // Exported from the live muon calculator, not
    // inferred by scaling the electron characteristic energy.
    double muonBaselineMassDensityGPerCm3_{};
    double muonMolecularDensityPerCm3_{};
    double muonSumCharge_{};
    double muonELpmMeV_{};
  };

  /** Outcome of a bremsstrahlung LPM query. */
  enum class BremsLpmStatus : std::uint32_t {
    Success = 0,
    ComponentNotFound = 1,
    InvalidSnapshot = 2,
    InvalidInput = 3,
    NonFiniteResult = 4,
  };

  /** Suppression factor and original query status; no upper probability clamp is applied.
   */
  struct BremsLpmResult {
    BremsLpmStatus status_{BremsLpmStatus::InvalidSnapshot};
    std::uint32_t reserved_{};
    double survivalProbability_{};
  };

  static_assert(std::is_standard_layout_v<BremsLpmSnapshot>);
  static_assert(std::is_trivially_copyable_v<BremsLpmSnapshot>);
  static_assert(std::is_standard_layout_v<BremsLpmResult>);
  static_assert(std::is_trivially_copyable_v<BremsLpmResult>);

  /**
   * Evaluate the production bremsstrahlung LPM suppression formula.
   * energy_MeV is the incident lepton total energy; v is its sampled stochastic
   * loss fraction. The local vertex density is in g/cm^3.
   * Snapshot, input and selected-component checks retain their original order.
   * The returned factor is not clamped to one. This function makes no random draw.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline BremsLpmResult brems_lpm_suppression_factor(
      BremsLpmSnapshot const& snapshot, std::uint64_t component_hash, double energy_MeV,
      double v, double local_mass_density_g_per_cm3);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/models/BremsLpm.inl>
