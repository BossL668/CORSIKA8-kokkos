/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "AuxiliaryModelReference.hpp"
#include "AuxiliaryModelFixture.hpp"
#include <cstddef>

namespace corsika::kokkos::test {
  static_assert(sizeof(PhotonPairLpmComponentSnapshot) ==
                sizeof(auxiliary_reference::PhotonPairLpmComponentSnapshot));
  static_assert(alignof(PhotonPairLpmComponentSnapshot) ==
                alignof(auxiliary_reference::PhotonPairLpmComponentSnapshot));
  static_assert(offsetof(PhotonPairLpmComponentSnapshot, componentHash_) ==
                offsetof(auxiliary_reference::PhotonPairLpmComponentSnapshot,
                         component_hash));
  static_assert(offsetof(PhotonPairLpmComponentSnapshot, nuclearCharge_) ==
                offsetof(auxiliary_reference::PhotonPairLpmComponentSnapshot,
                         nuclear_charge));
  static_assert(offsetof(PhotonPairLpmComponentSnapshot, radiationLogConstant_) ==
                offsetof(auxiliary_reference::PhotonPairLpmComponentSnapshot,
                         radiation_log_constant));
  inline auxiliary_reference::PhotonPairLpmComponentSnapshot to_reference(
      PhotonPairLpmComponentSnapshot const& input) {
    auxiliary_reference::PhotonPairLpmComponentSnapshot result{};
    result.component_hash = input.componentHash_;
    result.nuclear_charge = input.nuclearCharge_;
    result.radiation_log_constant = input.radiationLogConstant_;
    return result;
  }
  static_assert(sizeof(PhotonPairLpmSnapshot) ==
                sizeof(auxiliary_reference::PhotonPairLpmSnapshot));
  static_assert(alignof(PhotonPairLpmSnapshot) ==
                alignof(auxiliary_reference::PhotonPairLpmSnapshot));
  static_assert(offsetof(PhotonPairLpmSnapshot, baselineMassDensityGPerCm3_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot,
                         baseline_mass_density_g_per_cm3));
  static_assert(offsetof(PhotonPairLpmSnapshot, molecularDensityPerCm3_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot,
                         molecular_density_per_cm3));
  static_assert(offsetof(PhotonPairLpmSnapshot, sumCharge_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot, sum_charge));
  static_assert(offsetof(PhotonPairLpmSnapshot, eLpmMeV_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot, e_lpm_MeV));
  static_assert(offsetof(PhotonPairLpmSnapshot, classicalElectronRadiusCm_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot,
                         classical_electron_radius_cm));
  static_assert(offsetof(PhotonPairLpmSnapshot, fineStructureConstant_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot,
                         fine_structure_constant));
  static_assert(offsetof(PhotonPairLpmSnapshot, componentCount_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot, component_count));
  static_assert(offsetof(PhotonPairLpmSnapshot, reserved_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot, reserved));
  static_assert(offsetof(PhotonPairLpmSnapshot, components_) ==
                offsetof(auxiliary_reference::PhotonPairLpmSnapshot, components));
  inline auxiliary_reference::PhotonPairLpmSnapshot to_reference(
      PhotonPairLpmSnapshot const& input) {
    auxiliary_reference::PhotonPairLpmSnapshot result{};
    result.baseline_mass_density_g_per_cm3 = input.baselineMassDensityGPerCm3_;
    result.molecular_density_per_cm3 = input.molecularDensityPerCm3_;
    result.sum_charge = input.sumCharge_;
    result.e_lpm_MeV = input.eLpmMeV_;
    result.classical_electron_radius_cm = input.classicalElectronRadiusCm_;
    result.fine_structure_constant = input.fineStructureConstant_;
    result.component_count = input.componentCount_;
    result.reserved = input.reserved_;
    for (std::size_t index = 0; index < MaxPhotonPairLpmComponents; ++index)
      result.components[index] = to_reference(input.components_[index]);
    return result;
  }
  static_assert(sizeof(BremsLpmComponentSnapshot) ==
                sizeof(auxiliary_reference::BremsLpmComponentSnapshot));
  static_assert(alignof(BremsLpmComponentSnapshot) ==
                alignof(auxiliary_reference::BremsLpmComponentSnapshot));
  static_assert(offsetof(BremsLpmComponentSnapshot, componentHash_) ==
                offsetof(auxiliary_reference::BremsLpmComponentSnapshot, component_hash));
  static_assert(offsetof(BremsLpmComponentSnapshot, nuclearCharge_) ==
                offsetof(auxiliary_reference::BremsLpmComponentSnapshot, nuclear_charge));
  static_assert(offsetof(BremsLpmComponentSnapshot, atomicMassNumber_) ==
                offsetof(auxiliary_reference::BremsLpmComponentSnapshot,
                         atomic_mass_number));
  static_assert(offsetof(BremsLpmComponentSnapshot, radiationLogConstant_) ==
                offsetof(auxiliary_reference::BremsLpmComponentSnapshot,
                         radiation_log_constant));
  inline auxiliary_reference::BremsLpmComponentSnapshot to_reference(
      BremsLpmComponentSnapshot const& input) {
    auxiliary_reference::BremsLpmComponentSnapshot result{};
    result.component_hash = input.componentHash_;
    result.nuclear_charge = input.nuclearCharge_;
    result.atomic_mass_number = input.atomicMassNumber_;
    result.radiation_log_constant = input.radiationLogConstant_;
    return result;
  }
  static_assert(sizeof(BremsLpmSnapshot) ==
                sizeof(auxiliary_reference::BremsLpmSnapshot));
  static_assert(alignof(BremsLpmSnapshot) ==
                alignof(auxiliary_reference::BremsLpmSnapshot));
  static_assert(offsetof(BremsLpmSnapshot, baselineMassDensityGPerCm3_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot,
                         baseline_mass_density_g_per_cm3));
  static_assert(offsetof(BremsLpmSnapshot, molecularDensityPerCm3_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot,
                         molecular_density_per_cm3));
  static_assert(offsetof(BremsLpmSnapshot, sumCharge_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, sum_charge));
  static_assert(offsetof(BremsLpmSnapshot, eLpmMeV_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, e_lpm_MeV));
  static_assert(offsetof(BremsLpmSnapshot, leptonMassMeV_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, lepton_mass_MeV));
  static_assert(offsetof(BremsLpmSnapshot, electronMassMeV_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, electron_mass_MeV));
  static_assert(offsetof(BremsLpmSnapshot, muonMassMeV_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, muon_mass_MeV));
  static_assert(offsetof(BremsLpmSnapshot, classicalElectronRadiusCm_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot,
                         classical_electron_radius_cm));
  static_assert(offsetof(BremsLpmSnapshot, fineStructureConstant_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, fine_structure_constant));
  static_assert(offsetof(BremsLpmSnapshot, componentCount_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, component_count));
  static_assert(offsetof(BremsLpmSnapshot, reserved_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, reserved));
  static_assert(offsetof(BremsLpmSnapshot, components_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, components));
  static_assert(offsetof(BremsLpmSnapshot, muonBaselineMassDensityGPerCm3_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot,
                         muon_baseline_mass_density_g_per_cm3));
  static_assert(offsetof(BremsLpmSnapshot, muonMolecularDensityPerCm3_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot,
                         muon_molecular_density_per_cm3));
  static_assert(offsetof(BremsLpmSnapshot, muonSumCharge_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, muon_sum_charge));
  static_assert(offsetof(BremsLpmSnapshot, muonELpmMeV_) ==
                offsetof(auxiliary_reference::BremsLpmSnapshot, muon_e_lpm_MeV));
  inline auxiliary_reference::BremsLpmSnapshot to_reference(
      BremsLpmSnapshot const& input) {
    auxiliary_reference::BremsLpmSnapshot result{};
    result.baseline_mass_density_g_per_cm3 = input.baselineMassDensityGPerCm3_;
    result.molecular_density_per_cm3 = input.molecularDensityPerCm3_;
    result.sum_charge = input.sumCharge_;
    result.e_lpm_MeV = input.eLpmMeV_;
    result.lepton_mass_MeV = input.leptonMassMeV_;
    result.electron_mass_MeV = input.electronMassMeV_;
    result.muon_mass_MeV = input.muonMassMeV_;
    result.classical_electron_radius_cm = input.classicalElectronRadiusCm_;
    result.fine_structure_constant = input.fineStructureConstant_;
    result.component_count = input.componentCount_;
    result.reserved = input.reserved_;
    for (std::size_t index = 0; index < MaxBremsLpmComponents; ++index)
      result.components[index] = to_reference(input.components_[index]);
    result.muon_baseline_mass_density_g_per_cm3 = input.muonBaselineMassDensityGPerCm3_;
    result.muon_molecular_density_per_cm3 = input.muonMolecularDensityPerCm3_;
    result.muon_sum_charge = input.muonSumCharge_;
    result.muon_e_lpm_MeV = input.muonELpmMeV_;
    return result;
  }
  static_assert(sizeof(MoliereComponentSnapshot) ==
                sizeof(auxiliary_reference::MoliereComponentSnapshot));
  static_assert(alignof(MoliereComponentSnapshot) ==
                alignof(auxiliary_reference::MoliereComponentSnapshot));
  static_assert(offsetof(MoliereComponentSnapshot, nuclearCharge_) ==
                offsetof(auxiliary_reference::MoliereComponentSnapshot, nuclear_charge));
  static_assert(offsetof(MoliereComponentSnapshot, weightZz_) ==
                offsetof(auxiliary_reference::MoliereComponentSnapshot, weight_zz));
  static_assert(offsetof(MoliereComponentSnapshot, chi0SquaredMeV2_) ==
                offsetof(auxiliary_reference::MoliereComponentSnapshot,
                         chi_0_squared_MeV2));
  static_assert(offsetof(MoliereComponentSnapshot, coulombCorrection_) ==
                offsetof(auxiliary_reference::MoliereComponentSnapshot,
                         coulomb_correction));
  inline auxiliary_reference::MoliereComponentSnapshot to_reference(
      MoliereComponentSnapshot const& input) {
    auxiliary_reference::MoliereComponentSnapshot result{};
    result.nuclear_charge = input.nuclearCharge_;
    result.weight_zz = input.weightZz_;
    result.chi_0_squared_MeV2 = input.chi0SquaredMeV2_;
    result.coulomb_correction = input.coulombCorrection_;
    return result;
  }
  static_assert(sizeof(MoliereSnapshot) == sizeof(auxiliary_reference::MoliereSnapshot));
  static_assert(alignof(MoliereSnapshot) ==
                alignof(auxiliary_reference::MoliereSnapshot));
  static_assert(offsetof(MoliereSnapshot, particleMassMeV_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, particle_mass_MeV));
  static_assert(offsetof(MoliereSnapshot, electronMassMeV_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, electron_mass_MeV));
  static_assert(offsetof(MoliereSnapshot, fineStructureConstant_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, fine_structure_constant));
  static_assert(offsetof(MoliereSnapshot, avogadroPerMol_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, avogadro_per_mol));
  static_assert(offsetof(MoliereSnapshot, hbarMeVS_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, hbar_MeV_s));
  static_assert(offsetof(MoliereSnapshot, speedOfLightCmPerS_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, speed_of_light_cm_per_s));
  static_assert(offsetof(MoliereSnapshot, zSquaredOverAAverage_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, z_squared_over_a_average));
  static_assert(offsetof(MoliereSnapshot, inverseWeightZzSum_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, inverse_weight_zz_sum));
  static_assert(offsetof(MoliereSnapshot, eulerMascheroni_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, euler_mascheroni));
  static_assert(offsetof(MoliereSnapshot, componentCount_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, component_count));
  static_assert(offsetof(MoliereSnapshot, maximumWeightIndex_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, maximum_weight_index));
  static_assert(offsetof(MoliereSnapshot, components_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, components));
  static_assert(offsetof(MoliereSnapshot, c1_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, c1));
  static_assert(offsetof(MoliereSnapshot, c2_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, c2));
  static_assert(offsetof(MoliereSnapshot, c2Large_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, c2_large));
  static_assert(offsetof(MoliereSnapshot, s2Large_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, s2_large));
  static_assert(offsetof(MoliereSnapshot, c1IntegralLarge_) ==
                offsetof(auxiliary_reference::MoliereSnapshot, C1_large));
  inline auxiliary_reference::MoliereSnapshot to_reference(MoliereSnapshot const& input) {
    auxiliary_reference::MoliereSnapshot result{};
    result.particle_mass_MeV = input.particleMassMeV_;
    result.electron_mass_MeV = input.electronMassMeV_;
    result.fine_structure_constant = input.fineStructureConstant_;
    result.avogadro_per_mol = input.avogadroPerMol_;
    result.hbar_MeV_s = input.hbarMeVS_;
    result.speed_of_light_cm_per_s = input.speedOfLightCmPerS_;
    result.z_squared_over_a_average = input.zSquaredOverAAverage_;
    result.inverse_weight_zz_sum = input.inverseWeightZzSum_;
    result.euler_mascheroni = input.eulerMascheroni_;
    result.component_count = input.componentCount_;
    result.maximum_weight_index = input.maximumWeightIndex_;
    for (std::size_t index = 0; index < MaxMoliereComponents; ++index)
      result.components[index] = to_reference(input.components_[index]);
    for (std::size_t index = 0; index < MoliereSeriesCoefficientCount; ++index)
      result.c1[index] = input.c1_[index];
    for (std::size_t index = 0; index < MoliereSeriesCoefficientCount; ++index)
      result.c2[index] = input.c2_[index];
    for (std::size_t index = 0; index < MoliereLargeSeriesCoefficientCount; ++index)
      result.c2_large[index] = input.c2Large_[index];
    for (std::size_t index = 0; index < MoliereLargeSeriesCoefficientCount; ++index)
      result.s2_large[index] = input.s2Large_[index];
    for (std::size_t index = 0; index < MoliereLargeIntegralCoefficientCount; ++index)
      result.C1_large[index] = input.c1IntegralLarge_[index];
    return result;
  }
  struct AuxiliaryReferenceSnapshots {
    auxiliary_reference::PhotonPairLpmSnapshot photon_{};
    auxiliary_reference::BremsLpmSnapshot electron_{};
    auxiliary_reference::BremsLpmSnapshot muon_{};
    auxiliary_reference::MoliereSnapshot electronMoliere_{};
    auxiliary_reference::MoliereSnapshot muonMoliere_{};
  };
  inline AuxiliaryReferenceSnapshots to_reference(AuxiliarySnapshots const& input) {
    return {to_reference(input.photon_), to_reference(input.electron_),
            to_reference(input.muon_), to_reference(input.electronMoliere_),
            to_reference(input.muonMoliere_)};
  }
} // namespace corsika::kokkos::test
