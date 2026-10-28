/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
#include <corsika/kokkos/output/ProfileProjectionStorage.hpp>
#include <corsika/kokkos/runtime/ResidentMemoryBudget.hpp>
#include <tests/kokkos/output/ProfileStorageReference.hpp>
#include <tests/kokkos/output/ProfileAccumulationChecks.hpp>
namespace profile_storage_test {
  namespace current = corsika::kokkos;
  namespace reference = profile_storage_reference;
  inline reference::ProfileProjection convert(current::ProfileProjectionConfig const& a) {
    reference::ProfileProjection b;
    b.enabled = a.enabled_;
    b.accumulate_on_device = a.accumulateOnDevice_;
    b.axis_step_length_m = a.axisStepLengthM_;
    b.axis_grammage_g_per_cm2 = a.axisGrammageGPerCm2_;
    b.output_bin_count = a.outputBinCount_;
    b.output_bin_width_g_per_cm2 = a.outputBinWidthGPerCm2_;
    b.energy_loss_threshold_g_per_cm2 = a.energyLossThresholdGPerCm2_;
    b.fixed_point_weight_limit = a.fixedPointWeightLimit_;
    b.fixed_point_energy_limit_GeV = a.fixedPointEnergyLimitGeV_;
    for (int k = 0; k < 3; ++k) b.axis_start_position_m[k] = a.axisStartPositionM_[k];
    for (int k = 0; k < 3; ++k) b.axis_direction[k] = a.axisDirection_[k];
    b.crossing_mode = static_cast<reference::ProfileCrossingMode>(a.crossingMode_);
    return b;
  }
  inline void compare(current::ProfileResult const& a,
                      reference::GpuProfileResult const& b) {
    REQUIRE(a.photons_ == b.photons);
    REQUIRE(a.electrons_ == b.electrons);
    REQUIRE(a.positrons_ == b.positrons);
    REQUIRE(a.muonsMinus_ == b.muons_minus);
    REQUIRE(a.muonsPlus_ == b.muons_plus);
    REQUIRE(a.muonParentProductions_ == b.muon_parent_productions);
    REQUIRE(a.energyLossGeV_ == b.energy_loss_GeV);
    REQUIRE(a.muonEnergyLossGeV_ == b.muon_energy_loss_GeV);
    REQUIRE(a.steps_ == b.steps);
    REQUIRE(a.depositedSteps_ == b.deposited_steps);
    REQUIRE(a.particleCuts_ == b.particle_cuts);
    REQUIRE(a.fixedPointOverflows_ == b.fixed_point_overflows);
    REQUIRE(a.invalidRecords_ == b.invalid_records);
    REQUIRE(a.weightedDepositedEnergyGeV_ == b.weighted_deposited_energy_GeV);
    REQUIRE(a.weightedMediumRestMassInputGeV_ == b.weighted_medium_rest_mass_input_GeV);
    REQUIRE(a.weightedCutRestMassEnergyGeV_ == b.weighted_cut_rest_mass_energy_GeV);
    REQUIRE(a.weightedObservedTotalEnergyGeV_ == b.weighted_observed_total_energy_GeV);
    REQUIRE(a.weightedEscapedTotalEnergyGeV_ == b.weighted_escaped_total_energy_GeV);
    REQUIRE(a.weightedUnwrittenPhotoelectricBindingEnergyGeV_ ==
            b.weighted_unwritten_photoelectric_binding_energy_GeV);
    REQUIRE(a.weightedObservationCutOverlapEnergyGeV_ ==
            b.weighted_observation_cut_overlap_energy_GeV);
    REQUIRE(a.weightedMassConventionCorrectionGeV_ ==
            b.weighted_mass_convention_correction_GeV);
    REQUIRE(a.weightedNativeMuonPairCutEnergyGeV_ ==
            b.weighted_native_muon_pair_cut_energy_GeV);
  }
  inline void compare(current::FixedProfileSnapshot const& a,
                      reference::FixedProfileSnapshot const& b) {
    REQUIRE(a.showerIdentity_ == b.shower_identity);
    REQUIRE(a.weightUnits_ == b.weight_units);
    REQUIRE(a.energyUnits_ == b.energy_units);
    REQUIRE(a.histograms_ == b.histograms);
    profile_test::compare(a.counters_, b.counters);
  }
} // namespace profile_storage_test
