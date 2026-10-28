/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
/* Generated field-for-field comparison with the frozen production oracle. */
#pragma once
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
namespace counter_fixture {
  template <typename TCurrent, typename TOld>
  void compare_counters(TCurrent const& current, TOld const& old) {
    CHECK(current.getCount() == old.getCount());
    CHECK(current.getHistogram().labHist() == old.getHistogram().labHist());
    CHECK(current.getHistogram().CMSHist() == old.getHistogram().CMSHist());
    CHECK(std::isfinite(current.getTotalFinalStateTimeMs()));
    CHECK(current.getTotalFinalStateTimeMs() >= 0.);
    auto const& samples = current.getTimingSamples();
    auto const& oldSamples = old.getTimingSamples();
    REQUIRE(samples.size() == oldSamples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
      CHECK(samples[i].sequenceId_ == oldSamples[i].sequence_id);
      CHECK(samples[i].projectile_ == oldSamples[i].projectile);
      CHECK(samples[i].target_ == oldSamples[i].target);
      CHECK(samples[i].kineticEnergy_ == oldSamples[i].kinetic_energy);
      CHECK(std::isfinite(samples[i].finalStateTimeMs_));
      CHECK(samples[i].finalStateTimeMs_ >= 0.);
      if (samples[i].deferred_) CHECK(samples[i].finalStateTimeMs_ == 0.);
      CHECK(samples[i].deferred_ == oldSamples[i].deferred);
    }
    auto const& ledger = current.getEnergyLedgerStatistics();
    auto const& oldLedger = old.getEnergyLedgerStatistics();
    CHECK(ledger.auditedInteractions_ == oldLedger.audited_interactions);
    CHECK(ledger.deferredInteractions_ == oldLedger.deferred_interactions);
    CHECK(ledger.weightedProjectileTotalEnergyGeV_ ==
          oldLedger.weighted_projectile_total_energy_GeV);
    CHECK(ledger.weightedTargetTotalEnergyGeV_ ==
          oldLedger.weighted_target_total_energy_GeV);
    CHECK(ledger.weightedEffectiveTargetTotalEnergyGeV_ ==
          oldLedger.weighted_effective_target_total_energy_GeV);
    CHECK(ledger.weightedTargetIsotopeCorrectionGeV_ ==
          oldLedger.weighted_target_isotope_correction_GeV);
    CHECK(ledger.targetIsotopeAdjustedInteractions_ ==
          oldLedger.target_isotope_adjusted_interactions);
    CHECK(ledger.weightedSecondaryTotalEnergyGeV_ ==
          oldLedger.weighted_secondary_total_energy_GeV);
    CHECK(ledger.weightedEnergyResidualGeV_ == oldLedger.weighted_energy_residual_GeV);
    CHECK(ledger.weightedAbsoluteEnergyResidualGeV_ ==
          oldLedger.weighted_absolute_energy_residual_GeV);
    CHECK(ledger.maximumAbsoluteResidualGeV_ == oldLedger.maximum_absolute_residual_GeV);
    CHECK(ledger.weightedIsotopeCorrectedEnergyResidualGeV_ ==
          oldLedger.weighted_isotope_corrected_energy_residual_GeV);
    CHECK(ledger.weightedAbsoluteIsotopeCorrectedEnergyResidualGeV_ ==
          oldLedger.weighted_absolute_isotope_corrected_energy_residual_GeV);
    CHECK(ledger.maximumAbsoluteIsotopeCorrectedResidualGeV_ ==
          oldLedger.maximum_absolute_isotope_corrected_residual_GeV);
    CHECK(ledger.negativeIsotopeCorrectedResidualInteractions_ ==
          oldLedger.negative_isotope_corrected_residual_interactions);
    CHECK(ledger.maximumNegativeIsotopeCorrectedResidualGeV_ ==
          oldLedger.maximum_negative_isotope_corrected_residual_GeV);
    CHECK(ledger.spacelikeIsotopeCorrectedResidualInteractions_ ==
          oldLedger.spacelike_isotope_corrected_residual_interactions);
    CHECK(ledger.maximumIsotopeCorrectedSpacelikeExcessGeV_ ==
          oldLedger.maximum_isotope_corrected_spacelike_excess_GeV);
    CHECK(ledger.minimumIsotopeCorrectedResidualMassSquaredGeV2_ ==
          oldLedger.minimum_isotope_corrected_residual_mass_squared_GeV2);
    CHECK(ledger.maximumTargetTotalEnergyGeV_ ==
          oldLedger.maximum_target_total_energy_GeV);
    CHECK(ledger.weightedMomentumResidualNormGeV_ ==
          oldLedger.weighted_momentum_residual_norm_GeV);
    CHECK(ledger.maximumMomentumResidualNormGeV_ ==
          oldLedger.maximum_momentum_residual_norm_GeV);
    CHECK(ledger.negativeEnergyResidualInteractions_ ==
          oldLedger.negative_energy_residual_interactions);
    CHECK(ledger.spacelikeResidualInteractions_ ==
          oldLedger.spacelike_residual_interactions);
    CHECK(ledger.maximumNegativeEnergyResidualGeV_ ==
          oldLedger.maximum_negative_energy_residual_GeV);
    CHECK(ledger.maximumNegativeProjectilePdg_ ==
          oldLedger.maximum_negative_projectile_pdg);
    CHECK(ledger.maximumNegativeTargetPdg_ == oldLedger.maximum_negative_target_pdg);
    CHECK(ledger.maximumNegativeProjectileTotalEnergyGeV_ ==
          oldLedger.maximum_negative_projectile_total_energy_GeV);
    CHECK(ledger.maximumNegativeTargetTotalEnergyGeV_ ==
          oldLedger.maximum_negative_target_total_energy_GeV);
    CHECK(ledger.maximumNegativeSecondaryTotalEnergyGeV_ ==
          oldLedger.maximum_negative_secondary_total_energy_GeV);
    CHECK(ledger.maximumNegativeSecondaryCount_ ==
          oldLedger.maximum_negative_secondary_count);
    CHECK(ledger.maximumNegativeBaryonNumberResidual_ ==
          oldLedger.maximum_negative_baryon_number_residual);
    CHECK(ledger.maximumNegativeChargeNumberResidual_ ==
          oldLedger.maximum_negative_charge_number_residual);
    CHECK(ledger.maximumNegativeSecondaryPdgs_ ==
          oldLedger.maximum_negative_secondary_pdgs);
    CHECK(ledger.maximumSpacelikeExcessGeV_ == oldLedger.maximum_spacelike_excess_GeV);
    CHECK(ledger.minimumResidualMassSquaredGeV2_ ==
          oldLedger.minimum_residual_mass_squared_GeV2);
    CHECK(ledger.maximumTimelikeResidualMassGeV_ ==
          oldLedger.maximum_timelike_residual_mass_GeV);
    CHECK(ledger.maximumPositiveResidualBeta_ ==
          oldLedger.maximum_positive_residual_beta);
  }
} // namespace counter_fixture
