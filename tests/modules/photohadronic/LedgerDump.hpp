/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/modules/proposal/HadronicPhotonModel.hpp>
#include <ostream>
// Explicit member-spelling adapter; does not reproduce ledger arithmetic.
inline double print_ledger(
    corsika::proposal::HadronicPhotonEnergyLedgerStatistics const& values,
    std::ostream& output) {
#ifdef ORIGINAL_PRODUCTION
  auto const& interactions_ = values.interactions;
#else
  auto const& interactions_ = values.interactions_;
#endif
  output << "ledger interactions " << interactions_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& lowEnergyInteractions_ = values.low_energy_interactions;
#else
  auto const& lowEnergyInteractions_ = values.lowEnergyInteractions_;
#endif
  output << "ledger low_energy_interactions " << lowEnergyInteractions_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& highEnergyInteractions_ = values.high_energy_interactions;
#else
  auto const& highEnergyInteractions_ = values.highEnergyInteractions_;
#endif
  output << "ledger high_energy_interactions " << highEnergyInteractions_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& weightedProjectileTotalEnergyGeV_ =
      values.weighted_projectile_total_energy_GeV;
#else
  auto const& weightedProjectileTotalEnergyGeV_ =
      values.weightedProjectileTotalEnergyGeV_;
#endif
  output << "ledger weighted_projectile_total_energy_GeV "
         << weightedProjectileTotalEnergyGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& weightedTargetTotalEnergyGeV_ = values.weighted_target_total_energy_GeV;
#else
  auto const& weightedTargetTotalEnergyGeV_ = values.weightedTargetTotalEnergyGeV_;
#endif
  output << "ledger weighted_target_total_energy_GeV " << weightedTargetTotalEnergyGeV_
         << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& weightedSecondaryTotalEnergyGeV_ =
      values.weighted_secondary_total_energy_GeV;
#else
  auto const& weightedSecondaryTotalEnergyGeV_ = values.weightedSecondaryTotalEnergyGeV_;
#endif
  output << "ledger weighted_secondary_total_energy_GeV "
         << weightedSecondaryTotalEnergyGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& weightedEnergyResidualGeV_ = values.weighted_energy_residual_GeV;
#else
  auto const& weightedEnergyResidualGeV_ = values.weightedEnergyResidualGeV_;
#endif
  output << "ledger weighted_energy_residual_GeV " << weightedEnergyResidualGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& weightedAbsoluteEnergyResidualGeV_ =
      values.weighted_absolute_energy_residual_GeV;
#else
  auto const& weightedAbsoluteEnergyResidualGeV_ =
      values.weightedAbsoluteEnergyResidualGeV_;
#endif
  output << "ledger weighted_absolute_energy_residual_GeV "
         << weightedAbsoluteEnergyResidualGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumAbsoluteResidualGeV_ = values.maximum_absolute_residual_GeV;
#else
  auto const& maximumAbsoluteResidualGeV_ = values.maximumAbsoluteResidualGeV_;
#endif
  output << "ledger maximum_absolute_residual_GeV " << maximumAbsoluteResidualGeV_
         << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumTargetTotalEnergyGeV_ = values.maximum_target_total_energy_GeV;
#else
  auto const& maximumTargetTotalEnergyGeV_ = values.maximumTargetTotalEnergyGeV_;
#endif
  output << "ledger maximum_target_total_energy_GeV " << maximumTargetTotalEnergyGeV_
         << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& weightedMomentumResidualNormGeV_ =
      values.weighted_momentum_residual_norm_GeV;
#else
  auto const& weightedMomentumResidualNormGeV_ = values.weightedMomentumResidualNormGeV_;
#endif
  output << "ledger weighted_momentum_residual_norm_GeV "
         << weightedMomentumResidualNormGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumMomentumResidualNormGeV_ = values.maximum_momentum_residual_norm_GeV;
#else
  auto const& maximumMomentumResidualNormGeV_ = values.maximumMomentumResidualNormGeV_;
#endif
  output << "ledger maximum_momentum_residual_norm_GeV "
         << maximumMomentumResidualNormGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& negativeEnergyResidualInteractions_ =
      values.negative_energy_residual_interactions;
#else
  auto const& negativeEnergyResidualInteractions_ =
      values.negativeEnergyResidualInteractions_;
#endif
  output << "ledger negative_energy_residual_interactions "
         << negativeEnergyResidualInteractions_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& spacelikeResidualInteractions_ = values.spacelike_residual_interactions;
#else
  auto const& spacelikeResidualInteractions_ = values.spacelikeResidualInteractions_;
#endif
  output << "ledger spacelike_residual_interactions " << spacelikeResidualInteractions_
         << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeEnergyResidualGeV_ =
      values.maximum_negative_energy_residual_GeV;
#else
  auto const& maximumNegativeEnergyResidualGeV_ =
      values.maximumNegativeEnergyResidualGeV_;
#endif
  output << "ledger maximum_negative_energy_residual_GeV "
         << maximumNegativeEnergyResidualGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeProjectilePdg_ = values.maximum_negative_projectile_pdg;
#else
  auto const& maximumNegativeProjectilePdg_ = values.maximumNegativeProjectilePdg_;
#endif
  output << "ledger maximum_negative_projectile_pdg " << maximumNegativeProjectilePdg_
         << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeTargetPdg_ = values.maximum_negative_target_pdg;
#else
  auto const& maximumNegativeTargetPdg_ = values.maximumNegativeTargetPdg_;
#endif
  output << "ledger maximum_negative_target_pdg " << maximumNegativeTargetPdg_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeProjectileTotalEnergyGeV_ =
      values.maximum_negative_projectile_total_energy_GeV;
#else
  auto const& maximumNegativeProjectileTotalEnergyGeV_ =
      values.maximumNegativeProjectileTotalEnergyGeV_;
#endif
  output << "ledger maximum_negative_projectile_total_energy_GeV "
         << maximumNegativeProjectileTotalEnergyGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeTargetTotalEnergyGeV_ =
      values.maximum_negative_target_total_energy_GeV;
#else
  auto const& maximumNegativeTargetTotalEnergyGeV_ =
      values.maximumNegativeTargetTotalEnergyGeV_;
#endif
  output << "ledger maximum_negative_target_total_energy_GeV "
         << maximumNegativeTargetTotalEnergyGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeSecondaryTotalEnergyGeV_ =
      values.maximum_negative_secondary_total_energy_GeV;
#else
  auto const& maximumNegativeSecondaryTotalEnergyGeV_ =
      values.maximumNegativeSecondaryTotalEnergyGeV_;
#endif
  output << "ledger maximum_negative_secondary_total_energy_GeV "
         << maximumNegativeSecondaryTotalEnergyGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeSecondaryCount_ = values.maximum_negative_secondary_count;
#else
  auto const& maximumNegativeSecondaryCount_ = values.maximumNegativeSecondaryCount_;
#endif
  output << "ledger maximum_negative_secondary_count " << maximumNegativeSecondaryCount_
         << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeBaryonNumberResidual_ =
      values.maximum_negative_baryon_number_residual;
#else
  auto const& maximumNegativeBaryonNumberResidual_ =
      values.maximumNegativeBaryonNumberResidual_;
#endif
  output << "ledger maximum_negative_baryon_number_residual "
         << maximumNegativeBaryonNumberResidual_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeChargeNumberResidual_ =
      values.maximum_negative_charge_number_residual;
#else
  auto const& maximumNegativeChargeNumberResidual_ =
      values.maximumNegativeChargeNumberResidual_;
#endif
  output << "ledger maximum_negative_charge_number_residual "
         << maximumNegativeChargeNumberResidual_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumNegativeSecondaryPdgs_ = values.maximum_negative_secondary_pdgs;
#else
  auto const& maximumNegativeSecondaryPdgs_ = values.maximumNegativeSecondaryPdgs_;
#endif
  output << "ledger maximum_negative_secondary_pdgs";
  for (auto pdg : maximumNegativeSecondaryPdgs_) output << ' ' << pdg;
  output << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumSpacelikeExcessGeV_ = values.maximum_spacelike_excess_GeV;
#else
  auto const& maximumSpacelikeExcessGeV_ = values.maximumSpacelikeExcessGeV_;
#endif
  output << "ledger maximum_spacelike_excess_GeV " << maximumSpacelikeExcessGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& minimumResidualMassSquaredGeV2_ = values.minimum_residual_mass_squared_GeV2;
#else
  auto const& minimumResidualMassSquaredGeV2_ = values.minimumResidualMassSquaredGeV2_;
#endif
  output << "ledger minimum_residual_mass_squared_GeV2 "
         << minimumResidualMassSquaredGeV2_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumTimelikeResidualMassGeV_ = values.maximum_timelike_residual_mass_GeV;
#else
  auto const& maximumTimelikeResidualMassGeV_ = values.maximumTimelikeResidualMassGeV_;
#endif
  output << "ledger maximum_timelike_residual_mass_GeV "
         << maximumTimelikeResidualMassGeV_ << '\n';
#ifdef ORIGINAL_PRODUCTION
  auto const& maximumPositiveResidualBeta_ = values.maximum_positive_residual_beta;
#else
  auto const& maximumPositiveResidualBeta_ = values.maximumPositiveResidualBeta_;
#endif
  output << "ledger maximum_positive_residual_beta " << maximumPositiveResidualBeta_
         << '\n';
  return weightedTargetTotalEnergyGeV_;
}
