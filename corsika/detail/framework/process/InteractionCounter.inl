/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/InteractionHistogram.hpp>

#include <algorithm>
#include <cmath>

namespace corsika {

  inline bool NoInteractionDeferral::tryPrepare(Code const, Code const,
                                                FourMomentum const&,
                                                FourMomentum const&) noexcept {
    return false;
  }

  namespace interaction_counter_detail {
    inline int baryon_number(Code const code) {
      if (is_nucleus(code)) { return static_cast<int>(get_nucleus_A(code)); }
      auto const pdg = static_cast<int>(get_PDG(code));
      auto const absolute_pdg = std::abs(pdg);
      auto const is_baryon =
          absolute_pdg < 1000000000 && ((absolute_pdg / 1000) % 10) != 0;
      return is_baryon ? (pdg < 0 ? -1 : 1) : 0;
    }
  } // namespace interaction_counter_detail

  template <class TCountedProcess, typename TDeferral>
  inline InteractionCounter<TCountedProcess, TDeferral>::InteractionCounter(
      TCountedProcess& process)
      : process_(process) {}

  template <class TCountedProcess, typename TDeferral>
  template <typename TSecondaryView>
  inline void InteractionCounter<TCountedProcess, TDeferral>::doInteraction(
      TSecondaryView& view, Code const projectileId, Code const targetId,
      FourMomentum const& projectileP4, FourMomentum const& targetP4) {
    size_t const massNumber = is_nucleus(targetId) ? get_nucleus_A(targetId) : 1;
    auto const massTarget = massNumber * constants::nucleonMass;
    histogram_.fill(projectileId, projectileP4.getTimeLikeComponent(), massTarget);
    if (TDeferral::tryPrepare(projectileId, targetId, projectileP4, targetP4)) {
      timingSamples_.push_back(InteractionTimingSample{
          count_, projectileId, targetId,
          projectileP4.getTimeLikeComponent() - get_mass(projectileId), 0., true});
      ++energyLedgerStatistics_.deferredInteractions_;
      ++count_;
      return;
    }
    auto const start = std::chrono::steady_clock::now();
    process_.doInteraction(view, projectileId, targetId, projectileP4, targetP4);
    auto const elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    totalFinalStateTimeMs_ += elapsed_ms;
    timingSamples_.push_back(InteractionTimingSample{
        count_, projectileId, targetId,
        projectileP4.getTimeLikeComponent() - get_mass(projectileId), elapsed_ms, false});
    if constexpr (interaction_counter_detail::CanAuditEnergy<TSecondaryView>::value) {
      auto const projectile_weight = view.getProjectile().getWeight();
      auto const weighted_projectile_energy_GeV =
          projectileP4.getTimeLikeComponent() / 1_GeV * projectile_weight;
      auto const weighted_target_energy_GeV =
          targetP4.getTimeLikeComponent() / 1_GeV * projectile_weight;
      double weighted_secondary_energy_GeV = 0.;
      double weighted_secondary_baryon_number = 0.;
      double weighted_secondary_charge_number = 0.;
      double secondary_baryon_number = 0.;
      double secondary_charge_number = 0.;
      std::uint64_t secondary_count = 0;
      std::vector<int> secondary_pdgs;
      auto momentum_residual =
          projectileP4.getSpaceLikeComponents() + targetP4.getSpaceLikeComponents();
      for (auto const& secondary : view) {
        ++secondary_count;
        secondary_pdgs.push_back(static_cast<int>(get_PDG(secondary.getPID())));
        weighted_secondary_energy_GeV +=
            secondary.getEnergy() / 1_GeV * secondary.getWeight();
        weighted_secondary_baryon_number +=
            interaction_counter_detail::baryon_number(secondary.getPID()) *
            secondary.getWeight();
        secondary_baryon_number +=
            interaction_counter_detail::baryon_number(secondary.getPID());
        weighted_secondary_charge_number +=
            get_charge_number(secondary.getPID()) * secondary.getWeight();
        secondary_charge_number += get_charge_number(secondary.getPID());
        momentum_residual -= secondary.getMomentum();
      }
      auto const residual_GeV = weighted_projectile_energy_GeV +
                                weighted_target_energy_GeV -
                                weighted_secondary_energy_GeV;
      auto const projectile_baryon_number =
          interaction_counter_detail::baryon_number(projectileId);
      auto const projectile_charge_number = get_charge_number(projectileId);
      auto const inferred_target_A = static_cast<long long>(
          std::llround(secondary_baryon_number - projectile_baryon_number));
      auto const inferred_target_Z = static_cast<long long>(
          std::llround(secondary_charge_number - projectile_charge_number));
      auto const inferred_target_is_valid =
          inferred_target_A >= 1 && inferred_target_Z >= 0 &&
          inferred_target_Z <= inferred_target_A &&
          std::abs(secondary_baryon_number - projectile_baryon_number -
                   inferred_target_A) < 1.e-9 &&
          std::abs(secondary_charge_number - projectile_charge_number -
                   inferred_target_Z) < 1.e-9;
      double weighted_effective_target_energy_GeV = weighted_target_energy_GeV;
      if (inferred_target_is_valid) {
        auto const effective_target_mass_GeV =
            get_nucleus_mass(static_cast<unsigned int>(inferred_target_A),
                             static_cast<unsigned int>(inferred_target_Z)) /
            1_GeV;
        auto const declared_target_mass_GeV = get_mass(targetId) / 1_GeV;
        weighted_effective_target_energy_GeV +=
            (effective_target_mass_GeV - declared_target_mass_GeV) * projectile_weight;
      }
      auto const weighted_target_isotope_correction_GeV =
          weighted_effective_target_energy_GeV - weighted_target_energy_GeV;
      auto const isotope_corrected_residual_GeV =
          residual_GeV + weighted_target_isotope_correction_GeV;
      constexpr double residual_tolerance_GeV = 1.e-9;
      ++energyLedgerStatistics_.auditedInteractions_;
      energyLedgerStatistics_.weightedProjectileTotalEnergyGeV_ +=
          weighted_projectile_energy_GeV;
      energyLedgerStatistics_.weightedTargetTotalEnergyGeV_ += weighted_target_energy_GeV;
      energyLedgerStatistics_.weightedEffectiveTargetTotalEnergyGeV_ +=
          weighted_effective_target_energy_GeV;
      energyLedgerStatistics_.weightedTargetIsotopeCorrectionGeV_ +=
          weighted_target_isotope_correction_GeV;
      if (std::abs(weighted_target_isotope_correction_GeV) > 1.e-12) {
        ++energyLedgerStatistics_.targetIsotopeAdjustedInteractions_;
      }
      energyLedgerStatistics_.weightedSecondaryTotalEnergyGeV_ +=
          weighted_secondary_energy_GeV;
      energyLedgerStatistics_.weightedEnergyResidualGeV_ += residual_GeV;
      energyLedgerStatistics_.weightedAbsoluteEnergyResidualGeV_ +=
          std::abs(residual_GeV);
      energyLedgerStatistics_.maximumAbsoluteResidualGeV_ = std::max(
          energyLedgerStatistics_.maximumAbsoluteResidualGeV_, std::abs(residual_GeV));
      energyLedgerStatistics_.weightedIsotopeCorrectedEnergyResidualGeV_ +=
          isotope_corrected_residual_GeV;
      energyLedgerStatistics_.weightedAbsoluteIsotopeCorrectedEnergyResidualGeV_ +=
          std::abs(isotope_corrected_residual_GeV);
      energyLedgerStatistics_.maximumAbsoluteIsotopeCorrectedResidualGeV_ =
          std::max(energyLedgerStatistics_.maximumAbsoluteIsotopeCorrectedResidualGeV_,
                   std::abs(isotope_corrected_residual_GeV));
      if (isotope_corrected_residual_GeV < -residual_tolerance_GeV) {
        ++energyLedgerStatistics_.negativeIsotopeCorrectedResidualInteractions_;
        energyLedgerStatistics_.maximumNegativeIsotopeCorrectedResidualGeV_ =
            std::max(energyLedgerStatistics_.maximumNegativeIsotopeCorrectedResidualGeV_,
                     -isotope_corrected_residual_GeV);
      }
      energyLedgerStatistics_.maximumTargetTotalEnergyGeV_ =
          std::max(energyLedgerStatistics_.maximumTargetTotalEnergyGeV_,
                   weighted_target_energy_GeV);
      auto const weighted_momentum_residual_norm_GeV =
          momentum_residual.getNorm() / 1_GeV * projectile_weight;
      auto const residual_mass_squared_GeV2 =
          residual_GeV * residual_GeV -
          weighted_momentum_residual_norm_GeV * weighted_momentum_residual_norm_GeV;
      auto const isotope_corrected_residual_mass_squared_GeV2 =
          isotope_corrected_residual_GeV * isotope_corrected_residual_GeV -
          weighted_momentum_residual_norm_GeV * weighted_momentum_residual_norm_GeV;
      energyLedgerStatistics_.minimumIsotopeCorrectedResidualMassSquaredGeV2_ = std::min(
          energyLedgerStatistics_.minimumIsotopeCorrectedResidualMassSquaredGeV2_,
          isotope_corrected_residual_mass_squared_GeV2);
      if (isotope_corrected_residual_GeV >= -residual_tolerance_GeV &&
          weighted_momentum_residual_norm_GeV >
              std::max(0., isotope_corrected_residual_GeV) + residual_tolerance_GeV) {
        ++energyLedgerStatistics_.spacelikeIsotopeCorrectedResidualInteractions_;
        energyLedgerStatistics_.maximumIsotopeCorrectedSpacelikeExcessGeV_ = std::max(
            energyLedgerStatistics_.maximumIsotopeCorrectedSpacelikeExcessGeV_,
            weighted_momentum_residual_norm_GeV - isotope_corrected_residual_GeV);
      }
      auto const weighted_baryon_number_residual =
          (interaction_counter_detail::baryon_number(projectileId) +
           interaction_counter_detail::baryon_number(targetId)) *
              projectile_weight -
          weighted_secondary_baryon_number;
      auto const weighted_charge_number_residual =
          (get_charge_number(projectileId) + get_charge_number(targetId)) *
              projectile_weight -
          weighted_secondary_charge_number;
      energyLedgerStatistics_.weightedMomentumResidualNormGeV_ +=
          weighted_momentum_residual_norm_GeV;
      energyLedgerStatistics_.maximumMomentumResidualNormGeV_ =
          std::max(energyLedgerStatistics_.maximumMomentumResidualNormGeV_,
                   weighted_momentum_residual_norm_GeV);
      if (residual_GeV < -residual_tolerance_GeV) {
        ++energyLedgerStatistics_.negativeEnergyResidualInteractions_;
        if (-residual_GeV > energyLedgerStatistics_.maximumNegativeEnergyResidualGeV_) {
          energyLedgerStatistics_.maximumNegativeEnergyResidualGeV_ = -residual_GeV;
          energyLedgerStatistics_.maximumNegativeProjectilePdg_ =
              static_cast<int>(get_PDG(projectileId));
          energyLedgerStatistics_.maximumNegativeTargetPdg_ =
              static_cast<int>(get_PDG(targetId));
          energyLedgerStatistics_.maximumNegativeProjectileTotalEnergyGeV_ =
              weighted_projectile_energy_GeV;
          energyLedgerStatistics_.maximumNegativeTargetTotalEnergyGeV_ =
              weighted_target_energy_GeV;
          energyLedgerStatistics_.maximumNegativeSecondaryTotalEnergyGeV_ =
              weighted_secondary_energy_GeV;
          energyLedgerStatistics_.maximumNegativeSecondaryCount_ = secondary_count;
          energyLedgerStatistics_.maximumNegativeBaryonNumberResidual_ =
              weighted_baryon_number_residual;
          energyLedgerStatistics_.maximumNegativeChargeNumberResidual_ =
              weighted_charge_number_residual;
          energyLedgerStatistics_.maximumNegativeSecondaryPdgs_ =
              std::move(secondary_pdgs);
        }
      } else {
        if (weighted_momentum_residual_norm_GeV >
            std::max(0., residual_GeV) + residual_tolerance_GeV) {
          ++energyLedgerStatistics_.spacelikeResidualInteractions_;
          energyLedgerStatistics_.maximumSpacelikeExcessGeV_ =
              std::max(energyLedgerStatistics_.maximumSpacelikeExcessGeV_,
                       weighted_momentum_residual_norm_GeV - residual_GeV);
        }
        if (residual_GeV > 0.) {
          energyLedgerStatistics_.maximumPositiveResidualBeta_ =
              std::max(energyLedgerStatistics_.maximumPositiveResidualBeta_,
                       weighted_momentum_residual_norm_GeV / residual_GeV);
        }
      }
      energyLedgerStatistics_.minimumResidualMassSquaredGeV2_ =
          std::min(energyLedgerStatistics_.minimumResidualMassSquaredGeV2_,
                   residual_mass_squared_GeV2);
      if (residual_mass_squared_GeV2 >= 0. && residual_GeV >= 0.) {
        energyLedgerStatistics_.maximumTimelikeResidualMassGeV_ =
            std::max(energyLedgerStatistics_.maximumTimelikeResidualMassGeV_,
                     std::sqrt(residual_mass_squared_GeV2));
      }
    }
    ++count_;
  }

  template <class TCountedProcess, typename TDeferral>
  inline CrossSectionType InteractionCounter<TCountedProcess, TDeferral>::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {
    return process_.getCrossSection(projectileId, targetId, projectileP4, targetP4);
  }

  template <class TCountedProcess, typename TDeferral>
  inline InteractionHistogram const&
  InteractionCounter<TCountedProcess, TDeferral>::getHistogram() const {
    return histogram_;
  }

  template <class TCountedProcess, typename TDeferral>
  inline std::uint64_t InteractionCounter<TCountedProcess, TDeferral>::getCount() const {
    return count_;
  }

  template <class TCountedProcess, typename TDeferral>
  inline std::vector<InteractionTimingSample> const&
  InteractionCounter<TCountedProcess, TDeferral>::getTimingSamples() const {
    return timingSamples_;
  }

  template <class TCountedProcess, typename TDeferral>
  inline double InteractionCounter<TCountedProcess, TDeferral>::getTotalFinalStateTimeMs()
      const {
    return totalFinalStateTimeMs_;
  }

  template <class TCountedProcess, typename TDeferral>
  inline InteractionEnergyLedgerStatistics const&
  InteractionCounter<TCountedProcess, TDeferral>::getEnergyLedgerStatistics() const {
    return energyLedgerStatistics_;
  }

} // namespace corsika
