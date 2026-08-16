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

  namespace interaction_counter_detail {
    inline int baryonNumber(Code const code) {
      if (is_nucleus(code)) {
        return static_cast<int>(get_nucleus_A(code));
      }
      auto const pdg = static_cast<int>(get_PDG(code));
      auto const absolute_pdg = std::abs(pdg);
      auto const is_baryon =
          absolute_pdg < 1000000000 &&
          ((absolute_pdg / 1000) % 10) != 0;
      return is_baryon ? (pdg < 0 ? -1 : 1) : 0;
    }
  } // namespace interaction_counter_detail

  template <class TCountedProcess>
  inline InteractionCounter<TCountedProcess>::InteractionCounter(TCountedProcess& process)
      : process_(process) {}

  template <class TCountedProcess>
  template <typename TSecondaryView>
  inline void InteractionCounter<TCountedProcess>::doInteraction(
      TSecondaryView& view, Code const projectileId, Code const targetId,
      FourMomentum const& projectileP4, FourMomentum const& targetP4) {
    size_t const massNumber = is_nucleus(targetId) ? get_nucleus_A(targetId) : 1;
    auto const massTarget = massNumber * constants::nucleonMass;
    histogram_.fill(projectileId, projectileP4.getTimeLikeComponent(), massTarget);
    if constexpr (
        has_hadronic_worker_model_v<TCountedProcess>) {
      if (auto* context =
              HadronicInteractionDeferralContext::current();
          context != nullptr &&
          context->tryPrepare(
              TCountedProcess::hadronic_worker_model,
              projectileId, targetId, projectileP4,
              targetP4)) {
        timing_samples_.push_back(
            InteractionTimingSample{
                count_, projectileId, targetId,
                projectileP4.getTimeLikeComponent() -
                    get_mass(projectileId),
                0., true});
        ++energy_ledger_statistics_.deferred_interactions;
        ++count_;
        return;
      }
    }
    auto const start = std::chrono::steady_clock::now();
    process_.doInteraction(view, projectileId, targetId, projectileP4, targetP4);
    auto const elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    total_final_state_time_ms_ += elapsed_ms;
    timing_samples_.push_back(
        InteractionTimingSample{
            count_, projectileId, targetId,
            projectileP4.getTimeLikeComponent() -
                get_mass(projectileId),
            elapsed_ms, false});
    if constexpr (
        interaction_counter_detail::CanAuditEnergy<
            TSecondaryView>::value) {
      auto const projectile_weight =
          view.getProjectile().getWeight();
      auto const weighted_projectile_energy_GeV =
          projectileP4.getTimeLikeComponent() / 1_GeV *
          projectile_weight;
      auto const weighted_target_energy_GeV =
          targetP4.getTimeLikeComponent() / 1_GeV *
          projectile_weight;
      double weighted_secondary_energy_GeV = 0.;
      double weighted_secondary_baryon_number = 0.;
      double weighted_secondary_charge_number = 0.;
      double secondary_baryon_number = 0.;
      double secondary_charge_number = 0.;
      std::uint64_t secondary_count = 0;
      std::vector<int> secondary_pdgs;
      auto momentum_residual =
          projectileP4.getSpaceLikeComponents() +
          targetP4.getSpaceLikeComponents();
      for (auto const& secondary : view) {
        ++secondary_count;
        secondary_pdgs.push_back(
            static_cast<int>(get_PDG(secondary.getPID())));
        weighted_secondary_energy_GeV +=
            secondary.getEnergy() / 1_GeV *
            secondary.getWeight();
        weighted_secondary_baryon_number +=
            interaction_counter_detail::baryonNumber(
                secondary.getPID()) *
            secondary.getWeight();
        secondary_baryon_number +=
            interaction_counter_detail::baryonNumber(
                secondary.getPID());
        weighted_secondary_charge_number +=
            get_charge_number(secondary.getPID()) *
            secondary.getWeight();
        secondary_charge_number +=
            get_charge_number(secondary.getPID());
        momentum_residual -= secondary.getMomentum();
      }
      auto const residual_GeV =
          weighted_projectile_energy_GeV +
          weighted_target_energy_GeV -
          weighted_secondary_energy_GeV;
      auto const projectile_baryon_number =
          interaction_counter_detail::baryonNumber(projectileId);
      auto const projectile_charge_number =
          get_charge_number(projectileId);
      auto const inferred_target_A =
          static_cast<long long>(std::llround(
              secondary_baryon_number -
              projectile_baryon_number));
      auto const inferred_target_Z =
          static_cast<long long>(std::llround(
              secondary_charge_number -
              projectile_charge_number));
      auto const inferred_target_is_valid =
          inferred_target_A >= 1 && inferred_target_Z >= 0 &&
          inferred_target_Z <= inferred_target_A &&
          std::abs(
              secondary_baryon_number -
              projectile_baryon_number - inferred_target_A) <
              1.e-9 &&
          std::abs(
              secondary_charge_number -
              projectile_charge_number - inferred_target_Z) <
              1.e-9;
      double weighted_effective_target_energy_GeV =
          weighted_target_energy_GeV;
      if (inferred_target_is_valid) {
        auto const effective_target_mass_GeV =
            get_nucleus_mass(
                static_cast<unsigned int>(inferred_target_A),
                static_cast<unsigned int>(inferred_target_Z)) /
            1_GeV;
        auto const declared_target_mass_GeV =
            get_mass(targetId) / 1_GeV;
        weighted_effective_target_energy_GeV +=
            (effective_target_mass_GeV -
             declared_target_mass_GeV) *
            projectile_weight;
      }
      auto const weighted_target_isotope_correction_GeV =
          weighted_effective_target_energy_GeV -
          weighted_target_energy_GeV;
      auto const isotope_corrected_residual_GeV =
          residual_GeV +
          weighted_target_isotope_correction_GeV;
      constexpr double residual_tolerance_GeV = 1.e-9;
      ++energy_ledger_statistics_.audited_interactions;
      energy_ledger_statistics_
          .weighted_projectile_total_energy_GeV +=
          weighted_projectile_energy_GeV;
      energy_ledger_statistics_
          .weighted_target_total_energy_GeV +=
          weighted_target_energy_GeV;
      energy_ledger_statistics_
          .weighted_effective_target_total_energy_GeV +=
          weighted_effective_target_energy_GeV;
      energy_ledger_statistics_
          .weighted_target_isotope_correction_GeV +=
          weighted_target_isotope_correction_GeV;
      if (std::abs(weighted_target_isotope_correction_GeV) >
          1.e-12) {
        ++energy_ledger_statistics_
              .target_isotope_adjusted_interactions;
      }
      energy_ledger_statistics_
          .weighted_secondary_total_energy_GeV +=
          weighted_secondary_energy_GeV;
      energy_ledger_statistics_.weighted_energy_residual_GeV +=
          residual_GeV;
      energy_ledger_statistics_
          .weighted_absolute_energy_residual_GeV +=
          std::abs(residual_GeV);
      energy_ledger_statistics_.maximum_absolute_residual_GeV =
          std::max(
            energy_ledger_statistics_.maximum_absolute_residual_GeV,
            std::abs(residual_GeV));
      energy_ledger_statistics_
          .weighted_isotope_corrected_energy_residual_GeV +=
          isotope_corrected_residual_GeV;
      energy_ledger_statistics_
          .weighted_absolute_isotope_corrected_energy_residual_GeV +=
          std::abs(isotope_corrected_residual_GeV);
      energy_ledger_statistics_
          .maximum_absolute_isotope_corrected_residual_GeV =
          std::max(
              energy_ledger_statistics_
                  .maximum_absolute_isotope_corrected_residual_GeV,
              std::abs(isotope_corrected_residual_GeV));
      if (isotope_corrected_residual_GeV <
          -residual_tolerance_GeV) {
        ++energy_ledger_statistics_
              .negative_isotope_corrected_residual_interactions;
        energy_ledger_statistics_
            .maximum_negative_isotope_corrected_residual_GeV =
            std::max(
                energy_ledger_statistics_
                    .maximum_negative_isotope_corrected_residual_GeV,
                -isotope_corrected_residual_GeV);
      }
      energy_ledger_statistics_.maximum_target_total_energy_GeV =
          std::max(
              energy_ledger_statistics_.maximum_target_total_energy_GeV,
              weighted_target_energy_GeV);
      auto const weighted_momentum_residual_norm_GeV =
          momentum_residual.getNorm() / 1_GeV *
          projectile_weight;
      auto const residual_mass_squared_GeV2 =
          residual_GeV * residual_GeV -
          weighted_momentum_residual_norm_GeV *
              weighted_momentum_residual_norm_GeV;
      auto const isotope_corrected_residual_mass_squared_GeV2 =
          isotope_corrected_residual_GeV *
              isotope_corrected_residual_GeV -
          weighted_momentum_residual_norm_GeV *
              weighted_momentum_residual_norm_GeV;
      energy_ledger_statistics_
          .minimum_isotope_corrected_residual_mass_squared_GeV2 =
          std::min(
              energy_ledger_statistics_
                  .minimum_isotope_corrected_residual_mass_squared_GeV2,
              isotope_corrected_residual_mass_squared_GeV2);
      if (isotope_corrected_residual_GeV >=
              -residual_tolerance_GeV &&
          weighted_momentum_residual_norm_GeV >
              std::max(0., isotope_corrected_residual_GeV) +
                  residual_tolerance_GeV) {
        ++energy_ledger_statistics_
              .spacelike_isotope_corrected_residual_interactions;
        energy_ledger_statistics_
            .maximum_isotope_corrected_spacelike_excess_GeV =
            std::max(
                energy_ledger_statistics_
                    .maximum_isotope_corrected_spacelike_excess_GeV,
                weighted_momentum_residual_norm_GeV -
                    isotope_corrected_residual_GeV);
      }
      auto const weighted_baryon_number_residual =
          (interaction_counter_detail::baryonNumber(projectileId) +
           interaction_counter_detail::baryonNumber(targetId)) *
              projectile_weight -
          weighted_secondary_baryon_number;
      auto const weighted_charge_number_residual =
          (get_charge_number(projectileId) +
           get_charge_number(targetId)) *
              projectile_weight -
          weighted_secondary_charge_number;
      energy_ledger_statistics_
          .weighted_momentum_residual_norm_GeV +=
          weighted_momentum_residual_norm_GeV;
      energy_ledger_statistics_
          .maximum_momentum_residual_norm_GeV =
          std::max(
              energy_ledger_statistics_
                  .maximum_momentum_residual_norm_GeV,
              weighted_momentum_residual_norm_GeV);
      if (residual_GeV < -residual_tolerance_GeV) {
        ++energy_ledger_statistics_
              .negative_energy_residual_interactions;
        if (-residual_GeV >
            energy_ledger_statistics_
                .maximum_negative_energy_residual_GeV) {
          energy_ledger_statistics_
              .maximum_negative_energy_residual_GeV =
              -residual_GeV;
          energy_ledger_statistics_
              .maximum_negative_projectile_pdg =
              static_cast<int>(get_PDG(projectileId));
          energy_ledger_statistics_
              .maximum_negative_target_pdg =
              static_cast<int>(get_PDG(targetId));
          energy_ledger_statistics_
              .maximum_negative_projectile_total_energy_GeV =
              weighted_projectile_energy_GeV;
          energy_ledger_statistics_
              .maximum_negative_target_total_energy_GeV =
              weighted_target_energy_GeV;
          energy_ledger_statistics_
              .maximum_negative_secondary_total_energy_GeV =
              weighted_secondary_energy_GeV;
          energy_ledger_statistics_
              .maximum_negative_secondary_count =
              secondary_count;
          energy_ledger_statistics_
              .maximum_negative_baryon_number_residual =
              weighted_baryon_number_residual;
          energy_ledger_statistics_
              .maximum_negative_charge_number_residual =
              weighted_charge_number_residual;
          energy_ledger_statistics_
              .maximum_negative_secondary_pdgs =
              std::move(secondary_pdgs);
        }
      } else {
        if (weighted_momentum_residual_norm_GeV >
            std::max(0., residual_GeV) +
                residual_tolerance_GeV) {
          ++energy_ledger_statistics_
                .spacelike_residual_interactions;
          energy_ledger_statistics_
              .maximum_spacelike_excess_GeV =
              std::max(
                  energy_ledger_statistics_
                      .maximum_spacelike_excess_GeV,
                  weighted_momentum_residual_norm_GeV -
                      residual_GeV);
        }
        if (residual_GeV > 0.) {
          energy_ledger_statistics_
              .maximum_positive_residual_beta =
              std::max(
                  energy_ledger_statistics_
                      .maximum_positive_residual_beta,
                  weighted_momentum_residual_norm_GeV /
                      residual_GeV);
        }
      }
      energy_ledger_statistics_
          .minimum_residual_mass_squared_GeV2 =
          std::min(
              energy_ledger_statistics_
                  .minimum_residual_mass_squared_GeV2,
              residual_mass_squared_GeV2);
      if (residual_mass_squared_GeV2 >= 0. &&
          residual_GeV >= 0.) {
        energy_ledger_statistics_
            .maximum_timelike_residual_mass_GeV =
            std::max(
                energy_ledger_statistics_
                    .maximum_timelike_residual_mass_GeV,
                std::sqrt(residual_mass_squared_GeV2));
      }
    }
    ++count_;
  }

  template <class TCountedProcess>
  inline CrossSectionType InteractionCounter<TCountedProcess>::getCrossSection(
      Code const projectileId, Code const targetId, FourMomentum const& projectileP4,
      FourMomentum const& targetP4) const {
    return process_.getCrossSection(projectileId, targetId, projectileP4, targetP4);
  }

  template <class TCountedProcess>
  inline InteractionHistogram const& InteractionCounter<TCountedProcess>::getHistogram()
      const {
    return histogram_;
  }

  template <class TCountedProcess>
  inline std::uint64_t InteractionCounter<TCountedProcess>::getCount() const {
    return count_;
  }

  template <class TCountedProcess>
  inline std::vector<InteractionTimingSample> const&
  InteractionCounter<TCountedProcess>::getTimingSamples() const {
    return timing_samples_;
  }

  template <class TCountedProcess>
  inline double
  InteractionCounter<TCountedProcess>::getTotalFinalStateTimeMs() const {
    return total_final_state_time_ms_;
  }

  template <class TCountedProcess>
  inline InteractionEnergyLedgerStatistics const&
  InteractionCounter<TCountedProcess>::getEnergyLedgerStatistics() const {
    return energy_ledger_statistics_;
  }

} // namespace corsika
