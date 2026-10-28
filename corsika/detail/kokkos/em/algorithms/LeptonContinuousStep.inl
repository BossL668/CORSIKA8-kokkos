/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline bool lepton_continuous_detail::finite(double const value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline FallbackRequest
  lepton_continuous_detail::make_transport_fallback(InteractionRecord const& interaction,
                                                    FallbackReason const reason) {
    auto event = make_process_fallback_event(interaction, reason);
    event.inputIndex_ = interaction.inputIndex_;
    return event;
  }

  CORSIKA_HOST_DEVICE inline FallbackRequest
  lepton_continuous_detail::make_table_fallback(InteractionRecord const& interaction,
                                                TableLookupStatus const status,
                                                double const query_stage,
                                                double const query_value) {
    auto event = lepton_continuous_detail::make_transport_fallback(
        interaction, proposal_fallback_reason(status));
    event.diagnosticStatus_ = static_cast<std::int32_t>(status);
    event.diagnosticValue0_ = query_stage;
    event.diagnosticValue1_ = query_value;
    return event;
  }

  CORSIKA_HOST_DEVICE inline LeptonContinuousPreparation prepare_lepton_continuous_step(
      ContinuousPhysicsView const& table, InteractionRecord const& interaction) {
    LeptonContinuousPreparation output{};
    auto const& start = interaction.particle_;
    auto const mass = query_continuous_transport_mass(table, start.pid_);
    auto const minimum_energy = query_continuous_minimum_energy(table, start.pid_);
    if (mass.status_ != TableLookupStatus::Success) {
      output.fallback_ =
          lepton_continuous_detail::make_table_fallback(interaction, mass.status_, 1.);
      output.fallbackFlag_ = 1;
      return output;
    }
    if (minimum_energy.status_ != TableLookupStatus::Success) {
      output.fallback_ = lepton_continuous_detail::make_table_fallback(
          interaction, minimum_energy.status_, 2.);
      output.fallbackFlag_ = 1;
      return output;
    }
    output.massMeV_ = mass.value_;
    output.minimumEnergyMeV_ = minimum_energy.value_;
    auto const initial_energy_MeV = start.energyGeV_ * 1000.;
    if (!(initial_energy_MeV > mass.value_)) {
      output.fallback_ = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::InvalidFinalState);
      output.fallbackFlag_ = 1;
      return output;
    }
    auto const transport_cut = query_continuous_transport_cut(table, start.pid_);
    output.transportCutMeV_ = transport_cut.value_;
    if (transport_cut.status_ != TableLookupStatus::Success ||
        !lepton_continuous_detail::finite(output.transportCutMeV_) ||
        !(output.transportCutMeV_ > 0.)) {
      output.fallback_ = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::InvalidTableQuery);
      output.fallback_.diagnosticValue0_ = mass.value_;
      output.fallback_.diagnosticValue1_ = minimum_energy.value_;
      output.fallback_.diagnosticValue2_ = output.transportCutMeV_;
      output.fallbackFlag_ = 1;
      return output;
    }
    if (interaction.status_ == InteractionStatus::ParticleCut ||
        initial_energy_MeV - mass.value_ < output.transportCutMeV_) {
      output.particleCut_ = 1;
      return output;
    }

    auto const initial_range =
        query_continuous_range(table, start.pid_, initial_energy_MeV);
    if (initial_range.status_ != TableLookupStatus::Success) {
      output.fallback_ = lepton_continuous_detail::make_table_fallback(
          interaction, initial_range.status_, 3., initial_energy_MeV);
      output.fallbackFlag_ = 1;
      return output;
    }
    auto const relative_limit = (1. - MaximumContinuousRelativeLoss) * initial_energy_MeV;
    auto const target_energy_MeV =
        relative_limit > minimum_energy.value_ ? relative_limit : minimum_energy.value_;
    auto const target_range =
        query_continuous_range(table, start.pid_, target_energy_MeV);
    if (target_range.status_ != TableLookupStatus::Success) {
      output.fallback_ = lepton_continuous_detail::make_table_fallback(
          interaction, target_range.status_, 4., target_energy_MeV);
      output.fallbackFlag_ = 1;
      return output;
    }
    output.maximumGrammageGPerCm2_ = initial_range.value_ - target_range.value_;
    auto const range_scale = initial_range.value_ > 1. ? initial_range.value_ : 1.;
    if (output.maximumGrammageGPerCm2_ < 0. &&
        output.maximumGrammageGPerCm2_ >
            -64. * 2.22044604925031308085e-16 * range_scale) {
      output.maximumGrammageGPerCm2_ = 0.;
    }
    if (!lepton_continuous_detail::finite(output.maximumGrammageGPerCm2_) ||
        output.maximumGrammageGPerCm2_ < 0.) {
      output.fallback_ = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::InvalidTableQuery);
      output.fallback_.diagnosticValue0_ = initial_range.value_;
      output.fallback_.diagnosticValue1_ = target_range.value_;
      output.fallback_.diagnosticValue2_ = output.maximumGrammageGPerCm2_;
      output.fallbackFlag_ = 1;
    }
    return output;
  }

  CORSIKA_HOST_DEVICE inline LeptonContinuousLoss evaluate_lepton_continuous_loss(
      ContinuousPhysicsView const& table, InteractionRecord const& interaction,
      LeptonContinuousPreparation const& preparation,
      double const traversed_grammage_g_per_cm2, bool const continuous_limit_won) {
    LeptonContinuousLoss output{};
    auto const initial_energy_MeV = interaction.particle_.energyGeV_ * 1000.;
    auto const final_energy = query_energy_after_continuous_loss(
        table, interaction.particle_.pid_, initial_energy_MeV,
        traversed_grammage_g_per_cm2);
    if (final_energy.status_ == TableLookupStatus::Success) {
      output.finalEnergyMeV_ = final_energy.value_;
    } else if (final_energy.status_ == TableLookupStatus::TransportCutReached) {
      output.finalEnergyMeV_ = preparation.minimumEnergyMeV_;
      output.transportCutReached_ = 1;
    } else {
      output.fallback_ = lepton_continuous_detail::make_table_fallback(
          interaction, final_energy.status_, 5., traversed_grammage_g_per_cm2);
      output.fallbackFlag_ = 1;
      return output;
    }
    auto const actual_loss_MeV = initial_energy_MeV - output.finalEnergyMeV_;
    if (continuous_limit_won && !(actual_loss_MeV > 0.)) {
      output.fallback_ = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::InvalidTableQuery);
      output.fallback_.diagnosticValue0_ = initial_energy_MeV;
      output.fallback_.diagnosticValue1_ = output.finalEnergyMeV_;
      output.fallback_.diagnosticValue2_ = traversed_grammage_g_per_cm2;
      output.fallbackFlag_ = 1;
      return output;
    }
    output.depositedEnergyGeV_ = actual_loss_MeV / 1000.;
    return output;
  }

} // namespace corsika::kokkos
