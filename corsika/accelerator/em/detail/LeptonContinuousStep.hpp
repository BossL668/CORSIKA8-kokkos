/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>

namespace corsika::accelerator::em::detail {

  /** Match the scalar Cascade continuous-loss step limiter. */
  inline constexpr double MaximumContinuousRelativeLoss = 0.10;

  struct LeptonContinuousPreparation {
    double mass_MeV{};
    double minimum_energy_MeV{};
    double transport_cut_MeV{};
    double maximum_grammage_g_per_cm2{};
    std::uint32_t particle_cut{};
    std::uint32_t fallback_flag{};
    gpu::em::ProposalFallbackEvent fallback{};
  };

  struct LeptonContinuousLoss {
    double final_energy_MeV{};
    double deposited_energy_GeV{};
    std::uint32_t transport_cut_reached{};
    std::uint32_t fallback_flag{};
    gpu::em::ProposalFallbackEvent fallback{};
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline bool leptonContinuousFinite(
      double const value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  makeLeptonTransportFallback(
      gpu::em::EmInteractionRecord const& interaction,
      gpu::em::ProposalFallbackReason const reason) {
    auto event = gpu::em::makeProcessFallbackEvent(interaction, reason);
    event.input_index = interaction.input_index;
    return event;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  makeLeptonTableFallback(
      gpu::em::EmInteractionRecord const& interaction,
      gpu::em::tables::TableLookupStatus const status,
      double const query_stage = 0., double const query_value = 0.) {
    auto event = makeLeptonTransportFallback(
        interaction, gpu::em::proposalFallbackReason(status));
    event.diagnostic_status = static_cast<std::int32_t>(status);
    event.diagnostic_value0 = query_stage;
    event.diagnostic_value1 = query_value;
    return event;
  }

  /**
   * Resolve the particle mass/cut and the maximum grammage allowed by the
   * scalar 10% continuous-loss limiter.  Both native CUDA and Kokkos call
   * this exact function, so proposal-native range interpolation and all
   * endpoint decisions have one implementation.
   */
  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonContinuousPreparation
  prepareLeptonContinuousStep(
      gpu::em::tables::FlatRateTableView const& table,
      gpu::em::EmInteractionRecord const& interaction) {
    using namespace gpu::em;
    using namespace gpu::em::tables;
    LeptonContinuousPreparation output{};
    auto const& start = interaction.particle;
    auto const mass = queryContinuousMass(table, start.pid);
    auto const minimum_energy = queryContinuousMinimumEnergy(table, start.pid);
    if (mass.status != TableLookupStatus::Success) {
      output.fallback = makeLeptonTableFallback(interaction, mass.status, 1.);
      output.fallback_flag = 1;
      return output;
    }
    if (minimum_energy.status != TableLookupStatus::Success) {
      output.fallback =
          makeLeptonTableFallback(interaction, minimum_energy.status, 2.);
      output.fallback_flag = 1;
      return output;
    }
    output.mass_MeV = mass.value;
    output.minimum_energy_MeV = minimum_energy.value;
    auto const initial_energy_MeV = start.energy_GeV * 1000.;
    if (!(initial_energy_MeV > mass.value)) {
      output.fallback = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::InvalidFinalState);
      output.fallback_flag = 1;
      return output;
    }
    output.transport_cut_MeV =
        (minimum_energy.value - mass.value) / ContinuousCutSafetyFactor;
    if (!leptonContinuousFinite(output.transport_cut_MeV) ||
        !(output.transport_cut_MeV > 0.)) {
      output.fallback = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::InvalidTableQuery);
      output.fallback.diagnostic_value0 = mass.value;
      output.fallback.diagnostic_value1 = minimum_energy.value;
      output.fallback.diagnostic_value2 = output.transport_cut_MeV;
      output.fallback_flag = 1;
      return output;
    }
    if (interaction.status == EmInteractionStatus::ParticleCut ||
        initial_energy_MeV - mass.value < output.transport_cut_MeV) {
      output.particle_cut = 1;
      return output;
    }

    auto const initial_range =
        queryContinuousRange(table, start.pid, initial_energy_MeV);
    if (initial_range.status != TableLookupStatus::Success) {
      output.fallback = makeLeptonTableFallback(
          interaction, initial_range.status, 3., initial_energy_MeV);
      output.fallback_flag = 1;
      return output;
    }
    auto const relative_limit =
        (1. - MaximumContinuousRelativeLoss) * initial_energy_MeV;
    auto const target_energy_MeV =
        relative_limit > minimum_energy.value ? relative_limit
                                              : minimum_energy.value;
    auto const target_range =
        queryContinuousRange(table, start.pid, target_energy_MeV);
    if (target_range.status != TableLookupStatus::Success) {
      output.fallback = makeLeptonTableFallback(
          interaction, target_range.status, 4., target_energy_MeV);
      output.fallback_flag = 1;
      return output;
    }
    output.maximum_grammage_g_per_cm2 =
        initial_range.value - target_range.value;
    auto const range_scale = initial_range.value > 1. ? initial_range.value : 1.;
    if (output.maximum_grammage_g_per_cm2 < 0. &&
        output.maximum_grammage_g_per_cm2 >
            -64. * 2.22044604925031308085e-16 * range_scale) {
      output.maximum_grammage_g_per_cm2 = 0.;
    }
    if (!leptonContinuousFinite(output.maximum_grammage_g_per_cm2) ||
        output.maximum_grammage_g_per_cm2 < 0.) {
      output.fallback = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::InvalidTableQuery);
      output.fallback.diagnostic_value0 = initial_range.value;
      output.fallback.diagnostic_value1 = target_range.value;
      output.fallback.diagnostic_value2 =
          output.maximum_grammage_g_per_cm2;
      output.fallback_flag = 1;
    }
    return output;
  }

  /** Evaluate the post-step range inversion and scalar ParticleCut state. */
  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonContinuousLoss
  evaluateLeptonContinuousLoss(
      gpu::em::tables::FlatRateTableView const& table,
      gpu::em::EmInteractionRecord const& interaction,
      LeptonContinuousPreparation const& preparation,
      double const traversed_grammage_g_per_cm2,
      bool const continuous_limit_won) {
    using namespace gpu::em;
    using namespace gpu::em::tables;
    LeptonContinuousLoss output{};
    auto const initial_energy_MeV = interaction.particle.energy_GeV * 1000.;
    auto const final_energy = queryEnergyAfterContinuousLoss(
        table, interaction.particle.pid, initial_energy_MeV,
        traversed_grammage_g_per_cm2);
    if (final_energy.status == TableLookupStatus::Success) {
      output.final_energy_MeV = final_energy.value;
    } else if (final_energy.status == TableLookupStatus::TransportCutReached) {
      output.final_energy_MeV = preparation.minimum_energy_MeV;
      output.transport_cut_reached = 1;
    } else {
      output.fallback = makeLeptonTableFallback(
          interaction, final_energy.status, 5.,
          traversed_grammage_g_per_cm2);
      output.fallback_flag = 1;
      return output;
    }
    auto const actual_loss_MeV =
        initial_energy_MeV - output.final_energy_MeV;
    if (continuous_limit_won && !(actual_loss_MeV > 0.)) {
      output.fallback = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::InvalidTableQuery);
      output.fallback.diagnostic_value0 = initial_energy_MeV;
      output.fallback.diagnostic_value1 = output.final_energy_MeV;
      output.fallback.diagnostic_value2 = traversed_grammage_g_per_cm2;
      output.fallback_flag = 1;
      return output;
    }
    output.deposited_energy_GeV = actual_loss_MeV / 1000.;
    return output;
  }

} // namespace corsika::accelerator::em::detail
