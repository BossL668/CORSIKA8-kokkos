/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/accelerator/em/common/Philox.hpp>
#include <corsika/accelerator/em/common/ProposalFallback.hpp>

namespace corsika::accelerator::em::detail {

  namespace table = gpu::em::tables;

  /** One particle's deterministic selection result before stable compaction. */
  struct InteractionSelectionOutcome {
    gpu::em::EmInteractionRecord interaction{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t fallback_flag{};
    std::uint32_t native_newton_iterations{};
    std::uint32_t native_bisection_iterations{};
    std::uint32_t native_inverse_failures{};
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline InteractionSelectionOutcome
  selectDiscreteInteraction(
      table::NativePhysicsView const& physics,
      gpu::em::EmParticleState const& particle,
      std::uint64_t const input_index, std::uint64_t const random_seed,
      std::uint64_t const shower_id) {
    using namespace gpu::em;
    InteractionSelectionOutcome output{};
    auto& record = output.interaction;
    record.particle = particle;
    record.input_index = input_index;
    record.distance_draw_id = InteractionDistanceDrawId;
    record.process_draw_id = InteractionColumnDrawId;
    record.loss_draw_id = InteractionLossDrawId;
    record.process_random_process_id = InteractionColumnRandomProcessId;
    auto const infinity = HUGE_VAL;
    auto const energy_MeV = particle.energy_GeV * 1000.;

    if (exceedsParticleCutTime(particle.time_s)) {
      record.status = EmInteractionStatus::ParticleCut;
      record.interaction_grammage_g_per_cm2 = infinity;
      return output;
    }

    auto const charged_lepton = isChargedLeptonPid(particle.pid);
    auto const photon =
        particle.pid == static_cast<std::int32_t>(EmPid::Photon);
    if (photon && energy_MeV < physics.em_transport_cut_MeV) {
      record.status = EmInteractionStatus::ParticleCut;
      record.interaction_grammage_g_per_cm2 = infinity;
      return output;
    }
    if (charged_lepton) {
      auto const mass = table::queryContinuousMass(physics, particle.pid);
      auto const transport_mass =
          table::queryContinuousTransportMass(physics, particle.pid);
      if (mass.status == table::TableLookupStatus::Success) {
        record.particle_mass_GeV = mass.value / 1000.;
        if (isMuonPid(particle.pid) &&
            transport_mass.status == table::TableLookupStatus::Success &&
            particle.energy_GeV > transport_mass.value / 1000.) {
          RandomNumberKey const decay_key{
              random_seed, shower_id, particle.history_id, particle.step_id,
              MuonDecayRandomProcessId, MuonDecayDrawId};
          record.decay_uniform = uniformOpen01(decay_key);
          record.decay_draw_id = MuonDecayDrawId;
          auto const momentum_GeV = ::sqrt(
              (particle.energy_GeV - transport_mass.value / 1000.) *
              (particle.energy_GeV + transport_mass.value / 1000.));
          record.decay_distance_m =
              -::log(record.decay_uniform) * MuonDecaySpeedOfLightMPerS *
              MuonMeanLifetimeS * momentum_GeV / (transport_mass.value / 1000.);
        }
      }
      auto const minimum_energy =
          table::queryContinuousMinimumEnergy(physics, particle.pid);
      auto const transport_cut =
          table::queryContinuousTransportCut(physics, particle.pid);
      auto const below_cut =
          transport_mass.status == table::TableLookupStatus::Success &&
          transport_cut.status == table::TableLookupStatus::Success &&
          energy_MeV - transport_mass.value < transport_cut.value;
      if (below_cut ||
          (minimum_energy.status == table::TableLookupStatus::Success &&
           energy_MeV <= minimum_energy.value)) {
        record.status = EmInteractionStatus::ParticleCut;
        record.interaction_grammage_g_per_cm2 = infinity;
        return output;
      }
    }

    table::TableQuery const total_query{
        table::TableQueryKind::TotalRate, particle.pid, 0, 0, energy_MeV, 0.};
    auto const total = table::executeTableQuery(physics, total_query);
    if (total.status != table::TableLookupStatus::Success) {
      output.fallback = makeTableFallbackEvent(
          particle, total_query, total, 0, input_index);
      output.fallback_flag = 1;
      return output;
    }
    record.total_rate_cm2_per_g = total.value;
    record.vertex_total_rate_cm2_per_g = total.value;
    if (!(total.value > 0.)) {
      record.status = EmInteractionStatus::NoDiscreteInteraction;
      record.interaction_grammage_g_per_cm2 = infinity;
      return output;
    }

    RandomNumberKey const distance_key{
        random_seed, shower_id, particle.history_id, particle.step_id,
        InteractionDistanceRandomProcessId, InteractionDistanceDrawId};
    record.distance_uniform = uniformOpen01(distance_key);
    record.interaction_grammage_g_per_cm2 =
        -::log(record.distance_uniform) / total.value;
    RandomNumberKey const column_key{
        random_seed, shower_id, particle.history_id, particle.step_id,
        InteractionColumnRandomProcessId, InteractionColumnDrawId};
    record.process_uniform = uniformOpen01(column_key);

    // The process is selected at post-continuous-loss energy for charged
    // particles, exactly as in the scalar and native-CUDA paths.
    if (charged_lepton) {
      record.status = EmInteractionStatus::DistanceSampled;
      return output;
    }

    constexpr bool proposal_native = true;
    table::RateColumnSelectionResult selection{};
    
      RandomNumberKey const proposal_key{
          random_seed, shower_id, particle.history_id, particle.step_id,
          ProposalSelectionRandomProcessId, ProposalSelectionDrawId};
      record.proposal_selection_uniform = uniformOpen01(proposal_key);
      record.proposal_selection_random_process_id =
          ProposalSelectionRandomProcessId;
      record.proposal_selection_draw_id = ProposalSelectionDrawId;
      selection = table::selectRateColumnByUniform(
          physics, particle.pid, energy_MeV,
          record.proposal_selection_uniform);
    
    if (selection.status != table::TableLookupStatus::Success) {
      auto const result = table::TableQueryResult{selection.status, 0, 0.};
      output.fallback = makeTableFallbackEvent(
          particle, total_query, result, 0, input_index);
      
        output.fallback.selection_uniform =
            record.proposal_selection_uniform;
        output.fallback.outer_acceptance_uniform = record.process_uniform;
        output.fallback.random_process_id = ProposalSelectionRandomProcessId;
        output.fallback.random_draw_id = ProposalSelectionDrawId;
        output.fallback.outer_acceptance_random_process_id =
            InteractionColumnRandomProcessId;
        output.fallback.outer_acceptance_draw_id = InteractionColumnDrawId;
      
      output.fallback_flag = 1;
      return output;
    }
    if (selection.selected == 0) {
      output.fallback.particle = particle;
      output.fallback.input_index = input_index;
      output.fallback.reason = ProposalFallbackReason::ZeroTotalRate;
      output.fallback_flag = 1;
      return output;
    }

    record.process_id = selection.process_id;
    record.component_hash = selection.component_hash;
    
      record.loss_quantile = selection.residual_quantile;
      if (proposalNativeSelectionRequiresReplay(
              particle.pid, record.process_id, record.loss_quantile)) {
        // Selection is not a vertex. The photon must first fly the sampled
        // grammage (or stop at an earlier boundary/cut). Final-state
        // classification completes this request only at the transported vertex.
        record.deferred_photon_fallback_reason = static_cast<std::int32_t>(
            ProposalFallbackReason::NativeSelectionReplay);
        return output;
      }
    
    table::TableQuery const loss_query{
        table::TableQueryKind::LossFraction, particle.pid, record.process_id,
        record.component_hash, energy_MeV, record.loss_quantile};
    auto const loss = table::executeTableQuery(physics, loss_query);
    
      output.native_newton_iterations = table::nativeNewtonIterations(loss);
      output.native_bisection_iterations =
          table::nativeBisectionIterations(loss);
      output.native_inverse_failures =
          loss.status == table::TableLookupStatus::Success ? 0u : 1u;
    
    if (loss.status != table::TableLookupStatus::Success) {
      auto const reason = proposalFallbackReason(loss.status);
      if (proposalFallbackRequiresSelectedLoss(reason)) {
        record.deferred_photon_fallback_reason = static_cast<std::int32_t>(reason);
        record.deferred_photon_table_status = static_cast<std::int32_t>(loss.status);
        record.energy_fraction = loss.value; // diagnostic only; CPU completes v
        return output;
      }
      output.fallback = makeTableFallbackEvent(
          particle, loss_query, loss,
          proposal_native ? ProposalSelectionDrawId : InteractionLossDrawId,
          input_index);
      output.fallback.loss_quantile = record.loss_quantile;
      output.fallback.selection_uniform =
          proposal_native ? record.proposal_selection_uniform
                          : record.process_uniform;
      output.fallback.outer_acceptance_uniform = record.process_uniform;
      output.fallback.random_process_id =
          proposal_native ? ProposalSelectionRandomProcessId : 0u;
      output.fallback.outer_acceptance_random_process_id =
          record.process_random_process_id;
      output.fallback.outer_acceptance_draw_id = record.process_draw_id;
      output.fallback_flag = 1;
      return output;
    }

    record.status = EmInteractionStatus::Selected;
    record.energy_fraction = loss.value;
    return output;
  }

} // namespace corsika::accelerator::em::detail
