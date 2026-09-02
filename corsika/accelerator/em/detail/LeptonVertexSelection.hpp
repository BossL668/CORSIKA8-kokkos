/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/accelerator/em/detail/LeptonContinuousStep.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>

namespace corsika::accelerator::em::detail {

  struct LeptonVertexSelectionOutcome {
    gpu::em::EmInteractionRecord record{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t interaction_flag{};
    std::uint32_t continuation_flag{};
    std::uint32_t fallback_flag{};
    std::uint32_t native_newton_iterations{};
    std::uint32_t native_bisection_iterations{};
    std::uint32_t native_inverse_failures{};
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  vertexFallback(gpu::em::EmInteractionRecord const& candidate,
                 gpu::em::ProposalFallbackReason const reason) {
    auto event = gpu::em::makeProcessFallbackEvent(candidate, reason);
    event.input_index = candidate.input_index;
    return event;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  vertexTableFallback(
      gpu::em::EmInteractionRecord const& candidate,
      gpu::em::tables::TableQuery const& query,
      gpu::em::tables::TableQueryResult const& result,
      std::uint64_t const draw_id, double const loss_quantile = 0.) {
    auto event = gpu::em::makeTableFallbackEvent(
        candidate.particle, query, result, draw_id, candidate.input_index);
    auto const native_selection =
        candidate.proposal_selection_random_process_id != 0u;
    event.selection_uniform = native_selection
                                  ? candidate.proposal_selection_uniform
                                  : candidate.process_uniform;
    event.loss_quantile = loss_quantile;
    event.outer_acceptance_uniform = candidate.process_uniform;
    event.outer_acceptance_random_process_id =
        candidate.process_random_process_id;
    event.outer_acceptance_draw_id = candidate.process_draw_id;
    event.random_process_id = native_selection
                                  ? candidate.proposal_selection_random_process_id
                                  : 0u;
    if (native_selection) event.random_draw_id = candidate.proposal_selection_draw_id;
    return event;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonVertexSelectionOutcome
  selectLeptonVertex(
      gpu::em::tables::FlatRateTableView const& table,
      gpu::em::EmInteractionRecord const& candidate,
      std::uint64_t const random_seed, std::uint64_t const shower_id) {
    using namespace gpu::em;
    LeptonVertexSelectionOutcome output{};
      auto record = candidate;
      auto const& particle = record.particle;
      if (!isChargedLeptonPid(particle.pid)) {
        output.fallback = vertexFallback(
            record, ProposalFallbackReason::UnsupportedParticle);
        output.fallback_flag = 1;
        return output;
      }
      if (record.status !=
              EmInteractionStatus::RequiresReselection ||
          !leptonContinuousFinite(record.total_rate_cm2_per_g) ||
          !(record.total_rate_cm2_per_g > 0.) ||
          !leptonContinuousFinite(record.process_uniform) ||
          !(record.process_uniform > 0.) ||
          !(record.process_uniform < 1.)) {
        output.fallback = vertexFallback(
            record, ProposalFallbackReason::InvalidFinalState);
        output.fallback_flag = 1;
        return output;
      }

      auto const energy_MeV = particle.energy_GeV * 1000.;
      auto const outer_threshold =
          record.process_uniform *
          record.total_rate_cm2_per_g;
      auto const proposal_native = table.physics_source == 1u;
      tables::RateColumnSelectionResult selection{};
      if (proposal_native) {
        auto const vertex_total = tables::queryTotalRate(
            table, particle.pid, energy_MeV);
        if (vertex_total.status != tables::TableLookupStatus::Success) {
          tables::TableQuery const failed_total_query{
              tables::TableQueryKind::TotalRate, particle.pid,
              0, 0, energy_MeV, 0.};
          output.fallback = vertexTableFallback(
              record, failed_total_query, vertex_total,
              record.process_draw_id);
          output.fallback_flag = 1;
          return output;
        }
        record.vertex_total_rate_cm2_per_g = vertex_total.value;
        if (!(vertex_total.value > 0.) ||
            outer_threshold >= vertex_total.value) {
          if (record.particle.step_id == 0xffffffffffffffffULL) {
            output.fallback = vertexFallback(
                record, ProposalFallbackReason::InvalidFinalState);
            output.fallback_flag = 1;
            return output;
          }
          record.status = EmInteractionStatus::NoDiscreteInteraction;
          record.process_id = 0;
          record.component_hash = 0;
          record.energy_fraction = 0.;
          record.loss_quantile = 0.;
          record.particle.step_id++;
          output.record = record;
          output.continuation_flag = 1;
          return output;
        }
        RandomNumberKey const proposal_key{
            random_seed, shower_id, particle.history_id,
            particle.step_id, ProposalSelectionRandomProcessId,
            ProposalSelectionDrawId};
        record.proposal_selection_uniform = uniformOpen01(proposal_key);
        record.proposal_selection_random_process_id =
            ProposalSelectionRandomProcessId;
        record.proposal_selection_draw_id = ProposalSelectionDrawId;
        selection = tables::selectRateColumnByUniform(
            table, particle.pid, energy_MeV,
            record.proposal_selection_uniform);
      } else {
        selection = tables::selectRateColumnByThreshold(
            table, particle.pid, energy_MeV, outer_threshold);
      }
      if (selection.status != tables::TableLookupStatus::Success) {
        record.process_id = selection.process_id;
        record.component_hash = selection.component_hash;
        tables::TableQuery const failed_column_query{
            tables::TableQueryKind::Rate, particle.pid,
            selection.process_id, selection.component_hash,
            energy_MeV, 0.};
        auto const failure = tables::TableQueryResult{
            selection.status, 0, 0.};
        auto fallback = vertexTableFallback(
            record, failed_column_query, failure,
            record.process_draw_id);
        // Preserve the complete threshold inputs.  A non-finite selection
        // status cannot be diagnosed from the post-transport energy alone.
        fallback.diagnostic_value0 = energy_MeV;
        fallback.diagnostic_value1 = record.total_rate_cm2_per_g;
        fallback.diagnostic_value2 =
            proposal_native ? record.proposal_selection_uniform
                            : record.process_uniform;
        output.fallback = fallback;
        output.fallback_flag = 1;
        return output;
      }
      if (!proposal_native)
        record.vertex_total_rate_cm2_per_g = selection.total_rate;
      if (!proposal_native &&
          (!(selection.total_rate > 0.) ||
           outer_threshold >= selection.total_rate)) {
        if (record.particle.step_id ==
            0xffffffffffffffffULL) {
          output.fallback = vertexFallback(
              record, ProposalFallbackReason::InvalidFinalState);
          output.fallback_flag = 1;
          return output;
        }
        record.status =
            EmInteractionStatus::NoDiscreteInteraction;
        record.process_id = 0;
        record.component_hash = 0;
        record.energy_fraction = 0.;
        record.loss_quantile = 0.;
        record.particle.step_id++;
        output.record = record;
        output.continuation_flag = 1;
        return output;
      }
      if (selection.selected == 0) {
        output.fallback = vertexFallback(
            record, ProposalFallbackReason::ZeroTotalRate);
        output.fallback_flag = 1;
        return output;
      }

      record.process_id = selection.process_id;
      record.component_hash = selection.component_hash;
      if (proposal_native) {
        record.loss_quantile = selection.residual_quantile;
        record.loss_draw_id = ProposalSelectionDrawId;
        if (proposalNativeSelectionRequiresReplay(
                particle.pid, record.process_id,
                record.loss_quantile)) {
          output.fallback = vertexFallback(
              record,
              ProposalFallbackReason::NativeSelectionReplay);
          output.fallback_flag = 1;
          return output;
        }
      } else {
        RandomNumberKey const loss_key{
            random_seed, shower_id, particle.history_id,
            particle.step_id,
            static_cast<std::uint32_t>(record.process_id),
            InteractionLossDrawId};
        record.loss_quantile = uniformOpen01(loss_key);
        record.loss_draw_id = InteractionLossDrawId;
      }
      tables::TableQuery const loss_query{
          tables::TableQueryKind::LossFraction, particle.pid,
          record.process_id, record.component_hash,
          energy_MeV, record.loss_quantile};
      auto const loss =
          tables::executeTableQuery(table, loss_query);
      output.native_newton_iterations = tables::nativeNewtonIterations(loss);
      output.native_bisection_iterations = tables::nativeBisectionIterations(loss);
      output.native_inverse_failures =
          table.physics_source == 1u &&
                  loss.status != tables::TableLookupStatus::Success
              ? 1u
              : 0u;
      if (loss.status != tables::TableLookupStatus::Success) {
        output.fallback = vertexTableFallback(
            record, loss_query, loss,
            proposal_native ? ProposalSelectionDrawId
                            : InteractionLossDrawId,
            record.loss_quantile);
        output.fallback_flag = 1;
        return output;
      }
      record.status = EmInteractionStatus::Selected;
      record.energy_fraction = loss.value;
      output.record = record;
      output.interaction_flag = 1;
      return output;
  }

} // namespace corsika::accelerator::em::detail
