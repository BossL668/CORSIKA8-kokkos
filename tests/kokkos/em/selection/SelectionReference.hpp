/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Immutable production 41cdc543. Only namespaces, includes and device annotations
// are adapted; the reference never calls the receiver selection implementation.
#include "../stochastic/StochasticReference.hpp"
namespace stochastic_reference::tables {
  struct RateColumnSelectionResult {
    TableLookupStatus status{TableLookupStatus::InvalidTableView};
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    double total_rate{};
    double residual_quantile{};
    std::uint32_t selected{};
  };
  CORSIKA_HOST_DEVICE inline RateColumnSelectionResult selectRateColumnByUniform(
      NativePhysicsView const& view, std::int32_t pid, double energy, double uniform) {
    auto const result = discrete_reference::selectRateColumnByUniform(
        discrete_view(view), pid, energy, uniform);
    return {static_cast<TableLookupStatus>(result.status),
            result.process_id,
            result.component_hash,
            result.total_rate,
            result.residual_quantile,
            result.selected};
  }
} // namespace stochastic_reference::tables
namespace selection_reference {
  using namespace stochastic_reference;
  using continuous_reference::EmInteractionRecord;
  using continuous_reference::EmInteractionStatus;
  using continuous_reference::makeProcessFallbackEvent;
  using continuous_reference::transportMassGeV;
  using continuous_reference::step::leptonContinuousFinite;
  namespace gpu {
    namespace em = ::selection_reference;
  }
  namespace table = stochastic_reference::tables;
  inline constexpr std::uint32_t NativeMuonPairCutFlag = 1U;

  CORSIKA_HOST_DEVICE inline bool isNativeMuonPairCut(EmParticleState const& p) {
    return p.reserved == NativeMuonPairCutFlag && isMuonPid(p.pid) && p.generation > 0 &&
           p.step_id == 0 && p.energy_GeV > 0. && p.energy_GeV <= transportMassGeV(p.pid);
  }

  CORSIKA_HOST_DEVICE inline bool isNativeMuonPairCutDeposit(std::int32_t pid,
                                                             std::uint32_t tag,
                                                             double energy,
                                                             double deposit) {
    auto const mass = transportMassGeV(pid);
    auto const residual = deposit - (energy - mass);
    return tag == NativeMuonPairCutFlag && isMuonPid(pid) && energy > 0. &&
           energy <= mass && deposit <= 0. && residual > -1.e-15 && residual < 1.e-15;
  }
  struct InteractionSelectionOutcome {
    gpu::em::EmInteractionRecord interaction{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t fallback_flag{};
    std::uint32_t native_newton_iterations{};
    std::uint32_t native_bisection_iterations{};
    std::uint32_t native_inverse_failures{};
  };
  CORSIKA_HOST_DEVICE inline InteractionSelectionOutcome selectDiscreteInteraction(
      table::NativePhysicsView const& physics, gpu::em::EmParticleState const& particle,
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

    if (isNativeMuonPairCut(particle)) {
      record.status = EmInteractionStatus::ParticleCut;
      record.interaction_grammage_g_per_cm2 = infinity;
      return output; // no rate, decay, or momentum evaluation below rest mass
    }

    if (exceedsParticleCutTime(particle.time_s)) {
      record.status = EmInteractionStatus::ParticleCut;
      record.interaction_grammage_g_per_cm2 = infinity;
      return output;
    }

    auto const charged_lepton = isChargedLeptonPid(particle.pid);
    auto const photon = particle.pid == static_cast<std::int32_t>(EmPid::Photon);
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
          RandomNumberKey const decay_key{random_seed,
                                          shower_id,
                                          particle.history_id,
                                          particle.step_id,
                                          MuonDecayRandomProcessId,
                                          MuonDecayDrawId};
          record.decay_uniform = uniformOpen01(decay_key);
          record.decay_draw_id = MuonDecayDrawId;
          auto const momentum_GeV =
              ::sqrt((particle.energy_GeV - transport_mass.value / 1000.) *
                     (particle.energy_GeV + transport_mass.value / 1000.));
          record.decay_distance_m = -::log(record.decay_uniform) *
                                    MuonDecaySpeedOfLightMPerS * MuonMeanLifetimeS *
                                    momentum_GeV / (transport_mass.value / 1000.);
        }
      }
      auto const minimum_energy =
          table::queryContinuousMinimumEnergy(physics, particle.pid);
      auto const transport_cut =
          table::queryContinuousTransportCut(physics, particle.pid);
      auto const below_cut = transport_mass.status == table::TableLookupStatus::Success &&
                             transport_cut.status == table::TableLookupStatus::Success &&
                             energy_MeV - transport_mass.value < transport_cut.value;
      if (below_cut || (minimum_energy.status == table::TableLookupStatus::Success &&
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
      output.fallback =
          makeTableFallbackEvent(particle, total_query, total, 0, input_index);
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

    RandomNumberKey const distance_key{random_seed,
                                       shower_id,
                                       particle.history_id,
                                       particle.step_id,
                                       InteractionDistanceRandomProcessId,
                                       InteractionDistanceDrawId};
    record.distance_uniform = uniformOpen01(distance_key);
    record.interaction_grammage_g_per_cm2 = -::log(record.distance_uniform) / total.value;
    RandomNumberKey const column_key{random_seed,
                                     shower_id,
                                     particle.history_id,
                                     particle.step_id,
                                     InteractionColumnRandomProcessId,
                                     InteractionColumnDrawId};
    record.process_uniform = uniformOpen01(column_key);

    // The process is selected at post-continuous-loss energy for charged
    // particles, exactly as in the scalar and native-CUDA paths.
    if (charged_lepton) {
      record.status = EmInteractionStatus::DistanceSampled;
      return output;
    }

    constexpr bool proposal_native = true;
    table::RateColumnSelectionResult selection{};

    RandomNumberKey const proposal_key{random_seed,
                                       shower_id,
                                       particle.history_id,
                                       particle.step_id,
                                       ProposalSelectionRandomProcessId,
                                       ProposalSelectionDrawId};
    record.proposal_selection_uniform = uniformOpen01(proposal_key);
    record.proposal_selection_random_process_id = ProposalSelectionRandomProcessId;
    record.proposal_selection_draw_id = ProposalSelectionDrawId;
    selection = table::selectRateColumnByUniform(physics, particle.pid, energy_MeV,
                                                 record.proposal_selection_uniform);

    if (selection.status != table::TableLookupStatus::Success) {
      auto const result = table::TableQueryResult{selection.status, 0, 0.};
      output.fallback =
          makeTableFallbackEvent(particle, total_query, result, 0, input_index);

      output.fallback.selection_uniform = record.proposal_selection_uniform;
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
    if (proposalNativeSelectionRequiresReplay(particle.pid, record.process_id,
                                              record.loss_quantile)) {
      // Selection is not a vertex. The photon must first fly the sampled
      // grammage (or stop at an earlier boundary/cut). Final-state
      // classification completes this request only at the transported vertex.
      record.deferred_photon_fallback_reason =
          static_cast<std::int32_t>(ProposalFallbackReason::NativeSelectionReplay);
      return output;
    }

    table::TableQuery const loss_query{table::TableQueryKind::LossFraction,
                                       particle.pid,
                                       record.process_id,
                                       record.component_hash,
                                       energy_MeV,
                                       record.loss_quantile};
    auto const loss = table::executeTableQuery(physics, loss_query);

    output.native_newton_iterations = table::nativeNewtonIterations(loss);
    output.native_bisection_iterations = table::nativeBisectionIterations(loss);
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
          proposal_native ? ProposalSelectionDrawId : InteractionLossDrawId, input_index);
      output.fallback.loss_quantile = record.loss_quantile;
      output.fallback.selection_uniform =
          proposal_native ? record.proposal_selection_uniform : record.process_uniform;
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
  CORSIKA_HOST_DEVICE inline gpu::em::ProposalFallbackEvent vertexFallback(
      gpu::em::EmInteractionRecord const& candidate,
      gpu::em::ProposalFallbackReason const reason) {
    auto event = gpu::em::makeProcessFallbackEvent(candidate, reason);
    event.input_index = candidate.input_index;
    return event;
  }

  CORSIKA_HOST_DEVICE inline gpu::em::ProposalFallbackEvent vertexTableFallback(
      gpu::em::EmInteractionRecord const& candidate,
      gpu::em::tables::TableQuery const& query,
      gpu::em::tables::TableQueryResult const& result, std::uint64_t const draw_id,
      double const loss_quantile = 0.) {
    auto event = gpu::em::makeTableFallbackEvent(candidate.particle, query, result,
                                                 draw_id, candidate.input_index);
    event.interaction_vertex_reached = candidate.interaction_vertex_reached;
    auto const native_selection = candidate.proposal_selection_random_process_id != 0u;
    event.selection_uniform = native_selection ? candidate.proposal_selection_uniform
                                               : candidate.process_uniform;
    event.loss_quantile = loss_quantile;
    event.outer_acceptance_uniform = candidate.process_uniform;
    event.outer_acceptance_random_process_id = candidate.process_random_process_id;
    event.outer_acceptance_draw_id = candidate.process_draw_id;
    event.random_process_id =
        native_selection ? candidate.proposal_selection_random_process_id : 0u;
    if (native_selection) event.random_draw_id = candidate.proposal_selection_draw_id;
    return event;
  }

  CORSIKA_HOST_DEVICE inline LeptonVertexSelectionOutcome selectLeptonVertex(
      gpu::em::tables::NativePhysicsView const& table,
      gpu::em::EmInteractionRecord const& candidate, std::uint64_t const random_seed,
      std::uint64_t const shower_id) {
    using namespace gpu::em;
    LeptonVertexSelectionOutcome output{};
    auto record = candidate;
    auto const& particle = record.particle;
    if (record.status == EmInteractionStatus::AtRestAnnihilation &&
        particle.pid == static_cast<std::int32_t>(EmPid::Positron)) {
      output.record = record;
      output.interaction_flag = 1;
      return output;
    }
    if (!isChargedLeptonPid(particle.pid)) {
      output.fallback =
          vertexFallback(record, ProposalFallbackReason::UnsupportedParticle);
      output.fallback_flag = 1;
      return output;
    }
    if (record.status != EmInteractionStatus::RequiresReselection ||
        !leptonContinuousFinite(record.total_rate_cm2_per_g) ||
        !(record.total_rate_cm2_per_g > 0.) ||
        !leptonContinuousFinite(record.process_uniform) ||
        !(record.process_uniform > 0.) || !(record.process_uniform < 1.)) {
      output.fallback = vertexFallback(record, ProposalFallbackReason::InvalidFinalState);
      output.fallback_flag = 1;
      return output;
    }

    auto const energy_MeV = particle.energy_GeV * 1000.;
    auto const outer_threshold = record.process_uniform * record.total_rate_cm2_per_g;
    constexpr bool proposal_native = true;
    tables::RateColumnSelectionResult selection{};

    auto const vertex_total = tables::queryTotalRate(table, particle.pid, energy_MeV);
    if (vertex_total.status != tables::TableLookupStatus::Success) {
      tables::TableQuery const failed_total_query{
          tables::TableQueryKind::TotalRate, particle.pid, 0, 0, energy_MeV, 0.};
      output.fallback = vertexTableFallback(record, failed_total_query, vertex_total,
                                            record.process_draw_id);
      output.fallback_flag = 1;
      return output;
    }
    record.vertex_total_rate_cm2_per_g = vertex_total.value;
    if (!(vertex_total.value > 0.) || outer_threshold >= vertex_total.value) {
      if (record.particle.step_id == 0xffffffffffffffffULL) {
        output.fallback =
            vertexFallback(record, ProposalFallbackReason::InvalidFinalState);
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
    RandomNumberKey const proposal_key{random_seed,
                                       shower_id,
                                       particle.history_id,
                                       particle.step_id,
                                       ProposalSelectionRandomProcessId,
                                       ProposalSelectionDrawId};
    record.proposal_selection_uniform = uniformOpen01(proposal_key);
    record.proposal_selection_random_process_id = ProposalSelectionRandomProcessId;
    record.proposal_selection_draw_id = ProposalSelectionDrawId;
    selection = tables::selectRateColumnByUniform(table, particle.pid, energy_MeV,
                                                  record.proposal_selection_uniform);

    if (selection.status != tables::TableLookupStatus::Success) {
      record.process_id = selection.process_id;
      record.component_hash = selection.component_hash;
      tables::TableQuery const failed_column_query{
          tables::TableQueryKind::Rate, particle.pid, selection.process_id,
          selection.component_hash,     energy_MeV,   0.};
      auto const failure = tables::TableQueryResult{selection.status, 0, 0.};
      auto fallback = vertexTableFallback(record, failed_column_query, failure,
                                          record.process_draw_id);
      // Preserve the complete threshold inputs.  A non-finite selection
      // status cannot be diagnosed from the post-transport energy alone.
      fallback.diagnostic_value0 = energy_MeV;
      fallback.diagnostic_value1 = record.total_rate_cm2_per_g;
      fallback.diagnostic_value2 =
          proposal_native ? record.proposal_selection_uniform : record.process_uniform;
      output.fallback = fallback;
      output.fallback_flag = 1;
      return output;
    }
    if (!proposal_native) record.vertex_total_rate_cm2_per_g = selection.total_rate;
    if (!proposal_native &&
        (!(selection.total_rate > 0.) || outer_threshold >= selection.total_rate)) {
      if (record.particle.step_id == 0xffffffffffffffffULL) {
        output.fallback =
            vertexFallback(record, ProposalFallbackReason::InvalidFinalState);
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
    if (selection.selected == 0) {
      output.fallback = vertexFallback(record, ProposalFallbackReason::ZeroTotalRate);
      output.fallback_flag = 1;
      return output;
    }

    record.process_id = selection.process_id;
    record.component_hash = selection.component_hash;

    record.loss_quantile = selection.residual_quantile;
    record.loss_draw_id = ProposalSelectionDrawId;
    if (proposalNativeSelectionRequiresReplay(particle.pid, record.process_id,
                                              record.loss_quantile)) {
      output.fallback =
          vertexFallback(record, ProposalFallbackReason::NativeSelectionReplay);
      output.fallback_flag = 1;
      return output;
    }

    tables::TableQuery const loss_query{tables::TableQueryKind::LossFraction,
                                        particle.pid,
                                        record.process_id,
                                        record.component_hash,
                                        energy_MeV,
                                        record.loss_quantile};
    auto const loss = tables::executeTableQuery(table, loss_query);
    output.native_newton_iterations = tables::nativeNewtonIterations(loss);
    output.native_bisection_iterations = tables::nativeBisectionIterations(loss);
    output.native_inverse_failures =
        table.physics_source == 1u && loss.status != tables::TableLookupStatus::Success
            ? 1u
            : 0u;
    if (loss.status != tables::TableLookupStatus::Success) {
      output.fallback = vertexTableFallback(
          record, loss_query, loss,
          proposal_native ? ProposalSelectionDrawId : InteractionLossDrawId,
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

} // namespace selection_reference
