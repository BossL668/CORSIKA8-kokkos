/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include "ContinuousReference.hpp"
#include <corsika/kokkos/em/algorithms/LeptonContinuousStep.hpp>
namespace continuous_test {
  namespace current = corsika::kokkos;
  namespace reference = continuous_reference;
  struct Outcome {
    double values_[96]{};
    std::uint64_t states_[128]{};
    std::uint32_t valueCount_{}, stateCount_{};
  };
  CORSIKA_HOST_DEVICE inline void append(Outcome& result, double value) {
    result.values_[result.valueCount_++] = value;
  }
  template <typename T>
  CORSIKA_HOST_DEVICE inline void append(Outcome& result, T value) {
    result.states_[result.stateCount_++] = static_cast<std::uint64_t>(value);
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         current::ParticleState const& value) {
    append(result, value.pid_);
    append(result, value.mediumId_);
    append(result, value.generation_);
    append(result, value.reserved_);
    append(result, value.energyGeV_);
    for (int axis = 0; axis < 3; ++axis) append(result, value.positionM_[axis]);
    for (int axis = 0; axis < 3; ++axis) append(result, value.direction_[axis]);
    append(result, value.timeS_);
    append(result, value.weight_);
    append(result, value.historyId_);
    append(result, value.parentHistoryId_);
    append(result, value.stepId_);
    append(result, value.environmentNode_);
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         reference::EmParticleState const& value) {
    append(result, value.pid);
    append(result, value.medium_id);
    append(result, value.generation);
    append(result, value.reserved);
    append(result, value.energy_GeV);
    for (int axis = 0; axis < 3; ++axis) append(result, value.position_m[axis]);
    for (int axis = 0; axis < 3; ++axis) append(result, value.direction[axis]);
    append(result, value.time_s);
    append(result, value.weight);
    append(result, value.history_id);
    append(result, value.parent_history_id);
    append(result, value.step_id);
    append(result, value.environment_node);
  }
  CORSIKA_HOST_DEVICE inline reference::EmParticleState convert(
      current::ParticleState const& value) {
    reference::EmParticleState result;
    result.pid = value.pid_;
    result.medium_id = value.mediumId_;
    result.generation = value.generation_;
    result.reserved = value.reserved_;
    result.energy_GeV = value.energyGeV_;
    for (int axis = 0; axis < 3; ++axis) result.position_m[axis] = value.positionM_[axis];
    for (int axis = 0; axis < 3; ++axis) result.direction[axis] = value.direction_[axis];
    result.time_s = value.timeS_;
    result.weight = value.weight_;
    result.history_id = value.historyId_;
    result.parent_history_id = value.parentHistoryId_;
    result.step_id = value.stepId_;
    result.environment_node = value.environmentNode_;
    return result;
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         current::InteractionRecord const& value) {
    append(result, value.particle_);
    append(result, value.deferredPhotonFallbackReason_);
    append(result, value.deferredPhotonTableStatus_);
    append(result, value.interactionVertexReached_);
    append(result, value.inputIndex_);
    append(result, value.processId_);
    append(result, value.status_);
    append(result, value.componentHash_);
    append(result, value.totalRateCm2PerG_);
    append(result, value.vertexTotalRateCm2PerG_);
    append(result, value.interactionGrammageGPerCm2_);
    append(result, value.massDensityGPerCm3_);
    append(result, value.particleMassGeV_);
    append(result, value.decayDistanceM_);
    append(result, value.decayUniform_);
    append(result, value.decayDrawId_);
    append(result, value.energyFraction_);
    append(result, value.distanceUniform_);
    append(result, value.processUniform_);
    append(result, value.lossQuantile_);
    append(result, value.distanceDrawId_);
    append(result, value.processDrawId_);
    append(result, value.lossDrawId_);
    append(result, value.proposalSelectionUniform_);
    append(result, value.processRandomProcessId_);
    append(result, value.proposalSelectionRandomProcessId_);
    append(result, value.proposalSelectionDrawId_);
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         reference::EmInteractionRecord const& value) {
    append(result, value.particle);
    append(result, value.deferred_photon_fallback_reason);
    append(result, value.deferred_photon_table_status);
    append(result, value.interaction_vertex_reached);
    append(result, value.input_index);
    append(result, value.process_id);
    append(result, value.status);
    append(result, value.component_hash);
    append(result, value.total_rate_cm2_per_g);
    append(result, value.vertex_total_rate_cm2_per_g);
    append(result, value.interaction_grammage_g_per_cm2);
    append(result, value.mass_density_g_per_cm3);
    append(result, value.particle_mass_GeV);
    append(result, value.decay_distance_m);
    append(result, value.decay_uniform);
    append(result, value.decay_draw_id);
    append(result, value.energy_fraction);
    append(result, value.distance_uniform);
    append(result, value.process_uniform);
    append(result, value.loss_quantile);
    append(result, value.distance_draw_id);
    append(result, value.process_draw_id);
    append(result, value.loss_draw_id);
    append(result, value.proposal_selection_uniform);
    append(result, value.process_random_process_id);
    append(result, value.proposal_selection_random_process_id);
    append(result, value.proposal_selection_draw_id);
  }
  CORSIKA_HOST_DEVICE inline reference::EmInteractionRecord convert(
      current::InteractionRecord const& value) {
    reference::EmInteractionRecord result;
    result.particle = convert(value.particle_);
    result.deferred_photon_fallback_reason = value.deferredPhotonFallbackReason_;
    result.deferred_photon_table_status = value.deferredPhotonTableStatus_;
    result.interaction_vertex_reached = value.interactionVertexReached_;
    result.input_index = value.inputIndex_;
    result.process_id = value.processId_;
    result.status = static_cast<reference::EmInteractionStatus>(value.status_);
    result.component_hash = value.componentHash_;
    result.total_rate_cm2_per_g = value.totalRateCm2PerG_;
    result.vertex_total_rate_cm2_per_g = value.vertexTotalRateCm2PerG_;
    result.interaction_grammage_g_per_cm2 = value.interactionGrammageGPerCm2_;
    result.mass_density_g_per_cm3 = value.massDensityGPerCm3_;
    result.particle_mass_GeV = value.particleMassGeV_;
    result.decay_distance_m = value.decayDistanceM_;
    result.decay_uniform = value.decayUniform_;
    result.decay_draw_id = value.decayDrawId_;
    result.energy_fraction = value.energyFraction_;
    result.distance_uniform = value.distanceUniform_;
    result.process_uniform = value.processUniform_;
    result.loss_quantile = value.lossQuantile_;
    result.distance_draw_id = value.distanceDrawId_;
    result.process_draw_id = value.processDrawId_;
    result.loss_draw_id = value.lossDrawId_;
    result.proposal_selection_uniform = value.proposalSelectionUniform_;
    result.process_random_process_id = value.processRandomProcessId_;
    result.proposal_selection_random_process_id = value.proposalSelectionRandomProcessId_;
    result.proposal_selection_draw_id = value.proposalSelectionDrawId_;
    return result;
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         current::FallbackRequest const& value) {
    append(result, value.particle_);
    append(result, value.interactionVertexReached_);
    append(result, value.inputIndex_);
    append(result, value.processId_);
    append(result, value.reason_);
    append(result, value.diagnosticStatus_);
    append(result, value.diagnosticReserved_);
    append(result, value.diagnosticValue0_);
    append(result, value.diagnosticValue1_);
    append(result, value.diagnosticValue2_);
    append(result, value.componentHash_);
    append(result, value.mediumHash_);
    append(result, value.interactionHash_);
    append(result, value.energyFraction_);
    append(result, value.selectionUniform_);
    append(result, value.lossQuantile_);
    append(result, value.finalStateUniform_);
    append(result, value.outerAcceptanceUniform_);
    append(result, value.randomProcessId_);
    append(result, value.outerAcceptanceRandomProcessId_);
    append(result, value.randomDrawId_);
    append(result, value.outerAcceptanceDrawId_);
    append(result, value.finalStateDrawId_);
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         reference::ProposalFallbackEvent const& value) {
    append(result, value.particle);
    append(result, value.interaction_vertex_reached);
    append(result, value.input_index);
    append(result, value.process_id);
    append(result, value.reason);
    append(result, value.diagnostic_status);
    append(result, value.diagnostic_reserved);
    append(result, value.diagnostic_value0);
    append(result, value.diagnostic_value1);
    append(result, value.diagnostic_value2);
    append(result, value.component_hash);
    append(result, value.medium_hash);
    append(result, value.interaction_hash);
    append(result, value.energy_fraction);
    append(result, value.selection_uniform);
    append(result, value.loss_quantile);
    append(result, value.final_state_uniform);
    append(result, value.outer_acceptance_uniform);
    append(result, value.random_process_id);
    append(result, value.outer_acceptance_random_process_id);
    append(result, value.random_draw_id);
    append(result, value.outer_acceptance_draw_id);
    append(result, value.final_state_draw_id);
  }
  CORSIKA_HOST_DEVICE inline void append(
      Outcome& result, current::LeptonContinuousPreparation const& value) {
    append(result, value.massMeV_);
    append(result, value.minimumEnergyMeV_);
    append(result, value.transportCutMeV_);
    append(result, value.maximumGrammageGPerCm2_);
    append(result, value.particleCut_);
    append(result, value.fallbackFlag_);
    append(result, value.fallback_);
  }
  CORSIKA_HOST_DEVICE inline void append(
      Outcome& result, reference::step::LeptonContinuousPreparation const& value) {
    append(result, value.mass_MeV);
    append(result, value.minimum_energy_MeV);
    append(result, value.transport_cut_MeV);
    append(result, value.maximum_grammage_g_per_cm2);
    append(result, value.particle_cut);
    append(result, value.fallback_flag);
    append(result, value.fallback);
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         current::LeptonContinuousLoss const& value) {
    append(result, value.finalEnergyMeV_);
    append(result, value.depositedEnergyGeV_);
    append(result, value.transportCutReached_);
    append(result, value.fallbackFlag_);
    append(result, value.fallback_);
  }
  CORSIKA_HOST_DEVICE inline void append(
      Outcome& result, reference::step::LeptonContinuousLoss const& value) {
    append(result, value.final_energy_MeV);
    append(result, value.deposited_energy_GeV);
    append(result, value.transport_cut_reached);
    append(result, value.fallback_flag);
    append(result, value.fallback);
  }
  CORSIKA_HOST_DEVICE inline void append(Outcome& result,
                                         current::TableQueryResult const& value) {
    append(result, value.status_);
    append(result, value.reserved_);
    append(result, value.value_);
  }
  CORSIKA_HOST_DEVICE inline void append(
      Outcome& result, reference::tables::TableQueryResult const& value) {
    append(result, value.status);
    append(result, value.reserved);
    append(result, value.value);
  }
} // namespace continuous_test
