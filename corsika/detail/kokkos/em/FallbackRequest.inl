/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/TableQuery.hpp>

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline FallbackReason proposal_fallback_reason(
      TableLookupStatus status) {
    switch (status) {
      case TableLookupStatus::ParticleNotFound:
        return FallbackReason::ParticleTableMissing;
      case TableLookupStatus::ColumnNotFound:
        return FallbackReason::ProcessComponentMissing;
      case TableLookupStatus::RateEnergyOutOfRange:
        return FallbackReason::RateEnergyOutOfRange;
      case TableLookupStatus::InverseCdfUnavailable:
        return FallbackReason::InverseCdfUnavailable;
      case TableLookupStatus::LossEnergyOutOfRange:
        return FallbackReason::LossEnergyOutOfRange;
      case TableLookupStatus::LossQuantileOutOfRange:
        return FallbackReason::LossQuantileOutOfRange;
      case TableLookupStatus::ContinuousParticleNotFound:
        return FallbackReason::ParticleTableMissing;
      case TableLookupStatus::ContinuousEnergyOutOfRange:
        return FallbackReason::ContinuousEnergyOutOfRange;
      case TableLookupStatus::ContinuousRangeOutOfRange:
        return FallbackReason::ContinuousRangeOutOfRange;
      case TableLookupStatus::TransportCutReached:
        return FallbackReason::TransportCutReached;
      case TableLookupStatus::Success:
      case TableLookupStatus::NonFiniteInput:
      case TableLookupStatus::InvalidQueryKind:
      case TableLookupStatus::InvalidTableView:
        return FallbackReason::InvalidTableQuery;
    }
    return FallbackReason::InvalidTableQuery;
  }

  CORSIKA_HOST_DEVICE inline FallbackRequest make_process_fallback_event(
      InteractionRecord const& record, FallbackReason reason) {
    FallbackRequest event{};
    event.particle_ = record.particle_;
    event.interactionVertexReached_ = record.interactionVertexReached_;
    event.inputIndex_ = record.inputIndex_;
    event.processId_ = record.processId_;
    event.reason_ = reason;
    event.componentHash_ = record.componentHash_;
    event.energyFraction_ = record.energyFraction_;
    auto const native_selection = record.proposalSelectionRandomProcessId_ != 0u;
    event.selectionUniform_ =
        native_selection ? record.proposalSelectionUniform_ : record.processUniform_;
    event.lossQuantile_ = record.lossQuantile_;
    event.outerAcceptanceUniform_ = record.processUniform_;
    event.outerAcceptanceRandomProcessId_ = record.processRandomProcessId_;
    event.outerAcceptanceDrawId_ = record.processDrawId_;
    event.randomProcessId_ =
        native_selection ? record.proposalSelectionRandomProcessId_ : 0u;
    event.randomDrawId_ =
        native_selection ? record.proposalSelectionDrawId_ : record.lossDrawId_;
    return event;
  }

  CORSIKA_HOST_DEVICE inline bool proposal_fallback_requires_selected_loss(
      FallbackReason reason) {
    return reason == FallbackReason::NativeSelectionReplay ||
           reason == FallbackReason::InverseCdfUnavailable ||
           reason == FallbackReason::LossEnergyOutOfRange ||
           reason == FallbackReason::LossQuantileOutOfRange;
  }

  CORSIKA_HOST_DEVICE inline bool proposal_fallback_requires_interaction_vertex(
      FallbackReason reason) {
    return proposal_fallback_requires_selected_loss(reason) ||
           reason == FallbackReason::CpuOnlyProcess ||
           reason == FallbackReason::EpairRejectionEnvelopeExceeded ||
           reason == FallbackReason::EpairRejectionTrialsExhausted;
  }

  inline char const* proposal_fallback_reason_name(FallbackReason reason) noexcept {
    switch (reason) {
      case FallbackReason::UnsupportedParticle:
        return "unsupported_particle";
      case FallbackReason::UnsupportedMedium:
        return "unsupported_medium";
      case FallbackReason::UnsupportedGeometry:
        return "unsupported_geometry";
      case FallbackReason::ParticleTableMissing:
        return "particle_table_missing";
      case FallbackReason::ProcessComponentMissing:
        return "process_component_missing";
      case FallbackReason::RateEnergyOutOfRange:
        return "rate_energy_out_of_range";
      case FallbackReason::InverseCdfUnavailable:
        return "inverse_cdf_unavailable";
      case FallbackReason::LossEnergyOutOfRange:
        return "loss_energy_out_of_range";
      case FallbackReason::LossQuantileOutOfRange:
        return "loss_quantile_out_of_range";
      case FallbackReason::InvalidTableQuery:
        return "invalid_table_query";
      case FallbackReason::DeviceMemorySpill:
        return "device_memory_spill";
      case FallbackReason::ZeroTotalRate:
        return "zero_total_rate";
      case FallbackReason::CpuOnlyProcess:
        return "cpu_only_process";
      case FallbackReason::GpuProcessNotImplemented:
        return "gpu_process_not_implemented";
      case FallbackReason::InvalidFinalState:
        return "invalid_final_state";
      case FallbackReason::LpmParametersUnavailable:
        return "lpm_parameters_unavailable";
      case FallbackReason::InvalidMassDensity:
        return "invalid_mass_density";
      case FallbackReason::ContinuousEnergyOutOfRange:
        return "continuous_energy_out_of_range";
      case FallbackReason::ContinuousRangeOutOfRange:
        return "continuous_range_out_of_range";
      case FallbackReason::TransportCutReached:
        return "transport_cut_reached";
      case FallbackReason::MoliereParametersUnavailable:
        return "moliere_parameters_unavailable";
      case FallbackReason::MoliereSamplingFailed:
        return "moliere_sampling_failed";
      case FallbackReason::MagneticTransportFailed:
        return "magnetic_transport_failed";
      case FallbackReason::MagneticBoundaryFailed:
        return "magnetic_boundary_failed";
      case FallbackReason::AtmosphereGrammageFailed:
        return "atmosphere_grammage_failed";
      case FallbackReason::AtmosphereInverseGrammageFailed:
        return "atmosphere_inverse_grammage_failed";
      case FallbackReason::AtmosphereVertexLookupFailed:
        return "atmosphere_vertex_lookup_failed";
      case FallbackReason::EpairRejectionEnvelopeExceeded:
        return "epair_rejection_envelope_exceeded";
      case FallbackReason::EpairRejectionTrialsExhausted:
        return "epair_rejection_trials_exhausted";
      case FallbackReason::NativeSelectionReplay:
        return "native_selection_replay";
    }
    return "unknown";
  }

  CORSIKA_HOST_DEVICE inline FallbackRequest make_table_fallback_event(
      ParticleState const& particle, TableQuery const& query,
      TableQueryResult const& result, std::uint64_t random_draw_id,
      std::uint64_t input_index) {
    FallbackRequest event{};
    event.particle_ = particle;
    event.inputIndex_ = input_index;
    event.processId_ = query.processId_;
    event.reason_ = proposal_fallback_reason(result.status_);
    event.componentHash_ = query.componentHash_;
    event.randomDrawId_ = random_draw_id;
    event.diagnosticStatus_ = static_cast<std::int32_t>(result.status_);
    event.diagnosticValue0_ = query.energyMeV_;
    event.diagnosticValue1_ = query.quantile_;
    event.diagnosticValue2_ = result.value_;
    return event;
  }

} // namespace corsika::kokkos
