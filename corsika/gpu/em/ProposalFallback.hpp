/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

#if defined(__CUDACC__)
#define CORSIKA_GPU_FALLBACK_HOST_DEVICE __host__ __device__
#else
#define CORSIKA_GPU_FALLBACK_HOST_DEVICE
#endif

namespace corsika::gpu::em {

  inline char const*
  proposalFallbackReasonName(ProposalFallbackReason reason) noexcept {
    switch (reason) {
    case ProposalFallbackReason::UnsupportedParticle:
      return "unsupported_particle";
    case ProposalFallbackReason::UnsupportedMedium:
      return "unsupported_medium";
    case ProposalFallbackReason::UnsupportedGeometry:
      return "unsupported_geometry";
    case ProposalFallbackReason::ParticleTableMissing:
      return "particle_table_missing";
    case ProposalFallbackReason::ProcessComponentMissing:
      return "process_component_missing";
    case ProposalFallbackReason::RateEnergyOutOfRange:
      return "rate_energy_out_of_range";
    case ProposalFallbackReason::InverseCdfUnavailable:
      return "inverse_cdf_unavailable";
    case ProposalFallbackReason::LossEnergyOutOfRange:
      return "loss_energy_out_of_range";
    case ProposalFallbackReason::LossQuantileOutOfRange:
      return "loss_quantile_out_of_range";
    case ProposalFallbackReason::InvalidTableQuery:
      return "invalid_table_query";
    case ProposalFallbackReason::DeviceMemorySpill:
      return "device_memory_spill";
    case ProposalFallbackReason::ZeroTotalRate:
      return "zero_total_rate";
    case ProposalFallbackReason::CpuOnlyProcess:
      return "cpu_only_process";
    case ProposalFallbackReason::GpuProcessNotImplemented:
      return "gpu_process_not_implemented";
    case ProposalFallbackReason::InvalidFinalState:
      return "invalid_final_state";
    case ProposalFallbackReason::LpmParametersUnavailable:
      return "lpm_parameters_unavailable";
    case ProposalFallbackReason::InvalidMassDensity:
      return "invalid_mass_density";
    case ProposalFallbackReason::ContinuousEnergyOutOfRange:
      return "continuous_energy_out_of_range";
    case ProposalFallbackReason::ContinuousRangeOutOfRange:
      return "continuous_range_out_of_range";
    case ProposalFallbackReason::TransportCutReached:
      return "transport_cut_reached";
    case ProposalFallbackReason::MoliereParametersUnavailable:
      return "moliere_parameters_unavailable";
    case ProposalFallbackReason::MoliereSamplingFailed:
      return "moliere_sampling_failed";
    case ProposalFallbackReason::MagneticTransportFailed:
      return "magnetic_transport_failed";
    case ProposalFallbackReason::MagneticBoundaryFailed:
      return "magnetic_boundary_failed";
    case ProposalFallbackReason::AtmosphereGrammageFailed:
      return "atmosphere_grammage_failed";
    case ProposalFallbackReason::AtmosphereInverseGrammageFailed:
      return "atmosphere_inverse_grammage_failed";
    case ProposalFallbackReason::AtmosphereVertexLookupFailed:
      return "atmosphere_vertex_lookup_failed";
    case ProposalFallbackReason::EpairRejectionEnvelopeExceeded:
      return "epair_rejection_envelope_exceeded";
    }
    return "unknown";
  }

  CORSIKA_GPU_FALLBACK_HOST_DEVICE inline ProposalFallbackReason
  proposalFallbackReason(tables::TableLookupStatus status) {
    using tables::TableLookupStatus;
    switch (status) {
    case TableLookupStatus::ParticleNotFound:
      return ProposalFallbackReason::ParticleTableMissing;
    case TableLookupStatus::ColumnNotFound:
      return ProposalFallbackReason::ProcessComponentMissing;
    case TableLookupStatus::RateEnergyOutOfRange:
      return ProposalFallbackReason::RateEnergyOutOfRange;
    case TableLookupStatus::InverseCdfUnavailable:
      return ProposalFallbackReason::InverseCdfUnavailable;
    case TableLookupStatus::LossEnergyOutOfRange:
      return ProposalFallbackReason::LossEnergyOutOfRange;
    case TableLookupStatus::LossQuantileOutOfRange:
      return ProposalFallbackReason::LossQuantileOutOfRange;
    case TableLookupStatus::ContinuousParticleNotFound:
      return ProposalFallbackReason::ParticleTableMissing;
    case TableLookupStatus::ContinuousEnergyOutOfRange:
      return ProposalFallbackReason::ContinuousEnergyOutOfRange;
    case TableLookupStatus::ContinuousRangeOutOfRange:
      return ProposalFallbackReason::ContinuousRangeOutOfRange;
    case TableLookupStatus::TransportCutReached:
      return ProposalFallbackReason::TransportCutReached;
    case TableLookupStatus::Success:
    case TableLookupStatus::NonFiniteInput:
    case TableLookupStatus::InvalidQueryKind:
    case TableLookupStatus::InvalidTableView:
      return ProposalFallbackReason::InvalidTableQuery;
    }
    return ProposalFallbackReason::InvalidTableQuery;
  }

  /**
   * Preserve the exact selected process/component and random draw identity
   * when a device table lookup must be completed by CPU PROPOSAL.
   */
  CORSIKA_GPU_FALLBACK_HOST_DEVICE inline ProposalFallbackEvent
  makeTableFallbackEvent(EmParticleState const& particle,
                         tables::TableQuery const& query,
                         tables::TableQueryResult const& result,
                         std::uint64_t random_draw_id,
                         std::uint64_t input_index = 0) {
    ProposalFallbackEvent event{};
    event.particle = particle;
    event.input_index = input_index;
    event.process_id = query.process_id;
    event.reason = proposalFallbackReason(result.status);
    event.component_hash = query.component_hash;
    event.random_draw_id = random_draw_id;
    return event;
  }

  CORSIKA_GPU_FALLBACK_HOST_DEVICE inline ProposalFallbackEvent
  makeProcessFallbackEvent(EmInteractionRecord const& record,
                           ProposalFallbackReason reason) {
    ProposalFallbackEvent event{};
    event.particle = record.particle;
    event.input_index = record.input_index;
    event.process_id = record.process_id;
    event.reason = reason;
    event.component_hash = record.component_hash;
    event.energy_fraction = record.energy_fraction;
    event.selection_uniform = record.process_uniform;
    event.loss_quantile = record.loss_quantile;
    event.random_draw_id = record.loss_draw_id;
    return event;
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_FALLBACK_HOST_DEVICE
