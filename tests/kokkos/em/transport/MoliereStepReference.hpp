/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Independent production 41cdc543 bodies; only namespace/includes/annotations adapted.
#include "TransportReference.hpp"
#include "../stochastic/StochasticReference.hpp"
#include "../moliere/MoliereReference.hpp"
namespace moliere_stage_reference {
  using continuous_reference::ProposalFallbackEvent;
  using continuous_reference::ProposalFallbackReason;
  using continuous_reference::step::makeLeptonTransportFallback;
  using stochastic_reference::ContinuousScatteringRandomProcessId;
  using stochastic_reference::isMuonPid;
  using stochastic_reference::MoliereAzimuthDrawId;
  using stochastic_reference::MoliereFirstAngleDrawId;
  using stochastic_reference::MoliereSecondAngleDrawId;
  using stochastic_reference::RandomNumberKey;
  using stochastic_reference::uniformOpen01;
  using transport_reference::LeptonTransportLimit;
  using transport_reference::LeptonTransportRecord;
  using namespace corsika::kokkos::test::moliere_reference;
  /** Apply the PROPOSAL-compatible Moliere draw to one completed step. */
  template <std::size_t ComponentCapacity>
  KOKKOS_INLINE_FUNCTION bool applyMoliereScattering(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
      bool const enabled, std::uint64_t const random_seed, std::uint64_t const shower_id,
      LeptonTransportRecord& record, ProposalFallbackEvent& fallback) {
    if (!enabled) return true;
    auto const& start = record.start;
    RandomNumberKey const first_key{random_seed,
                                    shower_id,
                                    start.history_id,
                                    start.step_id,
                                    ContinuousScatteringRandomProcessId,
                                    MoliereFirstAngleDrawId};
    RandomNumberKey const second_key{random_seed,
                                     shower_id,
                                     start.history_id,
                                     start.step_id,
                                     ContinuousScatteringRandomProcessId,
                                     MoliereSecondAngleDrawId};
    RandomNumberKey const azimuth_key{random_seed,
                                      shower_id,
                                      start.history_id,
                                      start.step_id,
                                      ContinuousScatteringRandomProcessId,
                                      MoliereAzimuthDrawId};
    record.multiple_scattering_first_uniform = uniformOpen01(first_key);
    record.multiple_scattering_second_uniform = uniformOpen01(second_key);
    record.multiple_scattering_azimuth_uniform = uniformOpen01(azimuth_key);
    auto const scattering = sampleMoliereScatteringAngle2DForCapacity<ComponentCapacity>(
        snapshot, interpolation, record.traversed_grammage_g_per_cm2,
        record.start.energy_GeV * 1000., record.end.energy_GeV * 1000.,
        record.multiple_scattering_first_uniform,
        record.multiple_scattering_second_uniform, false);
    record.multiple_scattering_status = static_cast<std::uint16_t>(scattering.status);
    record.multiple_scattering_iterations = scattering.iterations;
    record.multiple_scattering_angle_rad = scattering.angle_rad;
    if (scattering.status == MoliereStatus::NoDeflection) return true;
    if (scattering.status != MoliereStatus::Success) {
      fallback = makeLeptonTransportFallback(
          record.interaction, ProposalFallbackReason::MoliereSamplingFailed);
      fallback.final_state_uniform = record.multiple_scattering_first_uniform;
      fallback.final_state_draw_id = MoliereFirstAngleDrawId;
      return false;
    }
    auto const direction = applyMoliereDirectionCpuStepCompatible(
        record.start.direction, record.end.direction, scattering.angle_rad,
        record.multiple_scattering_azimuth_uniform);
    if (direction.status != MoliereStatus::Success) {
      fallback = makeLeptonTransportFallback(
          record.interaction, ProposalFallbackReason::MoliereSamplingFailed);
      fallback.final_state_uniform = record.multiple_scattering_azimuth_uniform;
      fallback.final_state_draw_id = MoliereAzimuthDrawId;
      return false;
    }
    for (int axis = 0; axis < 3; ++axis)
      record.end.direction[axis] = direction.direction[axis];
    record.multiple_scattering_applied = 1;
    return true;
  }

  template <std::size_t ComponentCapacity>
  KOKKOS_INLINE_FUNCTION std::uint32_t applyMoliereScatteringStage(
      MoliereSnapshot const& electron_snapshot, MoliereSnapshot const& muon_snapshot,
      MoliereInterpolationView const& interpolation, bool const muon_snapshot_available,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      LeptonTransportRecord& record, ProposalFallbackEvent& fallback,
      std::uint32_t const transport_state) {
    if (transport_state == 2u) return 0u;
    if (transport_state != 0u) return transport_state;
    auto const muon = isMuonPid(record.start.pid);
    if (muon && !muon_snapshot_available) {
      fallback = makeLeptonTransportFallback(
          record.interaction, ProposalFallbackReason::MoliereParametersUnavailable);
      return 1u;
    }
    auto const& particle_snapshot = muon ? muon_snapshot : electron_snapshot;
    if (!applyMoliereScattering<ComponentCapacity>(particle_snapshot, interpolation, true,
                                                   random_seed, shower_id, record,
                                                   fallback)) {
      return 1u;
    }
    if (record.limit == LeptonTransportLimit::InteractionCandidate ||
        record.limit == LeptonTransportLimit::DecayCandidate) {
      record.interaction.particle = record.end;
    }
    return 0u;
  }

} // namespace moliere_stage_reference
