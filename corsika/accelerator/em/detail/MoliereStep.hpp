/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/accelerator/em/detail/LeptonContinuousStep.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/Philox.hpp>

namespace corsika::accelerator::em::detail {

  /** Apply the PROPOSAL-compatible Moliere draw to one completed step. */
  template <std::size_t ComponentCapacity>
  C8_ACCELERATOR_INLINE_FUNCTION inline bool applyMoliereScattering(
      gpu::em::MoliereSnapshot const& snapshot,
      gpu::em::MoliereInterpolationView const& interpolation,
      bool const enabled, std::uint64_t const random_seed,
      std::uint64_t const shower_id,
      gpu::em::LeptonTransportRecord& record,
      gpu::em::ProposalFallbackEvent& fallback) {
    using namespace gpu::em;
    if (!enabled) return true;
    auto const& start = record.start;
    RandomNumberKey const first_key{
        random_seed, shower_id, start.history_id, start.step_id,
        ContinuousScatteringRandomProcessId, MoliereFirstAngleDrawId};
    RandomNumberKey const second_key{
        random_seed, shower_id, start.history_id, start.step_id,
        ContinuousScatteringRandomProcessId, MoliereSecondAngleDrawId};
    RandomNumberKey const azimuth_key{
        random_seed, shower_id, start.history_id, start.step_id,
        ContinuousScatteringRandomProcessId, MoliereAzimuthDrawId};
    record.multiple_scattering_first_uniform = uniformOpen01(first_key);
    record.multiple_scattering_second_uniform = uniformOpen01(second_key);
    record.multiple_scattering_azimuth_uniform = uniformOpen01(azimuth_key);
    auto const scattering =
        sampleMoliereScatteringAngle2DForCapacity<ComponentCapacity>(
            snapshot, interpolation, record.traversed_grammage_g_per_cm2,
            record.start.energy_GeV * 1000., record.end.energy_GeV * 1000.,
            record.multiple_scattering_first_uniform,
            record.multiple_scattering_second_uniform, false);
    record.multiple_scattering_status =
        static_cast<std::uint16_t>(scattering.status);
    record.multiple_scattering_iterations = scattering.iterations;
    record.multiple_scattering_angle_rad = scattering.angle_rad;
    if (scattering.status == MoliereStatus::NoDeflection) return true;
    if (scattering.status != MoliereStatus::Success) {
      fallback = makeLeptonTransportFallback(
          record.interaction, ProposalFallbackReason::MoliereSamplingFailed);
      fallback.final_state_uniform =
          record.multiple_scattering_first_uniform;
      fallback.final_state_draw_id = MoliereFirstAngleDrawId;
      return false;
    }
    auto const direction = applyMoliereDirectionCpuStepCompatible(
        record.start.direction, record.end.direction, scattering.angle_rad,
        record.multiple_scattering_azimuth_uniform);
    if (direction.status != MoliereStatus::Success) {
      fallback = makeLeptonTransportFallback(
          record.interaction, ProposalFallbackReason::MoliereSamplingFailed);
      fallback.final_state_uniform =
          record.multiple_scattering_azimuth_uniform;
      fallback.final_state_draw_id = MoliereAzimuthDrawId;
      return false;
    }
    for (int axis = 0; axis < 3; ++axis)
      record.end.direction[axis] = direction.direction[axis];
    record.multiple_scattering_applied = 1;
    return true;
  }

  template <std::size_t ComponentCapacity>
  C8_ACCELERATOR_INLINE_FUNCTION inline std::uint32_t
  applyMoliereScatteringStage(
      gpu::em::MoliereSnapshot const& electron_snapshot,
      gpu::em::MoliereSnapshot const& muon_snapshot,
      gpu::em::MoliereInterpolationView const& interpolation,
      bool const muon_snapshot_available, std::uint64_t const random_seed,
      std::uint64_t const shower_id, gpu::em::LeptonTransportRecord& record,
      gpu::em::ProposalFallbackEvent& fallback,
      std::uint32_t const transport_state) {
    using namespace gpu::em;
    if (transport_state == 2u) return 0u;
    if (transport_state != 0u) return transport_state;
    auto const muon = isMuonPid(record.start.pid);
    if (muon && !muon_snapshot_available) {
      fallback = makeLeptonTransportFallback(
          record.interaction,
          ProposalFallbackReason::MoliereParametersUnavailable);
      return 1u;
    }
    auto const& particle_snapshot =
        muon ? muon_snapshot : electron_snapshot;
    if (!applyMoliereScattering<ComponentCapacity>(
            particle_snapshot, interpolation, true, random_seed, shower_id,
            record, fallback)) {
      return 1u;
    }
    if (record.limit == LeptonTransportLimit::InteractionCandidate ||
        record.limit == LeptonTransportLimit::DecayCandidate) {
      record.interaction.particle = record.end;
    }
    return 0u;
  }

} // namespace corsika::accelerator::em::detail
