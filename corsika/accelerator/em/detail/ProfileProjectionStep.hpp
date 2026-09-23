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
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/detail/ProfileProjectionData.hpp>

namespace corsika::accelerator::em::detail {

  C8_ACCELERATOR_INLINE_FUNCTION inline double projectProfileGrammage(
      gpu::em::detail::DeviceProfileProjection const& projection,
      double const position_m[3]) {
    auto projected_length_m = 0.;
    for (int axis = 0; axis < 3; ++axis)
      projected_length_m +=
          (position_m[axis] - projection.axis_start_position_m[axis]) *
          projection.axis_direction[axis];
    auto const fractional_bin =
        projected_length_m / projection.axis_step_length_m;
    if (fractional_bin < 0.)
      return projection.axis_grammage_g_per_cm2[0];
    auto const lower = static_cast<std::size_t>(fractional_bin);
    auto const upper = lower + 1;
    if (upper >= projection.axis_support_count)
      return projection
          .axis_grammage_g_per_cm2[projection.axis_support_count - 1];
    auto const fraction = fractional_bin - static_cast<double>(lower);
    return projection.axis_grammage_g_per_cm2[upper] * fraction +
           projection.axis_grammage_g_per_cm2[lower] * (1. - fraction);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProjectedEmStepRecord
  projectPhotonStep(
      gpu::em::detail::DeviceProfileProjection const& projection,
      gpu::em::PhotonTransportRecord const& record) {
    using namespace gpu::em;
    ProjectedEmStepRecord projected{};
    projected.history_id = record.start.history_id;
    projected.pid = record.start.pid;
    projected.process_id =
        record.limit == PhotonTransportLimit::Interaction
            ? record.interaction.process_id
            : 0;
    projected.transport_limit = static_cast<std::int32_t>(record.limit);
    projected.start_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.start.position_m);
    projected.end_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.end.position_m);
    projected.end_energy_GeV = record.end.energy_GeV;
    projected.deposited_energy_GeV = record.cut_deposited_energy_GeV;
    projected.cut_deposited_energy_GeV = record.cut_deposited_energy_GeV;
    projected.observation_surface_reached_before_cut =
        record.observation_surface_reached_before_cut;
    projected.weight = record.start.weight;
    return projected;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProjectedEmStepRecord
  projectLeptonStep(
      gpu::em::detail::DeviceProfileProjection const& projection,
      gpu::em::LeptonTransportRecord const& record) {
    using namespace gpu::em;
    ProjectedEmStepRecord projected{};
    projected.history_id = record.start.history_id;
    projected.pid = record.start.pid;
    projected.process_id =
        (record.limit == LeptonTransportLimit::InteractionCandidate ||
         record.interaction.status == EmInteractionStatus::AtRestAnnihilation)
            ? record.interaction.process_id
            : 0;
    projected.transport_limit = static_cast<std::int32_t>(record.limit);
    projected.reserved = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(record.multiple_scattering_applied) |
        (static_cast<std::uint32_t>(record.multiple_scattering_status) << 1) |
        (record.multiple_scattering_iterations << 8));
    projected.start_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.start.position_m);
    projected.end_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.end.position_m);
    projected.end_energy_GeV = record.end.energy_GeV;
    projected.deposited_energy_GeV =
        record.continuous_deposited_energy_GeV +
        record.cut_deposited_energy_GeV;
    projected.cut_deposited_energy_GeV = record.cut_deposited_energy_GeV;
    projected.observation_surface_reached_before_cut =
        record.observation_surface_reached_before_cut;
    projected.weight = record.start.weight;
    return projected;
  }

} // namespace corsika::accelerator::em::detail
