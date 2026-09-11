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
#include <corsika/accelerator/em/common/ObservationPlane.hpp>
#include <corsika/accelerator/em/common/ExternalTransportBoundary.hpp>
#include <corsika/accelerator/em/common/ProposalFallback.hpp>
#include <corsika/accelerator/em/common/SphericalAtmosphere.hpp>

namespace corsika::accelerator::em::detail {

  struct PhotonTransportOutcome {
    gpu::em::PhotonTransportRecord record{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t fallback_flag{};
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  makePhotonTransportFallback(
      gpu::em::EmInteractionRecord const& interaction,
      gpu::em::ProposalFallbackReason const reason) {
    gpu::em::ProposalFallbackEvent event{};
    event.particle = interaction.particle;
    event.input_index = interaction.input_index;
    event.process_id = interaction.process_id;
    event.reason = reason;
    event.component_hash = interaction.component_hash;
    event.energy_fraction = interaction.energy_fraction;
    event.loss_quantile = interaction.loss_quantile;
    event.random_draw_id = interaction.loss_draw_id;
    return event;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool transportRadiusClose(
      double const left, double const right) {
    auto const scale = left > right ? left : right;
    return ::fabs(left - right) <=
           128. * 2.22044604925031308085e-16 *
               (scale > 1. ? scale : 1.);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool transportFinite(
      double const value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  /** Straight photon step shared verbatim by native CUDA and Kokkos. */
  C8_ACCELERATOR_INLINE_FUNCTION inline PhotonTransportOutcome
  transportPhoton(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::EmInteractionRecord const& interaction,
      gpu::em::ExternalTransportBoundary const external = {}) {
    using namespace gpu::em;
    PhotonTransportOutcome output{};
    auto const start = interaction.particle;
    if (start.pid != static_cast<std::int32_t>(EmPid::Photon)) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::UnsupportedParticle);
      output.fallback_flag = 1;
      return output;
    }
    if (interaction.status != EmInteractionStatus::Selected &&
        interaction.status != EmInteractionStatus::NoDiscreteInteraction &&
        interaction.status != EmInteractionStatus::ParticleCut) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::InvalidFinalState);
      output.fallback_flag = 1;
      return output;
    }

    auto const layer = queryAtmosphereLayer(
        environment, start.position_m, start.direction);
    if (layer.status != AtmosphereStatus::Success) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::UnsupportedGeometry);
      output.fallback_flag = 1;
      return output;
    }
    auto& record = output.record;
    if (interaction.status == EmInteractionStatus::ParticleCut) {
      record.interaction = interaction;
      record.start = start;
      record.end = start;
      record.end.step_id++;
      record.input_index = interaction.input_index;
      record.limit = PhotonTransportLimit::ParticleCut;
      record.start_layer_index = layer.layer_index;
      record.end_layer_index = layer.layer_index;
      record.start_density_g_per_cm3 = layer.density_g_per_cm3;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
      record.cut_deposited_energy_GeV = start.energy_GeV;
      return output;
    }

    auto const atmosphere_boundary = distanceToAtmosphereBoundary(
        environment, start.position_m, start.direction);
    auto const observation = intersectObservationPlaneStraight(
        environment, start.position_m, start.direction);
    auto const has_atmosphere_boundary =
        atmosphere_boundary.status == AtmosphereStatus::Success;
    auto const has_observation =
        !external.disable_observation &&
        observation.status == ObservationPlaneStatus::Success;
    if (!has_atmosphere_boundary && !has_observation) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::UnsupportedGeometry);
      output.fallback_flag = 1;
      return output;
    }
    auto const observation_wins =
        has_observation &&
        (!has_atmosphere_boundary ||
         observation.distance_m < atmosphere_boundary.distance_m);
    auto boundary_distance_m =
        observation_wins ? observation.distance_m
                         : atmosphere_boundary.distance_m;
    auto const material_wins = external.enabled &&
        external.distance_m > 0. && external.distance_m < boundary_distance_m;
    if (material_wins) boundary_distance_m = external.distance_m;
    auto const limiting_radius_m =
        observation_wins ? 0. : atmosphere_boundary.radius_m;
    auto const boundary_grammage = geometryCompetitionGrammage(
        environment, layer.layer_index, start.position_m, start.direction,
        boundary_distance_m, external);
    if (boundary_grammage.status != AtmosphereStatus::Success) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
      output.fallback.diagnostic_status =
          static_cast<std::int32_t>(boundary_grammage.status);
      output.fallback.diagnostic_value0 = boundary_distance_m;
      output.fallback.diagnostic_value1 = layer.density_g_per_cm3;
      output.fallback.diagnostic_value2 = limiting_radius_m;
      output.fallback_flag = 1;
      return output;
    }

    record.interaction = interaction;
    record.start = start;
    record.input_index = interaction.input_index;
    record.start_layer_index = layer.layer_index;
    record.start_density_g_per_cm3 = layer.density_g_per_cm3;
    record.limiting_radius_m = limiting_radius_m;
    auto const reaches_interaction =
        interaction.status == EmInteractionStatus::Selected &&
        transportFinite(interaction.interaction_grammage_g_per_cm2) &&
        interaction.interaction_grammage_g_per_cm2 >= 0. &&
        interaction.interaction_grammage_g_per_cm2 <= boundary_grammage.value;
    if (reaches_interaction) {
      auto const distance = atmosphereDistanceFromGrammage(
          environment, layer.layer_index, start.position_m, start.direction,
          interaction.interaction_grammage_g_per_cm2);
      if (distance.status != AtmosphereStatus::Success ||
          distance.value > boundary_distance_m * (1. + 1.e-12)) {
        output.fallback = makePhotonTransportFallback(
            interaction,
            ProposalFallbackReason::AtmosphereInverseGrammageFailed);
        output.fallback.diagnostic_status =
            static_cast<std::int32_t>(distance.status);
        output.fallback.diagnostic_value0 = distance.value;
        output.fallback.diagnostic_value1 = boundary_distance_m;
        output.fallback.diagnostic_value2 =
            interaction.interaction_grammage_g_per_cm2;
        output.fallback_flag = 1;
        return output;
      }
      advancePhotonState(start, distance.value, record.end);
      auto const vertex = queryAtmosphereLayer(
          environment, record.end.position_m, record.end.direction);
      if (vertex.status != AtmosphereStatus::Success ||
          !transportFinite(vertex.density_g_per_cm3) ||
          !(vertex.density_g_per_cm3 > 0.)) {
        output.fallback = makePhotonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereVertexLookupFailed);
        output.fallback.diagnostic_status =
            static_cast<std::int32_t>(vertex.status);
        output.fallback.diagnostic_value0 = vertex.radius_m;
        output.fallback.diagnostic_value1 = vertex.density_g_per_cm3;
        output.fallback.diagnostic_value2 = distance.value;
        output.fallback_flag = 1;
        return output;
      }
      record.limit = PhotonTransportLimit::Interaction;
      record.end_layer_index = vertex.layer_index;
      record.distance_m = distance.value;
      record.traversed_grammage_g_per_cm2 =
          interaction.interaction_grammage_g_per_cm2;
      record.end_density_g_per_cm3 = vertex.density_g_per_cm3;
      record.interaction.particle = record.end;
      record.interaction.mass_density_g_per_cm3 =
          vertex.density_g_per_cm3;
      if (exceedsParticleCutTime(record.end.time_s)) {
        record.end.step_id++;
        record.limit = PhotonTransportLimit::ParticleCut;
        record.cut_deposited_energy_GeV = start.energy_GeV;
      }
      return output;
    }

    advancePhotonState(start, boundary_distance_m, record.end);
    record.end.step_id++;
    record.distance_m = boundary_distance_m;
    record.traversed_grammage_g_per_cm2 = boundary_grammage.value;
    if (material_wins) {
      record.limit = PhotonTransportLimit::MaterialBoundary;
      record.end_layer_index = layer.layer_index;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
    } else if (observation_wins) {
      double radial[3]{};
      record.limiting_radius_m = atmosphere_detail::radiusVector(
          environment, record.end.position_m, radial);
      record.limit = PhotonTransportLimit::ObservationSurface;
      record.end_layer_index = -1;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
    } else {
      auto const outermost =
          environment.atmosphere_layers[environment.number_of_layers - 1]
              .outer_radius_m;
      if (environment.geometry == EnvironmentGeometry::HomogeneousConvexPolyhedron ||
          transportRadiusClose(atmosphere_boundary.radius_m, outermost)) {
        record.limit = PhotonTransportLimit::EscapedEnvironment;
        record.end_layer_index = -1;
        record.end_density_g_per_cm3 = 0.;
      } else {
        auto const next = queryAtmosphereLayer(
            environment, record.end.position_m, record.end.direction);
        if (next.status != AtmosphereStatus::Success ||
            next.layer_index == layer.layer_index) {
          output.fallback = makePhotonTransportFallback(
              interaction, ProposalFallbackReason::UnsupportedGeometry);
          output.fallback_flag = 1;
          return output;
        }
        record.limit = PhotonTransportLimit::LayerBoundary;
        record.end_layer_index = next.layer_index;
        record.end.medium_id =
            environment.atmosphere_layers[next.layer_index].medium_id;
        record.end_density_g_per_cm3 = next.density_g_per_cm3;
      }
    }
    if (exceedsParticleCutTime(record.end.time_s)) {
      record.observation_surface_reached_before_cut =
          record.limit == PhotonTransportLimit::ObservationSurface ? 1U : 0U;
      record.limit = PhotonTransportLimit::ParticleCut;
      record.cut_deposited_energy_GeV = start.energy_GeV;
    }
    return output;
  }

} // namespace corsika::accelerator::em::detail
