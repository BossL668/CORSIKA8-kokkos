/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Independent production 41cdc543 NodeTree step. Old record names and
// control flow are retained. Only geometry adapters use the previously frozen
// query oracles; none call the migrated PhotonTransportStep or EnvironmentQuery.
#include "TransportReference.hpp"
#include "../stochastic/StochasticReference.hpp"
#include "../../environment/EnvironmentQueryReference.hpp"

namespace photon_transport_reference {
  using namespace transport_reference;
  using stochastic_reference::EmPid;
  using stochastic_reference::exceedsParticleCutTime;
  using stochastic_reference::ProposalFallbackEvent;
  using stochastic_reference::ProposalFallbackReason;
  namespace query_reference = environment_query_reference;
  using AtmosphereStatus = query_reference::Status;
  using ObservationPlaneStatus = query_reference::ObservationStatus;
  enum class AtmosphereBoundaryPolicy { StraightPathGuard };
  struct EnvironmentSnapshot {
    query_reference::FullView node_tree;
    double earth_center_m[3]{};
    double point[3]{};
    double normal[3]{};
  };
  struct ExternalTransportBoundary {
    double distance_m{HUGE_VAL};
    bool enabled{};
    bool disable_observation{};
  };
  struct LayerQuery {
    AtmosphereStatus status;
    int layer_index;
    double radius_m, density_g_per_cm3;
  };
  struct BoundaryQuery {
    AtmosphereStatus status;
    double distance_m, radius_m;
    int next_node;
  };
  struct GrammageQuery {
    AtmosphereStatus status;
    double value;
  };
  struct ObservationQuery {
    ObservationPlaneStatus status;
    double distance_m;
  };
  KOKKOS_INLINE_FUNCTION LayerQuery queryAtmosphereLayer(EnvironmentSnapshot const& e,
                                                         double const* p, double const* d,
                                                         AtmosphereBoundaryPolicy,
                                                         int node) {
    auto const q = query_reference::query(e.node_tree, p, d, node);
    return {q.status_, q.node_, q.radiusM_, q.densityGPerCm3_};
  }
  KOKKOS_INLINE_FUNCTION BoundaryQuery
  distanceToAtmosphereBoundary(EnvironmentSnapshot const& e, double const* p,
                               double const* d, AtmosphereBoundaryPolicy, int node) {
    auto const q = query_reference::boundary(e.node_tree, p, d, node);
    return {q.status_, q.distanceM_, q.radiusM_, q.nextNode_};
  }
  KOKKOS_INLINE_FUNCTION ObservationQuery intersectObservationPlaneStraight(
      EnvironmentSnapshot const& e, double const* p, double const* d) {
    auto const q = query_reference::observation(e.point, e.normal, p, d, HUGE_VAL);
    return {q.status_, q.distanceM_};
  }
  KOKKOS_INLINE_FUNCTION GrammageQuery geometryCompetitionGrammage(
      EnvironmentSnapshot const& e, int node, double const* p, double const* d,
      double length, ExternalTransportBoundary const&) {
    auto const q = query_reference::competition(e.node_tree, node, p, d, length);
    return {q.status_, q.value_};
  }
  KOKKOS_INLINE_FUNCTION GrammageQuery
  atmosphereDistanceFromGrammage(EnvironmentSnapshot const& e, int node, double const* p,
                                 double const* d, double depth) {
    auto const q = query_reference::arclength(e.node_tree, node, p, d, depth);
    return {q.status_, q.value_};
  }
  namespace atmosphere_detail {
    KOKKOS_INLINE_FUNCTION double radiusVector(EnvironmentSnapshot const& e,
                                               double const* p, double* radial) {
      for (int axis = 0; axis < 3; ++axis)
        radial[axis] = p[axis] - e.earth_center_m[axis];
      return ::sqrt(radial[0] * radial[0] + radial[1] * radial[1] +
                    radial[2] * radial[2]);
    }
  } // namespace atmosphere_detail
  inline constexpr double SpeedOfLightMPerS = 299792458.;
  struct PhotonTransportOutcome {
    PhotonTransportRecord record{};
    ProposalFallbackEvent fallback{};
    std::uint32_t fallback_flag{};
  };
  KOKKOS_INLINE_FUNCTION ProposalFallbackEvent makePhotonTransportFallback(
      EmInteractionRecord const& interaction, ProposalFallbackReason const reason) {
    ProposalFallbackEvent event{};
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
  KOKKOS_INLINE_FUNCTION bool transportFinite(double const value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }
  KOKKOS_INLINE_FUNCTION void advancePhotonState(EmParticleState const& start,
                                                 double distance_m,
                                                 EmParticleState& end) {
    end = start;
    for (int axis = 0; axis < 3; ++axis) {
      end.position_m[axis] = start.position_m[axis] + distance_m * start.direction[axis];
    }
    end.time_s = start.time_s + distance_m / SpeedOfLightMPerS;
  }
  KOKKOS_INLINE_FUNCTION PhotonTransportOutcome transportPhoton(
      EnvironmentSnapshot const& environment, EmInteractionRecord const& interaction,
      ExternalTransportBoundary const external = {}) {
    PhotonTransportOutcome output{};
    auto start = interaction.particle;
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
        environment, start.position_m, start.direction,
        AtmosphereBoundaryPolicy::StraightPathGuard, start.environment_node);
    if (layer.status != AtmosphereStatus::Success) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::UnsupportedGeometry);
      output.fallback_flag = 1;
      return output;
    }
    auto& record = output.record;
    start.environment_node = layer.layer_index;
    if (interaction.status == EmInteractionStatus::ParticleCut) {
      record.interaction = interaction;
      record.interaction.interaction_vertex_reached = 0;
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
        environment, start.position_m, start.direction,
        AtmosphereBoundaryPolicy::StraightPathGuard, start.environment_node);
    auto const observation =
        intersectObservationPlaneStraight(environment, start.position_m, start.direction);
    auto const has_atmosphere_boundary =
        atmosphere_boundary.status == AtmosphereStatus::Success;
    auto const has_observation = !external.disable_observation &&
                                 observation.status == ObservationPlaneStatus::Success;
    if (!has_atmosphere_boundary && !has_observation) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::UnsupportedGeometry);
      output.fallback_flag = 1;
      return output;
    }
    auto const observation_wins =
        has_observation && (!has_atmosphere_boundary ||
                            observation.distance_m < atmosphere_boundary.distance_m);
    auto boundary_distance_m =
        observation_wins ? observation.distance_m : atmosphere_boundary.distance_m;
    auto const material_wins = external.enabled && external.distance_m > 0. &&
                               external.distance_m < boundary_distance_m;
    if (material_wins) boundary_distance_m = external.distance_m;
    auto const limiting_radius_m = observation_wins ? 0. : atmosphere_boundary.radius_m;
    auto const boundary_grammage =
        geometryCompetitionGrammage(environment, layer.layer_index, start.position_m,
                                    start.direction, boundary_distance_m, external);
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
    record.interaction.interaction_vertex_reached = 0;
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
            interaction, ProposalFallbackReason::AtmosphereInverseGrammageFailed);
        output.fallback.diagnostic_status = static_cast<std::int32_t>(distance.status);
        output.fallback.diagnostic_value0 = distance.value;
        output.fallback.diagnostic_value1 = boundary_distance_m;
        output.fallback.diagnostic_value2 = interaction.interaction_grammage_g_per_cm2;
        output.fallback_flag = 1;
        return output;
      }
      advancePhotonState(start, distance.value, record.end);
      auto const vertex = queryAtmosphereLayer(
          environment, record.end.position_m, record.end.direction,
          AtmosphereBoundaryPolicy::StraightPathGuard, start.environment_node);
      if (vertex.status != AtmosphereStatus::Success ||
          !transportFinite(vertex.density_g_per_cm3) ||
          !(vertex.density_g_per_cm3 > 0.)) {
        output.fallback = makePhotonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereVertexLookupFailed);
        output.fallback.diagnostic_status = static_cast<std::int32_t>(vertex.status);
        output.fallback.diagnostic_value0 = vertex.radius_m;
        output.fallback.diagnostic_value1 = vertex.density_g_per_cm3;
        output.fallback.diagnostic_value2 = distance.value;
        output.fallback_flag = 1;
        return output;
      }
      record.limit = PhotonTransportLimit::Interaction;
      record.end_layer_index = vertex.layer_index;
      record.distance_m = distance.value;
      record.traversed_grammage_g_per_cm2 = interaction.interaction_grammage_g_per_cm2;
      record.end_density_g_per_cm3 = vertex.density_g_per_cm3;
      record.interaction.particle = record.end;
      record.interaction.mass_density_g_per_cm3 = vertex.density_g_per_cm3;
      if (exceedsParticleCutTime(record.end.time_s)) {
        record.end.step_id++;
        record.limit = PhotonTransportLimit::ParticleCut;
        record.cut_deposited_energy_GeV = start.energy_GeV;
      } else {
        record.interaction.interaction_vertex_reached = 1;
      }
      return output;
    }

    // Infinite column depth is permitted only as a competing boundary bound.
    // Without a finite winning interaction, never commit an infinite-depth
    // trajectory or silently lose the photon.
    if (!transportFinite(boundary_grammage.value)) {
      output.fallback = makePhotonTransportFallback(
          interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
      output.fallback.diagnostic_status =
          static_cast<std::int32_t>(AtmosphereStatus::GrammageOutOfRange);
      output.fallback.diagnostic_value0 = boundary_distance_m;
      output.fallback_flag = 1;
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
      record.limiting_radius_m =
          atmosphere_detail::radiusVector(environment, record.end.position_m, radial);
      record.limit = PhotonTransportLimit::ObservationSurface;
      record.end_layer_index = -1;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
    } else {
      auto const node = atmosphere_boundary.next_node;
      record.end.environment_node = node;
      auto model = environment.node_tree.model(node);
      if (!model) {
        record.limit = PhotonTransportLimit::EscapedEnvironment;
        record.end_layer_index = node;
        record.end_density_g_per_cm3 = 0.;
      } else {
        auto const next =
            queryAtmosphereLayer(environment, record.end.position_m, record.end.direction,
                                 AtmosphereBoundaryPolicy::StraightPathGuard, node);
        if (next.status != AtmosphereStatus::Success || node == layer.layer_index) {
          output.fallback = makePhotonTransportFallback(
              interaction, ProposalFallbackReason::UnsupportedGeometry);
          output.fallback_flag = 1;
          return output;
        }
        record.limit = PhotonTransportLimit::LayerBoundary;
        record.end_layer_index = node;
        record.end.medium_id = model->mediumId_;
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
} // namespace photon_transport_reference
