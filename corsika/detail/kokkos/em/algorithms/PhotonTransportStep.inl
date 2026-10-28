/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace photon_transport_detail {
    KOKKOS_INLINE_FUNCTION FallbackRequest
    make_fallback(InteractionRecord const& interaction, FallbackReason const reason) {
      FallbackRequest event{};
      event.particle_ = interaction.particle_;
      event.inputIndex_ = interaction.inputIndex_;
      event.processId_ = interaction.processId_;
      event.reason_ = reason;
      event.componentHash_ = interaction.componentHash_;
      event.energyFraction_ = interaction.energyFraction_;
      event.lossQuantile_ = interaction.lossQuantile_;
      event.randomDrawId_ = interaction.lossDrawId_;
      return event;
    }
    KOKKOS_INLINE_FUNCTION bool finite(double const value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
    KOKKOS_INLINE_FUNCTION void advance(ParticleState const& start, double distanceM,
                                        ParticleState& end) {
      end = start;
      for (int axis = 0; axis < 3; ++axis) {
        end.positionM_[axis] =
            start.positionM_[axis] + distanceM * start.direction_[axis];
      }
      end.timeS_ = start.timeS_ + distanceM / scalar_constants::SpeedOfLightMPerS;
    }

    KOKKOS_INLINE_FUNCTION double radius(double const (&centerM)[3],
                                         double const (&positionM)[3],
                                         double (&radial)[3]) {
      for (int axis = 0; axis < 3; ++axis) radial[axis] = positionM[axis] - centerM[axis];
#if defined(__CUDA_ARCH__)
      return ::sqrt(radial[0] * radial[0] + radial[1] * radial[1] +
                    radial[2] * radial[2]);
#else
      return std::sqrt(radial[0] * radial[0] + radial[1] * radial[1] +
                       radial[2] * radial[2]);
#endif
    }
  } // namespace photon_transport_detail

  template <typename TEnvironmentView>
  KOKKOS_INLINE_FUNCTION PhotonTransportOutcome transport_photon(
      TEnvironmentView const& environment, TransportGeometryConfig const& config,
      InteractionRecord const& interaction, ExternalTransportBoundary const external) {
    PhotonTransportOutcome output{};
    auto start = interaction.particle_;
    if (start.pid_ != static_cast<std::int32_t>(EmPid::Photon)) {
      output.fallback_ = photon_transport_detail::make_fallback(
          interaction, FallbackReason::UnsupportedParticle);
      output.fallbackFlag_ = 1;
      return output;
    }
    if (interaction.status_ != InteractionStatus::Selected &&
        interaction.status_ != InteractionStatus::NoDiscreteInteraction &&
        interaction.status_ != InteractionStatus::ParticleCut) {
      output.fallback_ = photon_transport_detail::make_fallback(
          interaction, FallbackReason::InvalidFinalState);
      output.fallbackFlag_ = 1;
      return output;
    }

    auto const layer = query_environment(environment, config.query_, start.positionM_,
                                         start.direction_, start.environmentNode_);
    if (layer.status_ != EnvironmentStatus::Success) {
      output.fallback_ = photon_transport_detail::make_fallback(
          interaction, FallbackReason::UnsupportedGeometry);
      output.fallbackFlag_ = 1;
      return output;
    }
    auto& record = output.record_;
    start.environmentNode_ = layer.nodeIndex_;
    if (interaction.status_ == InteractionStatus::ParticleCut) {
      record.interaction_ = interaction;
      record.interaction_.interactionVertexReached_ = 0;
      record.start_ = start;
      record.end_ = start;
      record.end_.stepId_++;
      record.inputIndex_ = interaction.inputIndex_;
      record.limit_ = PhotonTransportLimit::ParticleCut;
      record.startLayerIndex_ = layer.nodeIndex_;
      record.endLayerIndex_ = layer.nodeIndex_;
      record.startDensityGPerCm3_ = layer.densityGPerCm3_;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
      record.cutDepositedEnergyGeV_ = start.energyGeV_;
      return output;
    }

    auto const atmosphere_boundary =
        distance_to_environment_boundary(environment, config.query_, start.positionM_,
                                         start.direction_, start.environmentNode_);
    auto const observation = intersect_observation_plane_straight(
        config.observation_, start.positionM_, start.direction_);
    auto const has_atmosphere_boundary =
        atmosphere_boundary.status_ == EnvironmentStatus::Success;
    auto const has_observation = !external.disableObservation_ &&
                                 observation.status_ == ObservationPlaneStatus::Success;
    if (!has_atmosphere_boundary && !has_observation) {
      output.fallback_ = photon_transport_detail::make_fallback(
          interaction, FallbackReason::UnsupportedGeometry);
      output.fallbackFlag_ = 1;
      return output;
    }
    auto const observation_wins =
        has_observation && (!has_atmosphere_boundary ||
                            observation.distanceM_ < atmosphere_boundary.distanceM_);
    auto boundary_distance_m =
        observation_wins ? observation.distanceM_ : atmosphere_boundary.distanceM_;
    auto const material_wins = external.enabled_ && external.distanceM_ > 0. &&
                               external.distanceM_ < boundary_distance_m;
    if (material_wins) boundary_distance_m = external.distanceM_;
    auto const limiting_radius_m = observation_wins ? 0. : atmosphere_boundary.radiusM_;
    auto const boundary_grammage = geometry_competition_grammage(
        environment, config.query_, layer.nodeIndex_, start.positionM_, start.direction_,
        boundary_distance_m);
    if (boundary_grammage.status_ != EnvironmentStatus::Success) {
      output.fallback_ = photon_transport_detail::make_fallback(
          interaction, FallbackReason::AtmosphereGrammageFailed);
      output.fallback_.diagnosticStatus_ =
          static_cast<std::int32_t>(boundary_grammage.status_);
      output.fallback_.diagnosticValue0_ = boundary_distance_m;
      output.fallback_.diagnosticValue1_ = layer.densityGPerCm3_;
      output.fallback_.diagnosticValue2_ = limiting_radius_m;
      output.fallbackFlag_ = 1;
      return output;
    }

    record.interaction_ = interaction;
    record.interaction_.interactionVertexReached_ = 0;
    record.start_ = start;
    record.inputIndex_ = interaction.inputIndex_;
    record.startLayerIndex_ = layer.nodeIndex_;
    record.startDensityGPerCm3_ = layer.densityGPerCm3_;
    record.limitingRadiusM_ = limiting_radius_m;
    auto const reaches_interaction =
        interaction.status_ == InteractionStatus::Selected &&
        photon_transport_detail::finite(interaction.interactionGrammageGPerCm2_) &&
        interaction.interactionGrammageGPerCm2_ >= 0. &&
        interaction.interactionGrammageGPerCm2_ <= boundary_grammage.value_;
    if (reaches_interaction) {
      auto const distance = environment_arclength(
          environment, config.query_, layer.nodeIndex_, start.positionM_,
          start.direction_, interaction.interactionGrammageGPerCm2_);
      if (distance.status_ != EnvironmentStatus::Success ||
          distance.value_ > boundary_distance_m * (1. + 1.e-12)) {
        output.fallback_ = photon_transport_detail::make_fallback(
            interaction, FallbackReason::AtmosphereInverseGrammageFailed);
        output.fallback_.diagnosticStatus_ = static_cast<std::int32_t>(distance.status_);
        output.fallback_.diagnosticValue0_ = distance.value_;
        output.fallback_.diagnosticValue1_ = boundary_distance_m;
        output.fallback_.diagnosticValue2_ = interaction.interactionGrammageGPerCm2_;
        output.fallbackFlag_ = 1;
        return output;
      }
      photon_transport_detail::advance(start, distance.value_, record.end_);
      auto const vertex =
          query_environment(environment, config.query_, record.end_.positionM_,
                            record.end_.direction_, start.environmentNode_);
      if (vertex.status_ != EnvironmentStatus::Success ||
          !photon_transport_detail::finite(vertex.densityGPerCm3_) ||
          !(vertex.densityGPerCm3_ > 0.)) {
        output.fallback_ = photon_transport_detail::make_fallback(
            interaction, FallbackReason::AtmosphereVertexLookupFailed);
        output.fallback_.diagnosticStatus_ = static_cast<std::int32_t>(vertex.status_);
        output.fallback_.diagnosticValue0_ = vertex.radiusM_;
        output.fallback_.diagnosticValue1_ = vertex.densityGPerCm3_;
        output.fallback_.diagnosticValue2_ = distance.value_;
        output.fallbackFlag_ = 1;
        return output;
      }
      record.limit_ = PhotonTransportLimit::Interaction;
      record.endLayerIndex_ = vertex.nodeIndex_;
      record.distanceM_ = distance.value_;
      record.traversedGrammageGPerCm2_ = interaction.interactionGrammageGPerCm2_;
      record.endDensityGPerCm3_ = vertex.densityGPerCm3_;
      record.interaction_.particle_ = record.end_;
      record.interaction_.massDensityGPerCm3_ = vertex.densityGPerCm3_;
      if (exceeds_particle_cut_time(record.end_.timeS_)) {
        record.end_.stepId_++;
        record.limit_ = PhotonTransportLimit::ParticleCut;
        record.cutDepositedEnergyGeV_ = start.energyGeV_;
      } else {
        record.interaction_.interactionVertexReached_ = 1;
      }
      return output;
    }

    // Infinite column depth is permitted only as a competing boundary bound.
    // Without a finite winning interaction, never commit an infinite-depth
    // trajectory or silently lose the photon.
    if (!photon_transport_detail::finite(boundary_grammage.value_)) {
      output.fallback_ = photon_transport_detail::make_fallback(
          interaction, FallbackReason::AtmosphereGrammageFailed);
      output.fallback_.diagnosticStatus_ =
          static_cast<std::int32_t>(EnvironmentStatus::GrammageOutOfRange);
      output.fallback_.diagnosticValue0_ = boundary_distance_m;
      output.fallbackFlag_ = 1;
      return output;
    }
    photon_transport_detail::advance(start, boundary_distance_m, record.end_);
    record.end_.stepId_++;
    record.distanceM_ = boundary_distance_m;
    record.traversedGrammageGPerCm2_ = boundary_grammage.value_;
    if (material_wins) {
      record.limit_ = PhotonTransportLimit::MaterialBoundary;
      record.endLayerIndex_ = layer.nodeIndex_;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
    } else if (observation_wins) {
      double radial[3]{};
      record.limitingRadiusM_ = photon_transport_detail::radius(
          config.earthCenterM_, record.end_.positionM_, radial);
      record.limit_ = PhotonTransportLimit::ObservationSurface;
      record.endLayerIndex_ = -1;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
    } else {
      auto const node = atmosphere_boundary.nextNode_;
      record.end_.environmentNode_ = node;
      auto model = environment.getModel(node);
      if (!model) {
        record.limit_ = PhotonTransportLimit::EscapedEnvironment;
        record.endLayerIndex_ = node;
        record.endDensityGPerCm3_ = 0.;
      } else {
        auto const next =
            query_environment(environment, config.query_, record.end_.positionM_,
                              record.end_.direction_, node);
        if (next.status_ != EnvironmentStatus::Success || node == layer.nodeIndex_) {
          output.fallback_ = photon_transport_detail::make_fallback(
              interaction, FallbackReason::UnsupportedGeometry);
          output.fallbackFlag_ = 1;
          return output;
        }
        record.limit_ = PhotonTransportLimit::LayerBoundary;
        record.endLayerIndex_ = node;
        record.end_.mediumId_ = model->mediumId_;
        record.endDensityGPerCm3_ = next.densityGPerCm3_;
      }
    }
    if (exceeds_particle_cut_time(record.end_.timeS_)) {
      record.observationSurfaceReachedBeforeCut_ =
          record.limit_ == PhotonTransportLimit::ObservationSurface ? 1U : 0U;
      record.limit_ = PhotonTransportLimit::ParticleCut;
      record.cutDepositedEnergyGeV_ = start.energyGeV_;
    }
    return output;
  }

} // namespace corsika::kokkos
