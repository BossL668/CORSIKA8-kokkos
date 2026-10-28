/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace lepton_transport_detail {
    KOKKOS_INLINE_FUNCTION void prepare_stopped_annihilation(
        LeptonTransportRecord& record) {
      if constexpr (!EnableStoppedPositronAnnihilation) return;
      if (record.end_.pid_ != static_cast<std::int32_t>(EmPid::Positron) ||
          exceeds_particle_cut_time(record.end_.timeS_))
        return;
      record.interaction_.status_ = InteractionStatus::AtRestAnnihilation;
      record.interaction_.processId_ = AnnihilationProcessId;
      record.interaction_.interactionVertexReached_ = 1;
      record.interaction_.particle_ = record.end_;
      record.interaction_.particle_.energyGeV_ = transport_mass_gev(record.end_.pid_);
      record.interaction_.particleMassGeV_ = transport_mass_gev(record.end_.pid_);
    }
  } // namespace lepton_transport_detail
  template <typename TEnvironmentView>
  KOKKOS_INLINE_FUNCTION std::uint32_t transport_lepton(
      ContinuousPhysicsView const& table, bool applyMoliere,
      TEnvironmentView const& environment, TransportGeometryConfig const& config,
      InteractionRecord const& interaction, LeptonTransportRecord& record,
      FallbackRequest& fallback, ExternalTransportBoundary const external) {
    auto start = interaction.particle_;
    if (!is_charged_lepton_pid(start.pid_)) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::UnsupportedParticle);
      return 1;
    }
    if (interaction.status_ != InteractionStatus::Selected &&
        interaction.status_ != InteractionStatus::DistanceSampled &&
        interaction.status_ != InteractionStatus::NoDiscreteInteraction &&
        interaction.status_ != InteractionStatus::ParticleCut) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::InvalidFinalState);
      return 1;
    }
    if ((interaction.status_ == InteractionStatus::Selected ||
         interaction.status_ == InteractionStatus::DistanceSampled) &&
        (!lepton_continuous_detail::finite(interaction.interactionGrammageGPerCm2_) ||
         interaction.interactionGrammageGPerCm2_ < 0.)) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::InvalidFinalState);
      return 1;
    }

    if (interaction.status_ == InteractionStatus::ParticleCut &&
        is_native_muon_pair_cut(start)) {
      record = native_muon_pair_cut_record(interaction);
      // Exactly the scalar ParticleCut signed kinetic-energy accounting.
      // No magnetic/Moliere calculation and no finite radio trajectory.
      return applyMoliere ? 2U : 0U;
    }
    auto const continuous = prepare_lepton_continuous_step(table, interaction);
    if (continuous.fallbackFlag_ != 0) {
      fallback = continuous.fallback_;
      return 1;
    }
    auto const initial_energy_MeV = start.energyGeV_ * 1000.;
    auto const mass_MeV = continuous.massMeV_;
    auto const transport_cut_MeV = continuous.transportCutMeV_;

    if (continuous.particleCut_ != 0) {
      auto const layer = query_environment(environment, config.query_, start.positionM_,
                                           start.direction_, start.environmentNode_);
      if (layer.status_ != EnvironmentStatus::Success) {
        fallback = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::UnsupportedGeometry);
        return 1;
      }

      start.environmentNode_ = layer.nodeIndex_;

      // This state is already at the lowest energy represented by the
      // continuous table.  It cannot make a finite transport step, so
      // terminate it in-place exactly as ParticleCut would after a
      // continuous step and deposit all remaining kinetic energy.
      record = LeptonTransportRecord{};
      record.interaction_ = interaction;
      record.interaction_.interactionVertexReached_ = 0;
      record.interaction_.particleMassGeV_ =
          query_continuous_mass(table, start.pid_).value_ / 1000.;
      record.start_ = start;
      record.end_ = start;
      record.end_.stepId_++;
      record.inputIndex_ = interaction.inputIndex_;
      record.limit_ = LeptonTransportLimit::ParticleCut;
      record.startLayerIndex_ = layer.nodeIndex_;
      record.endLayerIndex_ = layer.nodeIndex_;
      record.startDensityGPerCm3_ = layer.densityGPerCm3_;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
      record.cutDepositedEnergyGeV_ = (initial_energy_MeV - mass_MeV) / 1000.;
      if (initial_energy_MeV - mass_MeV < transport_cut_MeV)
        lepton_transport_detail::prepare_stopped_annihilation(record);
      // State 2 is a successful zero-length cut that must bypass the
      // split Moliere kernel. That kernel normalizes it to state 0 before
      // fallback counting.
      return applyMoliere ? 2U : 0U;
    }

    auto const continuous_grammage = continuous.maximumGrammageGPerCm2_;

    auto const layer = query_environment(environment, config.query_, start.positionM_,
                                         start.direction_, start.environmentNode_);
    if (layer.status_ != EnvironmentStatus::Success) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::UnsupportedGeometry);
      return 1;
    }
    auto const charge_number = start.pid_ > 0 ? -1. : 1.;
    start.environmentNode_ = layer.nodeIndex_;
    auto const& field_T = environment.getModel(start.environmentNode_)->magneticFieldT_;
    auto const magnetic_limit =
        maximum_uniform_magnetic_step(start, mass_MeV / 1000., charge_number, field_T,
                                      config.query_.maximumMagneticDeflectionRad_);
    if (magnetic_limit.status_ == MagneticStepStatus::InvalidInput ||
        magnetic_limit.status_ == MagneticStepStatus::NonFiniteResult) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::MagneticTransportFailed);
      return 1;
    }

    auto geometry_distance_m = HUGE_VAL;
    auto limiting_radius_m = 0.;
    auto magnetic_step_wins = false;
    auto observation_plane_wins = false;
    auto next_node = start.environmentNode_;
    {
      auto const boundary =
          lepton_boundary(environment.getTree(), start, mass_MeV / 1000., charge_number,
                          field_T, magnetic_limit);
      if (boundary.failed_) {
        fallback = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::MagneticBoundaryFailed);
        return 1;
      }
      geometry_distance_m = boundary.distanceM_;
      next_node = boundary.nextNode_;
      magnetic_step_wins = next_node == start.environmentNode_;
      // Observation is a separate process, not a material-tree node.
      if (!external.disableObservation_) {
        auto const observation = lepton_observation_distance(
            environment.getTree(), start, mass_MeV / 1000., charge_number, field_T,
            magnetic_limit, config.observation_.pointM_, config.observation_.normal_);
        if (observation < geometry_distance_m) {
          geometry_distance_m = observation;
          observation_plane_wins = true;
        }
        if (observation_plane_wins) magnetic_step_wins = false;
      }
    }
    auto const material_boundary_wins = external.enabled_ && external.distanceM_ > 0. &&
                                        external.distanceM_ < geometry_distance_m;
    if (material_boundary_wins) {
      // External distance uses the same leapfrog length, not a chord length.
      // This material interface is never an absorbing detector.
      geometry_distance_m = external.distanceM_;
      observation_plane_wins = false;
      magnetic_step_wins = false;
    }
    auto const geometry_grammage = geometry_competition_grammage(
        environment, config.query_, layer.nodeIndex_, start.positionM_, start.direction_,
        geometry_distance_m);
    if (geometry_grammage.status_ != EnvironmentStatus::Success) {
      auto event = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::AtmosphereGrammageFailed);
      event.diagnosticStatus_ = static_cast<std::int32_t>(geometry_grammage.status_);
      event.diagnosticValue0_ = geometry_distance_m;
      event.diagnosticValue1_ = layer.densityGPerCm3_;
      event.diagnosticValue2_ = limiting_radius_m;
      fallback = event;
      return 1;
    }

    auto const has_interaction =
        interaction.status_ == InteractionStatus::Selected ||
        interaction.status_ == InteractionStatus::DistanceSampled;
    auto const interaction_grammage =
        has_interaction ? interaction.interactionGrammageGPerCm2_ : HUGE_VAL;
    auto continuous_wins = continuous_grammage < geometry_grammage.value_ &&
                           continuous_grammage < interaction_grammage;
    auto interaction_wins = !continuous_wins && has_interaction &&
                            interaction_grammage <= geometry_grammage.value_;

    // Construct the output directly in its device slot. Keeping this large
    // aggregate as a thread-local value caused local-memory spills in the
    // geometry-heavy kernel; failed slots are ignored by compaction.
    record = LeptonTransportRecord{};
    record.interaction_ = interaction;
    record.interaction_.interactionVertexReached_ = 0;
    record.interaction_.particleMassGeV_ =
        query_continuous_mass(table, start.pid_).value_ / 1000.;
    record.start_ = start;
    record.inputIndex_ = interaction.inputIndex_;
    record.startLayerIndex_ = layer.nodeIndex_;
    record.startDensityGPerCm3_ = layer.densityGPerCm3_;
    record.continuousStepGrammageGPerCm2_ = continuous_grammage;
    record.limitingRadiusM_ = limiting_radius_m;
    record.magneticStepStatus_ = static_cast<std::uint32_t>(magnetic_limit.status_);
    record.magneticStepLimitM_ = magnetic_limit.distanceM_;
    record.magneticGyroradiusM_ = magnetic_limit.gyroradiusM_;

    double traversed_grammage = geometry_grammage.value_;
    double distance_m = geometry_distance_m;
    if (continuous_wins) {
      traversed_grammage = continuous_grammage;
    } else if (interaction_wins) {
      traversed_grammage = interaction_grammage;
    }
    if (continuous_wins || interaction_wins) {
      auto const distance =
          environment_arclength(environment, config.query_, layer.nodeIndex_,
                                start.positionM_, start.direction_, traversed_grammage);
      if (distance.status_ != EnvironmentStatus::Success ||
          distance.value_ > geometry_distance_m * (1. + 1.e-12)) {
        auto event = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::AtmosphereInverseGrammageFailed);
        event.diagnosticStatus_ = static_cast<std::int32_t>(distance.status_);
        event.diagnosticValue0_ = distance.value_;
        event.diagnosticValue1_ = geometry_distance_m;
        event.diagnosticValue2_ = traversed_grammage;
        fallback = event;
        return 1;
      }
      distance_m = distance.value_;
    }
    auto const has_decay =
        is_muon_pid(start.pid_) &&
        lepton_continuous_detail::finite(interaction.decayDistanceM_) &&
        interaction.decayDistanceM_ >= 0.;
    auto const decay_wins = has_decay && interaction.decayDistanceM_ <= distance_m;
    if (decay_wins) {
      continuous_wins = false;
      interaction_wins = false;
      distance_m = interaction.decayDistanceM_;
      auto const decay_grammage =
          environment_grammage(environment, config.query_, layer.nodeIndex_,
                               start.positionM_, start.direction_, distance_m);
      if (decay_grammage.status_ != EnvironmentStatus::Success) {
        auto event = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::AtmosphereGrammageFailed);
        event.diagnosticStatus_ = static_cast<std::int32_t>(decay_grammage.status_);
        event.diagnosticValue0_ = distance_m;
        fallback = event;
        return 1;
      }
      traversed_grammage = decay_grammage.value_;
    }
    auto const observation_reached = observation_plane_wins && !continuous_wins &&
                                     !interaction_wins && !decay_wins &&
                                     !magnetic_step_wins;

    // Reuse the trajectory policy already used for boundary competition.
    // Scalar TrackingLeapFrogCurved::getLinearTrajectory explicitly removes
    // B for R > 1e9 m and p_perp < 1 eV/c, not only for an exactly zero field.
    // Passing the physical field here after a straight intersection would
    // move the endpoint off that boundary and corrupt its chord/track record.
    // Keep the environment unchanged: this decision applies to this step only.
    double const zero_field_T[3]{0., 0., 0.};
    auto const& propagation_field_T =
        magnetic_limit.status_ == MagneticStepStatus::Linear ? zero_field_T : field_T;
    auto const magnetic_advance = advance_uniform_magnetic_field(
        start, mass_MeV / 1000., charge_number, propagation_field_T, distance_m);
    if (magnetic_advance.status_ == MagneticStepStatus::InvalidInput ||
        magnetic_advance.status_ == MagneticStepStatus::NonFiniteResult) {
      fallback = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::MagneticTransportFailed);
      return 1;
    }
    record.end_ = magnetic_advance.particle_;
    if (observation_reached) {
      double radial[3]{};
      record.limitingRadiusM_ = photon_transport_detail::radius(
          config.earthCenterM_, record.end_.positionM_, radial);
    }
    record.magneticBendingApplied_ =
        magnetic_advance.status_ == MagneticStepStatus::Success &&
                magnetic_advance.bendParameter_ != 0.
            ? 1U
            : 0U;
    record.magneticBendParameter_ = magnetic_advance.bendParameter_;
    record.magneticChordLengthM_ = magnetic_advance.chordLengthM_;
    record.distanceM_ = distance_m;

    // Scalar CORSIKA uses the original leapfrog trajectory when it turns a
    // sampled grammage into the competing step length.  Once the endpoint
    // has been selected, however, Step retains only the displacement and
    // ContinuousProcess integrates the medium along
    // Step::getStraightTrack(): the chord between the two endpoints.  Keep
    // the start-tangent grammage above for distance competition, then match
    // the scalar continuous-process semantics here after magnetic advance.
    double chord_direction[3]{start.direction_[0], start.direction_[1],
                              start.direction_[2]};
    auto continuous_chord_length_squared = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      auto const displacement = record.end_.positionM_[axis] - start.positionM_[axis];
      chord_direction[axis] = displacement;
      continuous_chord_length_squared += displacement * displacement;
    }
    auto const continuous_chord_length_m = ::sqrt(continuous_chord_length_squared);
    if (continuous_chord_length_m > 0.) {
      for (int axis = 0; axis < 3; ++axis) {
        chord_direction[axis] /= continuous_chord_length_m;
      }
    } else {
      for (int axis = 0; axis < 3; ++axis) {
        chord_direction[axis] = start.direction_[axis];
      }
    }
    auto const chord_grammage = environment_grammage(
        environment, config.query_, layer.nodeIndex_, start.positionM_, chord_direction,
        continuous_chord_length_m);
    if (chord_grammage.status_ != EnvironmentStatus::Success) {
      auto event = lepton_continuous_detail::make_transport_fallback(
          interaction, FallbackReason::AtmosphereGrammageFailed);
      event.diagnosticStatus_ = static_cast<std::int32_t>(chord_grammage.status_);
      event.diagnosticValue0_ = continuous_chord_length_m;
      event.diagnosticValue1_ = distance_m;
      event.diagnosticValue2_ = traversed_grammage;
      fallback = event;
      return 1;
    }
    traversed_grammage = chord_grammage.value_;
    record.traversedGrammageGPerCm2_ = traversed_grammage;
    auto const loss = evaluate_lepton_continuous_loss(
        table, interaction, continuous, traversed_grammage, continuous_wins);
    if (loss.fallbackFlag_ != 0) {
      fallback = loss.fallback_;
      return 1;
    }
    auto const final_energy_MeV = loss.finalEnergyMeV_;
    record.end_.energyGeV_ = final_energy_MeV / 1000.;
    record.continuousDepositedEnergyGeV_ = loss.depositedEnergyGeV_;

    // In the scalar sequence, continuous loss, multiple scattering,
    // profile/radio projection and ObservationPlane all see this completed
    // step before the final ParticleCut check.  Terminate at the endpoint
    // (rather than truncating at exactly 10 ms), suppress any pending
    // interaction/decay, and retain the dual observation+cut outcome when
    // the observation plane was the limiting process.
    if (exceeds_particle_cut_time(record.end_.timeS_)) {
      record.end_.stepId_++;
      record.limit_ = LeptonTransportLimit::ParticleCut;
      record.endLayerIndex_ = layer.nodeIndex_;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
      record.cutDepositedEnergyGeV_ = (final_energy_MeV - mass_MeV) / 1000.;
      record.observationSurfaceReachedBeforeCut_ = observation_reached ? 1U : 0U;
      return 0;
    }

    auto const reaches_cut =
        loss.transportCutReached_ != 0 || final_energy_MeV - mass_MeV < transport_cut_MeV;
    // ObservationPlane does not short-circuit the scalar process sequence:
    // ParticleCut still observes the completed step after ground output.
    // Retain both records, as for the existing time-cut branch above.
    if (reaches_cut) {
      record.end_.stepId_++;
      record.limit_ = LeptonTransportLimit::ParticleCut;
      if (observation_reached) {
        // The absorbing surface may bound the supported medium. Do not
        // require a new medium on the far side of a terminal observation.
        record.endLayerIndex_ = -1;
        record.endDensityGPerCm3_ = layer.densityGPerCm3_;
      } else {
        auto const end_layer =
            query_environment(environment, config.query_, record.end_.positionM_,
                              record.end_.direction_, start.environmentNode_);
        if (end_layer.status_ != EnvironmentStatus::Success) {
          fallback = lepton_continuous_detail::make_transport_fallback(
              interaction, FallbackReason::UnsupportedGeometry);
          return 1;
        }
        record.endLayerIndex_ = end_layer.nodeIndex_;
        record.endDensityGPerCm3_ = end_layer.densityGPerCm3_;
      }
      record.cutDepositedEnergyGeV_ = (final_energy_MeV - mass_MeV) / 1000.;
      record.observationSurfaceReachedBeforeCut_ = observation_reached ? 1U : 0U;
      lepton_transport_detail::prepare_stopped_annihilation(record);
      return 0;
    }

    if (decay_wins) {
      auto const vertex =
          query_environment(environment, config.query_, record.end_.positionM_,
                            record.end_.direction_, start.environmentNode_);
      if (vertex.status_ != EnvironmentStatus::Success) {
        fallback = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::AtmosphereVertexLookupFailed);
        return 1;
      }
      record.limit_ = LeptonTransportLimit::DecayCandidate;
      record.endLayerIndex_ = vertex.nodeIndex_;
      record.endDensityGPerCm3_ = vertex.densityGPerCm3_;
      record.interaction_.particle_ = record.end_;
      record.interaction_.status_ = InteractionStatus::Selected;
      record.interaction_.interactionVertexReached_ = 1;
      record.interaction_.processId_ = DecayProcessId;
      record.interaction_.componentHash_ = 0;
      record.interaction_.energyFraction_ = 0.;
      record.interaction_.lossQuantile_ = 0.;
      return 0;
    }

    if (interaction_wins) {
      auto const vertex =
          query_environment(environment, config.query_, record.end_.positionM_,
                            record.end_.direction_, start.environmentNode_);
      if (vertex.status_ != EnvironmentStatus::Success ||
          !lepton_continuous_detail::finite(vertex.densityGPerCm3_) ||
          !(vertex.densityGPerCm3_ > 0.)) {
        auto event = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::AtmosphereVertexLookupFailed);
        event.diagnosticStatus_ = static_cast<std::int32_t>(vertex.status_);
        event.diagnosticValue0_ = vertex.radiusM_;
        event.diagnosticValue1_ = vertex.densityGPerCm3_;
        event.diagnosticValue2_ = distance_m;
        fallback = event;
        return 1;
      }
      record.limit_ = LeptonTransportLimit::InteractionCandidate;
      record.endLayerIndex_ = vertex.nodeIndex_;
      record.endDensityGPerCm3_ = vertex.densityGPerCm3_;
      record.interaction_.particle_ = record.end_;
      record.interaction_.status_ = InteractionStatus::RequiresReselection;
      record.interaction_.interactionVertexReached_ = 1;
      record.interaction_.processId_ = 0;
      record.interaction_.componentHash_ = 0;
      record.interaction_.energyFraction_ = 0.;
      record.interaction_.lossQuantile_ = 0.;
      record.interaction_.massDensityGPerCm3_ = vertex.densityGPerCm3_;
      return 0;
    }

    record.end_.stepId_++;
    if (continuous_wins) {
      auto const end_layer =
          query_environment(environment, config.query_, record.end_.positionM_,
                            record.end_.direction_, start.environmentNode_);
      if (end_layer.status_ != EnvironmentStatus::Success) {
        fallback = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::UnsupportedGeometry);
        return 1;
      }
      record.limit_ = LeptonTransportLimit::ContinuousStep;
      record.endLayerIndex_ = end_layer.nodeIndex_;
      record.endDensityGPerCm3_ = end_layer.densityGPerCm3_;
      return 0;
    }

    if (magnetic_step_wins) {
      auto const end_layer =
          query_environment(environment, config.query_, record.end_.positionM_,
                            record.end_.direction_, start.environmentNode_);
      if (end_layer.status_ != EnvironmentStatus::Success) {
        auto event = lepton_continuous_detail::make_transport_fallback(
            interaction, FallbackReason::MagneticBoundaryFailed);
        event.diagnosticStatus_ = static_cast<std::int32_t>(end_layer.status_);
        event.diagnosticValue0_ = end_layer.radiusM_;
        event.diagnosticValue1_ = magnetic_limit.distanceM_;
        fallback = event;
        return 1;
      }
      // Assigned-node queries cannot enter the legacy sphere-layer guard.
      record.limit_ = LeptonTransportLimit::MagneticStep;
      record.endLayerIndex_ = end_layer.nodeIndex_;
      record.endDensityGPerCm3_ = end_layer.densityGPerCm3_;
      return 0;
    }

    if (material_boundary_wins) {
      record.limit_ = LeptonTransportLimit::MaterialBoundary;
      record.endLayerIndex_ = layer.nodeIndex_;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
    } else if (observation_reached) {
      record.limit_ = LeptonTransportLimit::ObservationSurface;
      record.endLayerIndex_ = -1;
      record.endDensityGPerCm3_ = layer.densityGPerCm3_;
    } else {
      record.end_.environmentNode_ = next_node;
      auto const* model = environment.getModel(next_node);
      if (!model) {
        record.limit_ = LeptonTransportLimit::EscapedEnvironment;
        record.endLayerIndex_ = next_node;
        record.endDensityGPerCm3_ = 0.;
      } else {
        auto const next =
            query_environment(environment, config.query_, record.end_.positionM_,
                              record.end_.direction_, next_node);
        if (next.status_ != EnvironmentStatus::Success ||
            next_node == start.environmentNode_) {
          fallback = lepton_continuous_detail::make_transport_fallback(
              interaction, FallbackReason::UnsupportedGeometry);
          return 1;
        }
        record.limit_ = LeptonTransportLimit::LayerBoundary;
        record.endLayerIndex_ = next_node;
        record.end_.mediumId_ = model->mediumId_;
        record.endDensityGPerCm3_ = next.densityGPerCm3_;
      }
    }
    return 0;
  }

} // namespace corsika::kokkos
