/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include <corsika/accelerator/em/detail/LeptonContinuousStep.hpp>
#include <corsika/accelerator/em/common/ObservationPlane.hpp>
#include <corsika/accelerator/em/common/ExternalTransportBoundary.hpp>
#include <corsika/accelerator/em/common/SphericalAtmosphere.hpp>
#include <corsika/accelerator/em/common/UniformMagneticField.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>

namespace corsika::accelerator::em::detail {

  C8_ACCELERATOR_INLINE_FUNCTION inline void prepareStoppedAnnihilation(
      gpu::em::LeptonTransportRecord& record) {
    using namespace gpu::em;
    if (record.end.pid != static_cast<std::int32_t>(EmPid::Positron) ||
        exceedsParticleCutTime(record.end.time_s)) return;
    record.interaction.status = EmInteractionStatus::AtRestAnnihilation;
    record.interaction.process_id = AnnihilationProcessId;
    record.interaction.interaction_vertex_reached = 1;
    record.interaction.particle = record.end;
    record.interaction.particle.energy_GeV = transportMassGeV(record.end.pid);
    record.interaction.particle_mass_GeV = transportMassGeV(record.end.pid);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool leptonTransportRadiusClose(
      double const left, double const right) {
    auto const scale = left > right ? left : right;
    return ::fabs(left - right) <=
           128. * 2.22044604925031308085e-16 *
               (scale > 1. ? scale : 1.);
  }

  /**
   * One complete charged-lepton propagation step shared by native CUDA and
   * every Kokkos execution space.  The return value is the stable compaction
   * state: 0 success, 1 explicit CPU fallback, 2 zero-length cut that must
   * bypass the split Moliere stage.
   */
  C8_ACCELERATOR_INLINE_FUNCTION inline std::uint32_t transportLepton(
      gpu::em::tables::NativePhysicsView const& table,
      bool const apply_moliere,
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::EmInteractionRecord const& interaction,
      gpu::em::LeptonTransportRecord& record,
      gpu::em::ProposalFallbackEvent& fallback,
      gpu::em::ExternalTransportBoundary const external = {}) {
    using namespace gpu::em;
      auto const& start = interaction.particle;
      if (!isChargedLeptonPid(start.pid)) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::UnsupportedParticle);
        return 1;
      }
      if (interaction.status != EmInteractionStatus::Selected &&
          interaction.status !=
              EmInteractionStatus::DistanceSampled &&
          interaction.status !=
              EmInteractionStatus::NoDiscreteInteraction &&
          interaction.status !=
              EmInteractionStatus::ParticleCut) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::InvalidFinalState);
        return 1;
      }
      if ((interaction.status == EmInteractionStatus::Selected ||
           interaction.status ==
               EmInteractionStatus::DistanceSampled) &&
          (!leptonContinuousFinite(interaction.interaction_grammage_g_per_cm2) ||
           interaction.interaction_grammage_g_per_cm2 < 0.)) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::InvalidFinalState);
        return 1;
      }

      auto const continuous =
          prepareLeptonContinuousStep(
              table, interaction);
      if (continuous.fallback_flag != 0) {
        fallback = continuous.fallback;
        return 1;
      }
      auto const initial_energy_MeV = start.energy_GeV * 1000.;
      auto const mass_MeV = continuous.mass_MeV;
      auto const transport_cut_MeV = continuous.transport_cut_MeV;

      if (continuous.particle_cut != 0) {
        auto const layer = queryAtmosphereLayer(
            environment, start.position_m, start.direction);
        if (layer.status != AtmosphereStatus::Success) {
          fallback = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::UnsupportedGeometry);
          return 1;
        }

        // This state is already at the lowest energy represented by the
        // continuous table.  It cannot make a finite transport step, so
        // terminate it in-place exactly as ParticleCut would after a
        // continuous step and deposit all remaining kinetic energy.
                record = LeptonTransportRecord{};
        record.interaction = interaction;
        record.interaction.interaction_vertex_reached = 0;
        record.interaction.particle_mass_GeV =
            tables::queryContinuousMass(table, start.pid).value / 1000.;
        record.start = start;
        record.end = start;
        record.end.step_id++;
        record.input_index = interaction.input_index;
        record.limit = LeptonTransportLimit::ParticleCut;
        record.start_layer_index = layer.layer_index;
        record.end_layer_index = layer.layer_index;
        record.start_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.cut_deposited_energy_GeV =
            (initial_energy_MeV - mass_MeV) / 1000.;
        if (initial_energy_MeV - mass_MeV < transport_cut_MeV)
          prepareStoppedAnnihilation(record);
        // State 2 is a successful zero-length cut that must bypass the
        // split Moliere kernel. That kernel normalizes it to state 0 before
        // fallback counting.
        return apply_moliere ? 2U : 0U;
      }

      auto const continuous_grammage =
          continuous.maximum_grammage_g_per_cm2;

      auto const layer = queryAtmosphereLayer(
          environment, start.position_m, start.direction);
      if (layer.status != AtmosphereStatus::Success) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::UnsupportedGeometry);
        return 1;
      }
      auto const charge_number = start.pid > 0 ? -1. : 1.;
      auto const magnetic_limit = maximumUniformMagneticStep(
          start, mass_MeV / 1000., charge_number,
          environment.magnetic_field_T,
          environment.maximum_magnetic_deflection_rad);
      if (magnetic_limit.status ==
              MagneticStepStatus::InvalidInput ||
          magnetic_limit.status ==
              MagneticStepStatus::NonFiniteResult) {
        fallback = makeLeptonTransportFallback(
            interaction,
            ProposalFallbackReason::MagneticTransportFailed);
        return 1;
      }

      auto geometry_distance_m = HUGE_VAL;
      auto limiting_radius_m = 0.;
      auto magnetic_step_wins = false;
      auto observation_plane_wins = false;
      if (magnetic_limit.status ==
          MagneticStepStatus::Linear) {
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
          fallback = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::UnsupportedGeometry);
          return 1;
        }
        observation_plane_wins =
            has_observation &&
            (!has_atmosphere_boundary ||
             observation.distance_m < atmosphere_boundary.distance_m);
        geometry_distance_m =
            observation_plane_wins
                ? observation.distance_m
                : atmosphere_boundary.distance_m;
        limiting_radius_m =
            observation_plane_wins ? 0.
                                   : atmosphere_boundary.radius_m;
      } else {
        auto const& current_layer =
            environment.atmosphere_layers[layer.layer_index];
        auto const inner_radius_m = current_layer.inner_radius_m;
        auto const inner =
            intersectUniformMagneticSphere(
                start, mass_MeV / 1000., charge_number,
                environment.magnetic_field_T,
                environment.earth_center_m, inner_radius_m,
                magnetic_limit.distance_m,
                environment.maximum_magnetic_deflection_rad, -1);
        auto const outer =
            intersectUniformMagneticSphere(
                start, mass_MeV / 1000., charge_number,
                environment.magnetic_field_T,
                environment.earth_center_m,
                current_layer.outer_radius_m,
                magnetic_limit.distance_m,
                environment.maximum_magnetic_deflection_rad, +1);
        auto const observation =
            intersectUniformMagneticPlane(
                start, mass_MeV / 1000., charge_number,
                environment.magnetic_field_T, environment,
                magnetic_limit.distance_m,
                environment.maximum_magnetic_deflection_rad);
        if (inner.status ==
                MagneticIntersectionStatus::InvalidInput ||
            inner.status ==
                MagneticIntersectionStatus::NonFiniteResult ||
            outer.status ==
                MagneticIntersectionStatus::InvalidInput ||
            outer.status ==
                MagneticIntersectionStatus::NonFiniteResult ||
            observation.status ==
                MagneticIntersectionStatus::InvalidInput ||
            observation.status ==
                MagneticIntersectionStatus::NonFiniteResult) {
          fallback = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::MagneticBoundaryFailed);
          return 1;
        }
        auto const use_inner_sphere =
            inner.status ==
                MagneticIntersectionStatus::Success &&
            (outer.status !=
                 MagneticIntersectionStatus::Success ||
             inner.distance_m < outer.distance_m);
        auto const has_sphere_boundary =
            use_inner_sphere ||
            outer.status ==
                MagneticIntersectionStatus::Success;
        auto const sphere_distance_m =
            has_sphere_boundary
                ? (use_inner_sphere ? inner.distance_m
                                    : outer.distance_m)
                : HUGE_VAL;
        auto const has_observation =
            !external.disable_observation &&
            observation.status ==
                MagneticIntersectionStatus::Success;
        observation_plane_wins =
            has_observation &&
            observation.distance_m < sphere_distance_m;
        if (observation_plane_wins) {
          geometry_distance_m = observation.distance_m;
          limiting_radius_m = 0.;
        } else if (has_sphere_boundary) {
          geometry_distance_m =
              sphere_distance_m;
          limiting_radius_m =
              use_inner_sphere ? inner_radius_m
                               : current_layer.outer_radius_m;
        } else {
          geometry_distance_m =
              magnetic_limit.distance_m;
          magnetic_step_wins = true;
        }
      }
      auto const material_boundary_wins = external.enabled &&
          external.distance_m > 0. && external.distance_m < geometry_distance_m;
      if (material_boundary_wins) {
        // External distance uses the same leapfrog length, not a chord length.
        // This material interface is never an absorbing detector.
        geometry_distance_m = external.distance_m;
        observation_plane_wins = false;
        magnetic_step_wins = false;
      }
      auto const geometry_grammage = geometryCompetitionGrammage(
          environment, layer.layer_index, start.position_m,
          start.direction, geometry_distance_m, external);
      if (geometry_grammage.status != AtmosphereStatus::Success) {
        auto event = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
        event.diagnostic_status =
            static_cast<std::int32_t>(geometry_grammage.status);
        event.diagnostic_value0 = geometry_distance_m;
        event.diagnostic_value1 = layer.density_g_per_cm3;
        event.diagnostic_value2 = limiting_radius_m;
        fallback = event;
        return 1;
      }

      auto const has_interaction =
          interaction.status == EmInteractionStatus::Selected ||
          interaction.status ==
              EmInteractionStatus::DistanceSampled;
      auto const interaction_grammage =
          has_interaction
              ? interaction.interaction_grammage_g_per_cm2
              : HUGE_VAL;
      auto continuous_wins =
          continuous_grammage < geometry_grammage.value &&
          continuous_grammage < interaction_grammage;
      auto interaction_wins =
          !continuous_wins && has_interaction &&
          interaction_grammage <= geometry_grammage.value;

      // Construct the output directly in its device slot. Keeping this large
      // aggregate as a thread-local value caused local-memory spills in the
      // geometry-heavy kernel; failed slots are ignored by compaction.
            record = LeptonTransportRecord{};
      record.interaction = interaction;
      record.interaction.interaction_vertex_reached = 0;
      record.interaction.particle_mass_GeV =
          tables::queryContinuousMass(table, start.pid).value / 1000.;
      record.start = start;
      record.input_index = interaction.input_index;
      record.start_layer_index = layer.layer_index;
      record.start_density_g_per_cm3 =
          layer.density_g_per_cm3;
      record.continuous_step_grammage_g_per_cm2 =
          continuous_grammage;
      record.limiting_radius_m = limiting_radius_m;
      record.magnetic_step_status =
          static_cast<std::uint32_t>(magnetic_limit.status);
      record.magnetic_step_limit_m =
          magnetic_limit.distance_m;
      record.magnetic_gyroradius_m =
          magnetic_limit.gyroradius_m;

      double traversed_grammage = geometry_grammage.value;
      double distance_m = geometry_distance_m;
      if (continuous_wins) {
        traversed_grammage = continuous_grammage;
      } else if (interaction_wins) {
        traversed_grammage = interaction_grammage;
      }
      if (continuous_wins || interaction_wins) {
        auto const distance = atmosphereDistanceFromGrammage(
            environment, layer.layer_index, start.position_m,
            start.direction, traversed_grammage);
        if (distance.status != AtmosphereStatus::Success ||
            distance.value >
                geometry_distance_m * (1. + 1.e-12)) {
          auto event = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::AtmosphereInverseGrammageFailed);
          event.diagnostic_status =
              static_cast<std::int32_t>(distance.status);
          event.diagnostic_value0 = distance.value;
          event.diagnostic_value1 = geometry_distance_m;
          event.diagnostic_value2 = traversed_grammage;
          fallback = event;
          return 1;
        }
        distance_m = distance.value;
      }
      auto const has_decay =
          isMuonPid(start.pid) &&
          leptonContinuousFinite(interaction.decay_distance_m) &&
          interaction.decay_distance_m >= 0.;
      auto const decay_wins =
          has_decay &&
          interaction.decay_distance_m <= distance_m;
      if (decay_wins) {
        continuous_wins = false;
        interaction_wins = false;
        distance_m = interaction.decay_distance_m;
        auto const decay_grammage = atmosphereGrammage(
            environment, layer.layer_index, start.position_m,
            start.direction, distance_m);
        if (decay_grammage.status !=
            AtmosphereStatus::Success) {
          auto event = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::AtmosphereGrammageFailed);
          event.diagnostic_status =
              static_cast<std::int32_t>(
                  decay_grammage.status);
          event.diagnostic_value0 = distance_m;
          fallback = event;
          return 1;
        }
        traversed_grammage = decay_grammage.value;
      }
      auto const observation_reached =
          observation_plane_wins && !continuous_wins &&
          !interaction_wins && !decay_wins &&
          !magnetic_step_wins;

      // Reuse the trajectory policy already used for boundary competition.
      // Scalar TrackingLeapFrogCurved::getLinearTrajectory explicitly removes
      // B for R > 1e9 m and p_perp < 1 eV/c, not only for an exactly zero field.
      // Passing the physical field here after a straight intersection would
      // move the endpoint off that boundary and corrupt its chord/track record.
      // Keep the environment unchanged: this decision applies to this step only.
      double const zero_field_T[3]{0., 0., 0.};
      auto const* propagation_field_T =
          magnetic_limit.status == MagneticStepStatus::Linear
              ? zero_field_T
              : environment.magnetic_field_T;
      auto const magnetic_advance =
          advanceUniformMagneticField(
              start, mass_MeV / 1000., charge_number,
              propagation_field_T, distance_m);
      if (magnetic_advance.status ==
              MagneticStepStatus::InvalidInput ||
          magnetic_advance.status ==
              MagneticStepStatus::NonFiniteResult) {
        fallback = makeLeptonTransportFallback(
            interaction,
            ProposalFallbackReason::MagneticTransportFailed);
        return 1;
      }
      record.end = magnetic_advance.particle;
      if (observation_reached) {
        double radial[3]{};
        record.limiting_radius_m =
            atmosphere_detail::radiusVector(
                environment, record.end.position_m, radial);
      }
      record.magnetic_bending_applied =
          magnetic_advance.status ==
                  MagneticStepStatus::Success &&
                  magnetic_advance.bend_parameter != 0.
              ? 1U
              : 0U;
      record.magnetic_bend_parameter =
          magnetic_advance.bend_parameter;
      record.magnetic_chord_length_m =
          magnetic_advance.chord_length_m;
      record.distance_m = distance_m;

      // Scalar CORSIKA uses the original leapfrog trajectory when it turns a
      // sampled grammage into the competing step length.  Once the endpoint
      // has been selected, however, Step retains only the displacement and
      // ContinuousProcess integrates the medium along
      // Step::getStraightTrack(): the chord between the two endpoints.  Keep
      // the start-tangent grammage above for distance competition, then match
      // the scalar continuous-process semantics here after magnetic advance.
      double chord_direction[3]{
          start.direction[0], start.direction[1],
          start.direction[2]};
      auto continuous_chord_length_squared = 0.;
      for (int axis = 0; axis < 3; ++axis) {
        auto const displacement =
            record.end.position_m[axis] - start.position_m[axis];
        chord_direction[axis] = displacement;
        continuous_chord_length_squared += displacement * displacement;
      }
      auto const continuous_chord_length_m =
          ::sqrt(continuous_chord_length_squared);
      if (continuous_chord_length_m > 0.) {
        for (int axis = 0; axis < 3; ++axis) {
          chord_direction[axis] /= continuous_chord_length_m;
        }
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          chord_direction[axis] = start.direction[axis];
        }
      }
      auto const chord_grammage = atmosphereGrammage(
          environment, layer.layer_index, start.position_m,
          chord_direction, continuous_chord_length_m);
      if (chord_grammage.status != AtmosphereStatus::Success) {
        auto event = makeLeptonTransportFallback(
            interaction,
            ProposalFallbackReason::AtmosphereGrammageFailed);
        event.diagnostic_status =
            static_cast<std::int32_t>(chord_grammage.status);
        event.diagnostic_value0 =
            continuous_chord_length_m;
        event.diagnostic_value1 = distance_m;
        event.diagnostic_value2 = traversed_grammage;
        fallback = event;
        return 1;
      }
      traversed_grammage = chord_grammage.value;
      record.traversed_grammage_g_per_cm2 =
          traversed_grammage;
      auto const loss =
          evaluateLeptonContinuousLoss(
              table, interaction, continuous, traversed_grammage,
              continuous_wins);
      if (loss.fallback_flag != 0) {
        fallback = loss.fallback;
        return 1;
      }
      auto const final_energy_MeV = loss.final_energy_MeV;
      record.end.energy_GeV = final_energy_MeV / 1000.;
      record.continuous_deposited_energy_GeV =
          loss.deposited_energy_GeV;

      // In the scalar sequence, continuous loss, multiple scattering,
      // profile/radio projection and ObservationPlane all see this completed
      // step before the final ParticleCut check.  Terminate at the endpoint
      // (rather than truncating at exactly 10 ms), suppress any pending
      // interaction/decay, and retain the dual observation+cut outcome when
      // the observation plane was the limiting process.
      if (exceedsParticleCutTime(record.end.time_s)) {
        record.end.step_id++;
        record.limit = LeptonTransportLimit::ParticleCut;
        record.end_layer_index = layer.layer_index;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
        record.cut_deposited_energy_GeV =
            (final_energy_MeV - mass_MeV) / 1000.;
        record.observation_surface_reached_before_cut =
            observation_reached ? 1U : 0U;
        return 0;
      }

      auto const reaches_cut =
          loss.transport_cut_reached != 0 ||
          final_energy_MeV - mass_MeV <
              transport_cut_MeV;
      // ObservationPlane does not short-circuit the scalar process sequence:
      // ParticleCut still observes the completed step after ground output.
      // Retain both records, as for the existing time-cut branch above.
      if (reaches_cut) {
        record.end.step_id++;
        record.limit = LeptonTransportLimit::ParticleCut;
        if (observation_reached) {
          // The absorbing surface may bound the supported medium. Do not
          // require a new medium on the far side of a terminal observation.
          record.end_layer_index = -1;
          record.end_density_g_per_cm3 = layer.density_g_per_cm3;
        } else {
          auto const end_layer = queryAtmosphereLayer(
              environment, record.end.position_m, record.end.direction);
          if (end_layer.status != AtmosphereStatus::Success) {
            fallback = makeLeptonTransportFallback(
                interaction, ProposalFallbackReason::UnsupportedGeometry);
            return 1;
          }
          record.end_layer_index = end_layer.layer_index;
          record.end_density_g_per_cm3 = end_layer.density_g_per_cm3;
        }
        record.cut_deposited_energy_GeV =
            (final_energy_MeV - mass_MeV) / 1000.;
        record.observation_surface_reached_before_cut =
            observation_reached ? 1U : 0U;
        prepareStoppedAnnihilation(record);
        return 0;
      }

      if (decay_wins) {
        auto const vertex = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (vertex.status != AtmosphereStatus::Success) {
          fallback = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::
                  AtmosphereVertexLookupFailed);
          return 1;
        }
        record.limit = LeptonTransportLimit::DecayCandidate;
        record.end_layer_index = vertex.layer_index;
        record.end_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        record.interaction.particle = record.end;
        record.interaction.status =
            EmInteractionStatus::Selected;
        record.interaction.interaction_vertex_reached = 1;
        record.interaction.process_id = DecayProcessId;
        record.interaction.component_hash = 0;
        record.interaction.energy_fraction = 0.;
        record.interaction.loss_quantile = 0.;
        return 0;
      }

      if (interaction_wins) {
        auto const vertex = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (vertex.status != AtmosphereStatus::Success ||
            !leptonContinuousFinite(vertex.density_g_per_cm3) ||
            !(vertex.density_g_per_cm3 > 0.)) {
          auto event = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::AtmosphereVertexLookupFailed);
          event.diagnostic_status =
              static_cast<std::int32_t>(vertex.status);
          event.diagnostic_value0 = vertex.radius_m;
          event.diagnostic_value1 = vertex.density_g_per_cm3;
          event.diagnostic_value2 = distance_m;
          fallback = event;
          return 1;
        }
        record.limit =
            LeptonTransportLimit::InteractionCandidate;
        record.end_layer_index = vertex.layer_index;
        record.end_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        record.interaction.particle = record.end;
        record.interaction.status =
            EmInteractionStatus::RequiresReselection;
        record.interaction.interaction_vertex_reached = 1;
        record.interaction.process_id = 0;
        record.interaction.component_hash = 0;
        record.interaction.energy_fraction = 0.;
        record.interaction.loss_quantile = 0.;
        record.interaction.mass_density_g_per_cm3 =
            vertex.density_g_per_cm3;
        return 0;
      }

      record.end.step_id++;
      if (continuous_wins) {
        auto const end_layer = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (end_layer.status != AtmosphereStatus::Success) {
          fallback = makeLeptonTransportFallback(
              interaction, ProposalFallbackReason::UnsupportedGeometry);
          return 1;
        }
        record.limit =
            LeptonTransportLimit::ContinuousStep;
        record.end_layer_index = end_layer.layer_index;
        record.end_density_g_per_cm3 =
            end_layer.density_g_per_cm3;
        return 0;
      }

      if (magnetic_step_wins) {
        auto const end_layer = queryAtmosphereLayer(
            environment, record.end.position_m,
            record.end.direction);
        if (end_layer.status != AtmosphereStatus::Success) {
          auto event = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::MagneticBoundaryFailed);
          event.diagnostic_status =
              static_cast<std::int32_t>(end_layer.status);
          event.diagnostic_value0 = end_layer.radius_m;
          event.diagnostic_value1 =
              magnetic_limit.distance_m;
          fallback = event;
          return 1;
        }
        if (end_layer.layer_index != layer.layer_index) {
          // queryAtmosphereLayer deliberately assigns a point within the
          // 0.1 mm tracking guard to the layer selected by its direction.
          // The bounded curved-sphere solver can correctly report no root
          // when the root lies only a few micrometres beyond the magnetic
          // step.  Accept that direction-owned adjacent-layer transition;
          // otherwise a valid maximum-deflection step is misclassified as
          // a numerical transport failure on the next ownership query.
          auto const outward =
              end_layer.layer_index == layer.layer_index + 1;
          auto const inward =
              end_layer.layer_index == layer.layer_index - 1;
          auto const shared_radius_m =
              outward
                  ? environment
                        .atmosphere_layers[layer.layer_index]
                        .outer_radius_m
                  : (inward
                         ? environment
                               .atmosphere_layers[layer.layer_index]
                               .inner_radius_m
                         : 0.);
          auto const floating_tolerance_m =
              128. * 2.22044604925031308085e-16 *
              (shared_radius_m > 1. ? shared_radius_m : 1.);
          auto const boundary_tolerance_m =
              floating_tolerance_m > AtmosphereBoundaryGuardM
                  ? floating_tolerance_m
                  : AtmosphereBoundaryGuardM;
          if ((outward || inward) &&
              ::fabs(end_layer.radius_m - shared_radius_m) <=
                  boundary_tolerance_m) {
            record.limit = LeptonTransportLimit::LayerBoundary;
            record.limiting_radius_m = shared_radius_m;
            record.end_layer_index = end_layer.layer_index;
            record.end_density_g_per_cm3 =
                end_layer.density_g_per_cm3;
            record.end.medium_id =
                environment
                    .atmosphere_layers[end_layer.layer_index]
                    .medium_id;
            return 0;
          }
          auto event = makeLeptonTransportFallback(
              interaction,
              ProposalFallbackReason::MagneticBoundaryFailed);
          event.diagnostic_status = end_layer.layer_index;
          event.diagnostic_value0 = end_layer.radius_m;
          event.diagnostic_value1 = shared_radius_m;
          event.diagnostic_value2 =
              magnetic_limit.distance_m;
          fallback = event;
          return 1;
        }
        record.limit = LeptonTransportLimit::MagneticStep;
        record.end_layer_index = end_layer.layer_index;
        record.end_density_g_per_cm3 =
            end_layer.density_g_per_cm3;
        return 0;
      }

      if (material_boundary_wins) {
        record.limit = LeptonTransportLimit::MaterialBoundary;
        record.end_layer_index = layer.layer_index;
        record.end_density_g_per_cm3 = layer.density_g_per_cm3;
      } else if (observation_reached) {
        record.limit =
            LeptonTransportLimit::ObservationSurface;
        record.end_layer_index = -1;
        record.end_density_g_per_cm3 =
            layer.density_g_per_cm3;
      } else {
        auto const outermost =
            environment.atmosphere_layers[
                environment.number_of_layers - 1]
                .outer_radius_m;
        if (environment.geometry == EnvironmentGeometry::HomogeneousConvexPolyhedron ||
            leptonTransportRadiusClose(limiting_radius_m, outermost)) {
          record.limit =
              LeptonTransportLimit::EscapedEnvironment;
          record.end_layer_index = -1;
          record.end_density_g_per_cm3 = 0.;
        } else {
          auto const next = queryAtmosphereLayer(
              environment, record.end.position_m,
              record.end.direction);
          if (next.status != AtmosphereStatus::Success ||
              next.layer_index == layer.layer_index) {
            fallback = makeLeptonTransportFallback(
                interaction,
                ProposalFallbackReason::UnsupportedGeometry);
            return 1;
          }
          record.limit =
              LeptonTransportLimit::LayerBoundary;
          record.end_layer_index = next.layer_index;
          record.end.medium_id =
              environment
                  .atmosphere_layers[next.layer_index]
                  .medium_id;
          record.end_density_g_per_cm3 =
              next.density_g_per_cm3;
        }
      }
      return 0;
  }

} // namespace corsika::accelerator::em::detail
