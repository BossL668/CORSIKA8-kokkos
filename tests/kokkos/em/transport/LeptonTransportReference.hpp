/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Fixed production 41cdc543; no migrated transport/continuous/geometry
// implementation is used to compute the reference. Adapters change records only.
#include "TransportReference.hpp"
#include "../continuous/ContinuousAdapters.hpp"
#include "PhotonTransportReference.hpp"
#include "../../environment/LeptonBoundaryReference.hpp"
#include "../../environment/UniformMagneticFieldReference.hpp"

namespace lepton_transport_reference {
  using namespace transport_reference;
  using continuous_reference::ProposalFallbackEvent;
  using continuous_reference::ProposalFallbackReason;
  using continuous_reference::step::evaluateLeptonContinuousLoss;
  using continuous_reference::step::leptonContinuousFinite;
  using continuous_reference::step::makeLeptonTransportFallback;
  using continuous_reference::step::prepareLeptonContinuousStep;
  namespace tables = continuous_reference::tables;
  namespace query_reference = environment_query_reference;
  using photon_transport_reference::atmosphereDistanceFromGrammage;
  using photon_transport_reference::AtmosphereStatus;
  using photon_transport_reference::EnvironmentSnapshot;
  using photon_transport_reference::ExternalTransportBoundary;
  using photon_transport_reference::geometryCompetitionGrammage;
  using photon_transport_reference::GrammageQuery;
  using photon_transport_reference::LayerQuery;
  namespace atmosphere_detail = photon_transport_reference::atmosphere_detail;
  using uniform_magnetic_reference::MagneticStepLimitResult;
  using uniform_magnetic_reference::MagneticStepStatus;
  enum class AtmosphereBoundaryPolicy { RadialGuard };
  inline constexpr int DecayProcessId = 1000000011;
  inline constexpr int AnnihilationProcessId = 1000000012;
  inline constexpr bool EnableStoppedPositronAnnihilation = false;
  using EmPid = stochastic_reference::EmPid;
  using stochastic_reference::exceedsParticleCutTime;
  using stochastic_reference::isChargedLeptonPid;
  using stochastic_reference::isMuonPid;

  KOKKOS_INLINE_FUNCTION LayerQuery queryAtmosphereLayer(EnvironmentSnapshot const& e,
                                                         double const* p, double const* d,
                                                         AtmosphereBoundaryPolicy,
                                                         int node) {
    auto q = query_reference::query(e.node_tree, p, d, node);
    return {q.status_, q.node_, q.radiusM_, q.densityGPerCm3_};
  }
  KOKKOS_INLINE_FUNCTION GrammageQuery atmosphereGrammage(EnvironmentSnapshot const& e,
                                                          int node, double const* p,
                                                          double const* d,
                                                          double length) {
    auto q = query_reference::grammage(e.node_tree, node, p, d, length);
    return {q.status_, q.value_};
  }
  KOKKOS_INLINE_FUNCTION corsika::kokkos::ParticleState particle(
      EmParticleState const& old) {
    corsika::kokkos::ParticleState result;
    result.pid_ = old.pid;
    result.mediumId_ = old.medium_id;
    result.environmentNode_ = old.environment_node;
    result.generation_ = old.generation;
    result.reserved_ = old.reserved;
    result.energyGeV_ = old.energy_GeV;
    result.timeS_ = old.time_s;
    result.weight_ = old.weight;
    result.historyId_ = old.history_id;
    result.parentHistoryId_ = old.parent_history_id;
    result.stepId_ = old.step_id;
    for (int axis = 0; axis < 3; ++axis) {
      result.positionM_[axis] = old.position_m[axis];
      result.direction_[axis] = old.direction[axis];
    }
    return result;
  }
  KOKKOS_INLINE_FUNCTION MagneticStepLimitResult
  maximumUniformMagneticStep(EmParticleState const& p, double mass, double charge,
                             double const* field, double deflection) {
    return uniform_magnetic_reference::maximumUniformMagneticStep(
        particle(p), mass, charge, field, deflection);
  }
  struct Advance {
    MagneticStepStatus status;
    EmParticleState particle;
    double bend_parameter, chord_length_m;
  };
  KOKKOS_INLINE_FUNCTION Advance advanceUniformMagneticField(EmParticleState const& p,
                                                             double mass, double charge,
                                                             double const* field,
                                                             double length) {
    auto out = uniform_magnetic_reference::advanceUniformMagneticField(
        particle(p), mass, charge, field, length);
    return {out.status, continuous_test::convert(out.particle), out.bend_parameter,
            out.chord_length_m};
  }
  KOKKOS_INLINE_FUNCTION corsika::kokkos::MagneticStepLimitResult limit(
      MagneticStepLimitResult const& old) {
    return {static_cast<corsika::kokkos::MagneticStepStatus>(old.status), old.reserved,
            old.distance_m, old.gyroradius_m};
  }
  KOKKOS_INLINE_FUNCTION straight_boundary_reference::Tree tree(
      query_reference::FullView const& view) {
    return {view.tree_.nodes_, view.tree_.shapes_, view.tree_.children_,
            view.tree_.exclusions_};
  }
  namespace environment_tree {
    struct Boundary {
      double distance_m;
      int next_node;
      bool failed;
    };
    KOKKOS_INLINE_FUNCTION Boundary leptonBoundary(query_reference::FullView const& view,
                                                   EmParticleState const& p, double mass,
                                                   double charge,
                                                   double const (&field)[3],
                                                   MagneticStepLimitResult const& l) {
      auto out = lepton_boundary_reference::lepton_boundary(tree(view), particle(p), mass,
                                                            charge, field, limit(l));
      return {out.distanceM_, out.nextNode_, out.failed_};
    }
    KOKKOS_INLINE_FUNCTION double leptonObservationDistance(
        EnvironmentSnapshot const& e, EmParticleState const& p, double mass,
        double charge, double const (&field)[3], MagneticStepLimitResult const& l) {
      return lepton_boundary_reference::lepton_observation_distance(
          tree(e.node_tree), particle(p), mass, charge, field, limit(l), e.point,
          e.normal);
    }
  } // namespace environment_tree
  inline constexpr std::uint32_t NativeMuonPairCutFlag = 1U;
  KOKKOS_INLINE_FUNCTION bool isNativeMuonPairCut(EmParticleState const& p) {
    return p.reserved == NativeMuonPairCutFlag && isMuonPid(p.pid) && p.generation > 0 &&
           p.step_id == 0 && p.energy_GeV > 0. && p.energy_GeV <= transportMassGeV(p.pid);
  }
  KOKKOS_INLINE_FUNCTION void prepareStoppedAnnihilation(LeptonTransportRecord& record) {
    if constexpr (!EnableStoppedPositronAnnihilation) return;
    if (record.end.pid != static_cast<std::int32_t>(EmPid::Positron) ||
        exceedsParticleCutTime(record.end.time_s))
      return;
    record.interaction.status = EmInteractionStatus::AtRestAnnihilation;
    record.interaction.process_id = AnnihilationProcessId;
    record.interaction.interaction_vertex_reached = 1;
    record.interaction.particle = record.end;
    record.interaction.particle.energy_GeV = transportMassGeV(record.end.pid);
    record.interaction.particle_mass_GeV = transportMassGeV(record.end.pid);
  }
  KOKKOS_INLINE_FUNCTION std::uint32_t transportLepton(
      tables::NativePhysicsView const& table, bool const apply_moliere,
      EnvironmentSnapshot const& environment, EmInteractionRecord const& interaction,
      LeptonTransportRecord& record, ProposalFallbackEvent& fallback,
      ExternalTransportBoundary const external = {}) {
    auto start = interaction.particle;
    if (!isChargedLeptonPid(start.pid)) {
      fallback = makeLeptonTransportFallback(interaction,
                                             ProposalFallbackReason::UnsupportedParticle);
      return 1;
    }
    if (interaction.status != EmInteractionStatus::Selected &&
        interaction.status != EmInteractionStatus::DistanceSampled &&
        interaction.status != EmInteractionStatus::NoDiscreteInteraction &&
        interaction.status != EmInteractionStatus::ParticleCut) {
      fallback = makeLeptonTransportFallback(interaction,
                                             ProposalFallbackReason::InvalidFinalState);
      return 1;
    }
    if ((interaction.status == EmInteractionStatus::Selected ||
         interaction.status == EmInteractionStatus::DistanceSampled) &&
        (!leptonContinuousFinite(interaction.interaction_grammage_g_per_cm2) ||
         interaction.interaction_grammage_g_per_cm2 < 0.)) {
      fallback = makeLeptonTransportFallback(interaction,
                                             ProposalFallbackReason::InvalidFinalState);
      return 1;
    }

    if (interaction.status == EmInteractionStatus::ParticleCut &&
        isNativeMuonPairCut(start)) {
      record = nativeMuonPairCutRecord(interaction);
      // Exactly the scalar ParticleCut signed kinetic-energy accounting.
      // No magnetic/Moliere calculation and no finite radio trajectory.
      return apply_moliere ? 2U : 0U;
    }
    auto const continuous = prepareLeptonContinuousStep(table, interaction);
    if (continuous.fallback_flag != 0) {
      fallback = continuous.fallback;
      return 1;
    }
    auto const initial_energy_MeV = start.energy_GeV * 1000.;
    auto const mass_MeV = continuous.mass_MeV;
    auto const transport_cut_MeV = continuous.transport_cut_MeV;

    if (continuous.particle_cut != 0) {
      auto const layer = queryAtmosphereLayer(
          environment, start.position_m, start.direction,
          AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
      if (layer.status != AtmosphereStatus::Success) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::UnsupportedGeometry);
        return 1;
      }

      start.environment_node = layer.layer_index;

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
      record.start_density_g_per_cm3 = layer.density_g_per_cm3;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
      record.cut_deposited_energy_GeV = (initial_energy_MeV - mass_MeV) / 1000.;
      if (initial_energy_MeV - mass_MeV < transport_cut_MeV)
        prepareStoppedAnnihilation(record);
      // State 2 is a successful zero-length cut that must bypass the
      // split Moliere kernel. That kernel normalizes it to state 0 before
      // fallback counting.
      return apply_moliere ? 2U : 0U;
    }

    auto const continuous_grammage = continuous.maximum_grammage_g_per_cm2;

    auto const layer = queryAtmosphereLayer(
        environment, start.position_m, start.direction,
        AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
    if (layer.status != AtmosphereStatus::Success) {
      fallback = makeLeptonTransportFallback(interaction,
                                             ProposalFallbackReason::UnsupportedGeometry);
      return 1;
    }
    auto const charge_number = start.pid > 0 ? -1. : 1.;
    start.environment_node = layer.layer_index;
    auto const& field_T =
        environment.node_tree.model(start.environment_node)->magneticFieldT_;
    auto const magnetic_limit =
        maximumUniformMagneticStep(start, mass_MeV / 1000., charge_number, field_T,
                                   environment.node_tree.maximumDeflection_);
    if (magnetic_limit.status == MagneticStepStatus::InvalidInput ||
        magnetic_limit.status == MagneticStepStatus::NonFiniteResult) {
      fallback = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::MagneticTransportFailed);
      return 1;
    }

    auto geometry_distance_m = HUGE_VAL;
    auto limiting_radius_m = 0.;
    auto magnetic_step_wins = false;
    auto observation_plane_wins = false;
    auto next_node = start.environment_node;
    {
      auto const boundary =
          environment_tree::leptonBoundary(environment.node_tree, start, mass_MeV / 1000.,
                                           charge_number, field_T, magnetic_limit);
      if (boundary.failed) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::MagneticBoundaryFailed);
        return 1;
      }
      geometry_distance_m = boundary.distance_m;
      next_node = boundary.next_node;
      magnetic_step_wins = next_node == start.environment_node;
      // Observation is a separate process, not a material-tree node.
      if (!external.disable_observation) {
        auto const observation = environment_tree::leptonObservationDistance(
            environment, start, mass_MeV / 1000., charge_number, field_T, magnetic_limit);
        if (observation < geometry_distance_m) {
          geometry_distance_m = observation;
          observation_plane_wins = true;
        }
        if (observation_plane_wins) magnetic_step_wins = false;
      }
    }
    auto const material_boundary_wins = external.enabled && external.distance_m > 0. &&
                                        external.distance_m < geometry_distance_m;
    if (material_boundary_wins) {
      // External distance uses the same leapfrog length, not a chord length.
      // This material interface is never an absorbing detector.
      geometry_distance_m = external.distance_m;
      observation_plane_wins = false;
      magnetic_step_wins = false;
    }
    auto const geometry_grammage =
        geometryCompetitionGrammage(environment, layer.layer_index, start.position_m,
                                    start.direction, geometry_distance_m, external);
    if (geometry_grammage.status != AtmosphereStatus::Success) {
      auto event = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
      event.diagnostic_status = static_cast<std::int32_t>(geometry_grammage.status);
      event.diagnostic_value0 = geometry_distance_m;
      event.diagnostic_value1 = layer.density_g_per_cm3;
      event.diagnostic_value2 = limiting_radius_m;
      fallback = event;
      return 1;
    }

    auto const has_interaction =
        interaction.status == EmInteractionStatus::Selected ||
        interaction.status == EmInteractionStatus::DistanceSampled;
    auto const interaction_grammage =
        has_interaction ? interaction.interaction_grammage_g_per_cm2 : HUGE_VAL;
    auto continuous_wins = continuous_grammage < geometry_grammage.value &&
                           continuous_grammage < interaction_grammage;
    auto interaction_wins = !continuous_wins && has_interaction &&
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
    record.start_density_g_per_cm3 = layer.density_g_per_cm3;
    record.continuous_step_grammage_g_per_cm2 = continuous_grammage;
    record.limiting_radius_m = limiting_radius_m;
    record.magnetic_step_status = static_cast<std::uint32_t>(magnetic_limit.status);
    record.magnetic_step_limit_m = magnetic_limit.distance_m;
    record.magnetic_gyroradius_m = magnetic_limit.gyroradius_m;

    double traversed_grammage = geometry_grammage.value;
    double distance_m = geometry_distance_m;
    if (continuous_wins) {
      traversed_grammage = continuous_grammage;
    } else if (interaction_wins) {
      traversed_grammage = interaction_grammage;
    }
    if (continuous_wins || interaction_wins) {
      auto const distance =
          atmosphereDistanceFromGrammage(environment, layer.layer_index, start.position_m,
                                         start.direction, traversed_grammage);
      if (distance.status != AtmosphereStatus::Success ||
          distance.value > geometry_distance_m * (1. + 1.e-12)) {
        auto event = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereInverseGrammageFailed);
        event.diagnostic_status = static_cast<std::int32_t>(distance.status);
        event.diagnostic_value0 = distance.value;
        event.diagnostic_value1 = geometry_distance_m;
        event.diagnostic_value2 = traversed_grammage;
        fallback = event;
        return 1;
      }
      distance_m = distance.value;
    }
    auto const has_decay = isMuonPid(start.pid) &&
                           leptonContinuousFinite(interaction.decay_distance_m) &&
                           interaction.decay_distance_m >= 0.;
    auto const decay_wins = has_decay && interaction.decay_distance_m <= distance_m;
    if (decay_wins) {
      continuous_wins = false;
      interaction_wins = false;
      distance_m = interaction.decay_distance_m;
      auto const decay_grammage = atmosphereGrammage(
          environment, layer.layer_index, start.position_m, start.direction, distance_m);
      if (decay_grammage.status != AtmosphereStatus::Success) {
        auto event = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
        event.diagnostic_status = static_cast<std::int32_t>(decay_grammage.status);
        event.diagnostic_value0 = distance_m;
        fallback = event;
        return 1;
      }
      traversed_grammage = decay_grammage.value;
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
    auto const* propagation_field_T =
        magnetic_limit.status == MagneticStepStatus::Linear ? zero_field_T : field_T;
    auto const magnetic_advance = advanceUniformMagneticField(
        start, mass_MeV / 1000., charge_number, propagation_field_T, distance_m);
    if (magnetic_advance.status == MagneticStepStatus::InvalidInput ||
        magnetic_advance.status == MagneticStepStatus::NonFiniteResult) {
      fallback = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::MagneticTransportFailed);
      return 1;
    }
    record.end = magnetic_advance.particle;
    if (observation_reached) {
      double radial[3]{};
      record.limiting_radius_m =
          atmosphere_detail::radiusVector(environment, record.end.position_m, radial);
    }
    record.magnetic_bending_applied =
        magnetic_advance.status == MagneticStepStatus::Success &&
                magnetic_advance.bend_parameter != 0.
            ? 1U
            : 0U;
    record.magnetic_bend_parameter = magnetic_advance.bend_parameter;
    record.magnetic_chord_length_m = magnetic_advance.chord_length_m;
    record.distance_m = distance_m;

    // Scalar CORSIKA uses the original leapfrog trajectory when it turns a
    // sampled grammage into the competing step length.  Once the endpoint
    // has been selected, however, Step retains only the displacement and
    // ContinuousProcess integrates the medium along
    // Step::getStraightTrack(): the chord between the two endpoints.  Keep
    // the start-tangent grammage above for distance competition, then match
    // the scalar continuous-process semantics here after magnetic advance.
    double chord_direction[3]{start.direction[0], start.direction[1], start.direction[2]};
    auto continuous_chord_length_squared = 0.;
    for (int axis = 0; axis < 3; ++axis) {
      auto const displacement = record.end.position_m[axis] - start.position_m[axis];
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
        chord_direction[axis] = start.direction[axis];
      }
    }
    auto const chord_grammage =
        atmosphereGrammage(environment, layer.layer_index, start.position_m,
                           chord_direction, continuous_chord_length_m);
    if (chord_grammage.status != AtmosphereStatus::Success) {
      auto event = makeLeptonTransportFallback(
          interaction, ProposalFallbackReason::AtmosphereGrammageFailed);
      event.diagnostic_status = static_cast<std::int32_t>(chord_grammage.status);
      event.diagnostic_value0 = continuous_chord_length_m;
      event.diagnostic_value1 = distance_m;
      event.diagnostic_value2 = traversed_grammage;
      fallback = event;
      return 1;
    }
    traversed_grammage = chord_grammage.value;
    record.traversed_grammage_g_per_cm2 = traversed_grammage;
    auto const loss = evaluateLeptonContinuousLoss(table, interaction, continuous,
                                                   traversed_grammage, continuous_wins);
    if (loss.fallback_flag != 0) {
      fallback = loss.fallback;
      return 1;
    }
    auto const final_energy_MeV = loss.final_energy_MeV;
    record.end.energy_GeV = final_energy_MeV / 1000.;
    record.continuous_deposited_energy_GeV = loss.deposited_energy_GeV;

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
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
      record.cut_deposited_energy_GeV = (final_energy_MeV - mass_MeV) / 1000.;
      record.observation_surface_reached_before_cut = observation_reached ? 1U : 0U;
      return 0;
    }

    auto const reaches_cut = loss.transport_cut_reached != 0 ||
                             final_energy_MeV - mass_MeV < transport_cut_MeV;
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
            environment, record.end.position_m, record.end.direction,
            AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
        if (end_layer.status != AtmosphereStatus::Success) {
          fallback = makeLeptonTransportFallback(
              interaction, ProposalFallbackReason::UnsupportedGeometry);
          return 1;
        }
        record.end_layer_index = end_layer.layer_index;
        record.end_density_g_per_cm3 = end_layer.density_g_per_cm3;
      }
      record.cut_deposited_energy_GeV = (final_energy_MeV - mass_MeV) / 1000.;
      record.observation_surface_reached_before_cut = observation_reached ? 1U : 0U;
      prepareStoppedAnnihilation(record);
      return 0;
    }

    if (decay_wins) {
      auto const vertex = queryAtmosphereLayer(
          environment, record.end.position_m, record.end.direction,
          AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
      if (vertex.status != AtmosphereStatus::Success) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereVertexLookupFailed);
        return 1;
      }
      record.limit = LeptonTransportLimit::DecayCandidate;
      record.end_layer_index = vertex.layer_index;
      record.end_density_g_per_cm3 = vertex.density_g_per_cm3;
      record.interaction.particle = record.end;
      record.interaction.status = EmInteractionStatus::Selected;
      record.interaction.interaction_vertex_reached = 1;
      record.interaction.process_id = DecayProcessId;
      record.interaction.component_hash = 0;
      record.interaction.energy_fraction = 0.;
      record.interaction.loss_quantile = 0.;
      return 0;
    }

    if (interaction_wins) {
      auto const vertex = queryAtmosphereLayer(
          environment, record.end.position_m, record.end.direction,
          AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
      if (vertex.status != AtmosphereStatus::Success ||
          !leptonContinuousFinite(vertex.density_g_per_cm3) ||
          !(vertex.density_g_per_cm3 > 0.)) {
        auto event = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::AtmosphereVertexLookupFailed);
        event.diagnostic_status = static_cast<std::int32_t>(vertex.status);
        event.diagnostic_value0 = vertex.radius_m;
        event.diagnostic_value1 = vertex.density_g_per_cm3;
        event.diagnostic_value2 = distance_m;
        fallback = event;
        return 1;
      }
      record.limit = LeptonTransportLimit::InteractionCandidate;
      record.end_layer_index = vertex.layer_index;
      record.end_density_g_per_cm3 = vertex.density_g_per_cm3;
      record.interaction.particle = record.end;
      record.interaction.status = EmInteractionStatus::RequiresReselection;
      record.interaction.interaction_vertex_reached = 1;
      record.interaction.process_id = 0;
      record.interaction.component_hash = 0;
      record.interaction.energy_fraction = 0.;
      record.interaction.loss_quantile = 0.;
      record.interaction.mass_density_g_per_cm3 = vertex.density_g_per_cm3;
      return 0;
    }

    record.end.step_id++;
    if (continuous_wins) {
      auto const end_layer = queryAtmosphereLayer(
          environment, record.end.position_m, record.end.direction,
          AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
      if (end_layer.status != AtmosphereStatus::Success) {
        fallback = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::UnsupportedGeometry);
        return 1;
      }
      record.limit = LeptonTransportLimit::ContinuousStep;
      record.end_layer_index = end_layer.layer_index;
      record.end_density_g_per_cm3 = end_layer.density_g_per_cm3;
      return 0;
    }

    if (magnetic_step_wins) {
      auto const end_layer = queryAtmosphereLayer(
          environment, record.end.position_m, record.end.direction,
          AtmosphereBoundaryPolicy::RadialGuard, start.environment_node);
      if (end_layer.status != AtmosphereStatus::Success) {
        auto event = makeLeptonTransportFallback(
            interaction, ProposalFallbackReason::MagneticBoundaryFailed);
        event.diagnostic_status = static_cast<std::int32_t>(end_layer.status);
        event.diagnostic_value0 = end_layer.radius_m;
        event.diagnostic_value1 = magnetic_limit.distance_m;
        fallback = event;
        return 1;
      }
      if (end_layer.layer_index != layer.layer_index) { return 99U; }
      record.limit = LeptonTransportLimit::MagneticStep;
      record.end_layer_index = end_layer.layer_index;
      record.end_density_g_per_cm3 = end_layer.density_g_per_cm3;
      return 0;
    }

    if (material_boundary_wins) {
      record.limit = LeptonTransportLimit::MaterialBoundary;
      record.end_layer_index = layer.layer_index;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
    } else if (observation_reached) {
      record.limit = LeptonTransportLimit::ObservationSurface;
      record.end_layer_index = -1;
      record.end_density_g_per_cm3 = layer.density_g_per_cm3;
    } else {
      record.end.environment_node = next_node;
      auto const* model = environment.node_tree.model(next_node);
      if (!model) {
        record.limit = LeptonTransportLimit::EscapedEnvironment;
        record.end_layer_index = next_node;
        record.end_density_g_per_cm3 = 0.;
      } else {
        auto const next =
            queryAtmosphereLayer(environment, record.end.position_m, record.end.direction,
                                 AtmosphereBoundaryPolicy::RadialGuard, next_node);
        if (next.status != AtmosphereStatus::Success ||
            next_node == start.environment_node) {
          fallback = makeLeptonTransportFallback(
              interaction, ProposalFallbackReason::UnsupportedGeometry);
          return 1;
        }
        record.limit = LeptonTransportLimit::LayerBoundary;
        record.end_layer_index = next_node;
        record.end.medium_id = model->mediumId_;
        record.end_density_g_per_cm3 = next.density_g_per_cm3;
      }
    }
    return 0;
  }
} // namespace lepton_transport_reference
