/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Independent production records and function from 41cdc543.
#include "../continuous/ContinuousReference.hpp"
namespace transport_reference {
  using continuous_reference::EmInteractionRecord;
  using continuous_reference::EmInteractionStatus;
  using continuous_reference::EmParticleState;
  using continuous_reference::transportMassGeV;
  enum class PhotonTransportLimit : std::int32_t {
    Interaction = 0,
    LayerBoundary = 1,
    ObservationSurface = 2,
    EscapedEnvironment = 3,
    ParticleCut = 4,
    MaterialBoundary = 5,
  };

  struct PhotonTransportRecord {
    EmInteractionRecord interaction{};
    EmParticleState start{};
    EmParticleState end{};
    std::uint64_t input_index{};
    PhotonTransportLimit limit{PhotonTransportLimit::LayerBoundary};
    std::int32_t start_layer_index{-1};
    std::int32_t end_layer_index{-1};
    double distance_m{};
    double traversed_grammage_g_per_cm2{};
    double start_density_g_per_cm3{};
    double end_density_g_per_cm3{};
    double limiting_radius_m{};
    double cut_deposited_energy_GeV{};
    // ObservationPlane precedes ParticleCut in c8_air_shower's scalar
    // process sequence.  A step that reaches the plane after 10 ms must
    // therefore both publish the observation and terminate at ParticleCut.
    std::uint32_t observation_surface_reached_before_cut{};
  };

  enum class LeptonTransportLimit : std::int32_t {
    InteractionCandidate = 0,
    ContinuousStep = 1,
    ParticleCut = 2,
    LayerBoundary = 3,
    ObservationSurface = 4,
    EscapedEnvironment = 5,
    MagneticStep = 6,
    DecayCandidate = 7,
    MaterialBoundary = 8,
  };

  struct LeptonTransportRecord {
    EmInteractionRecord interaction{};
    EmParticleState start{};
    EmParticleState end{};
    std::uint64_t input_index{};
    LeptonTransportLimit limit{LeptonTransportLimit::ContinuousStep};
    std::int32_t start_layer_index{-1};
    std::int32_t end_layer_index{-1};
    double distance_m{};
    double traversed_grammage_g_per_cm2{};
    double continuous_step_grammage_g_per_cm2{};
    double start_density_g_per_cm3{};
    double end_density_g_per_cm3{};
    double limiting_radius_m{};
    double continuous_deposited_energy_GeV{};
    double cut_deposited_energy_GeV{};
    std::uint32_t observation_surface_reached_before_cut{};
    // Two 16-bit flags plus the iteration counter retain the former
    // eight-byte layout while exposing the dominant Moliere work per step.
    std::uint16_t multiple_scattering_applied{};
    std::uint16_t multiple_scattering_status{};
    std::uint32_t multiple_scattering_iterations{};
    double multiple_scattering_angle_rad{};
    double multiple_scattering_first_uniform{};
    double multiple_scattering_second_uniform{};
    double multiple_scattering_azimuth_uniform{};
    std::uint32_t magnetic_bending_applied{};
    std::uint32_t magnetic_step_status{};
    double magnetic_step_limit_m{};
    double magnetic_gyroradius_m{};
    double magnetic_bend_parameter{};
    double magnetic_chord_length_m{};
  };
  CORSIKA_HOST_DEVICE inline LeptonTransportRecord nativeMuonPairCutRecord(
      EmInteractionRecord const& interaction) {
    LeptonTransportRecord record{};
    record.interaction = interaction;
    record.interaction.status = EmInteractionStatus::ParticleCut;
    record.interaction.interaction_vertex_reached = 0;
    record.start = record.end = interaction.particle;
    ++record.end.step_id;
    record.input_index = interaction.input_index;
    record.limit = LeptonTransportLimit::ParticleCut;
    record.cut_deposited_energy_GeV =
        record.start.energy_GeV - transportMassGeV(record.start.pid);
    return record;
  }
} // namespace transport_reference
