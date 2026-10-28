/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "TransportReference.hpp"
#include "../stochastic/StochasticChecks.hpp"
#include <corsika/kokkos/em/PhotonTransportRecord.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCutRecord.hpp>
#include <cstddef>
namespace transport_test {
  namespace current = corsika::kokkos;
  namespace reference = transport_reference;
  using continuous_test::append;
  using continuous_test::Outcome;
  using stochastic_test::check;
  using stochastic_test::execution_space_type;
  using stochastic_test::memory_space_type;
  using stochastic_test::upload;
  static_assert(sizeof(current::PhotonTransportRecord) ==
                sizeof(reference::PhotonTransportRecord));
  static_assert(alignof(current::PhotonTransportRecord) ==
                alignof(reference::PhotonTransportRecord));
  static_assert(offsetof(current::PhotonTransportRecord, interaction_) ==
                offsetof(reference::PhotonTransportRecord, interaction));
  static_assert(offsetof(current::PhotonTransportRecord, start_) ==
                offsetof(reference::PhotonTransportRecord, start));
  static_assert(offsetof(current::PhotonTransportRecord, end_) ==
                offsetof(reference::PhotonTransportRecord, end));
  static_assert(offsetof(current::PhotonTransportRecord, inputIndex_) ==
                offsetof(reference::PhotonTransportRecord, input_index));
  static_assert(offsetof(current::PhotonTransportRecord, limit_) ==
                offsetof(reference::PhotonTransportRecord, limit));
  static_assert(offsetof(current::PhotonTransportRecord, startLayerIndex_) ==
                offsetof(reference::PhotonTransportRecord, start_layer_index));
  static_assert(offsetof(current::PhotonTransportRecord, endLayerIndex_) ==
                offsetof(reference::PhotonTransportRecord, end_layer_index));
  static_assert(offsetof(current::PhotonTransportRecord, distanceM_) ==
                offsetof(reference::PhotonTransportRecord, distance_m));
  static_assert(offsetof(current::PhotonTransportRecord, traversedGrammageGPerCm2_) ==
                offsetof(reference::PhotonTransportRecord, traversed_grammage_g_per_cm2));
  static_assert(offsetof(current::PhotonTransportRecord, startDensityGPerCm3_) ==
                offsetof(reference::PhotonTransportRecord, start_density_g_per_cm3));
  static_assert(offsetof(current::PhotonTransportRecord, endDensityGPerCm3_) ==
                offsetof(reference::PhotonTransportRecord, end_density_g_per_cm3));
  static_assert(offsetof(current::PhotonTransportRecord, limitingRadiusM_) ==
                offsetof(reference::PhotonTransportRecord, limiting_radius_m));
  static_assert(offsetof(current::PhotonTransportRecord, cutDepositedEnergyGeV_) ==
                offsetof(reference::PhotonTransportRecord, cut_deposited_energy_GeV));
  static_assert(offsetof(current::PhotonTransportRecord,
                         observationSurfaceReachedBeforeCut_) ==
                offsetof(reference::PhotonTransportRecord,
                         observation_surface_reached_before_cut));
  CORSIKA_HOST_DEVICE inline Outcome flatten(
      current::PhotonTransportRecord const& value) {
    Outcome result;
    append(result, value.interaction_);
    append(result, value.start_);
    append(result, value.end_);
    append(result, value.inputIndex_);
    append(result, value.limit_);
    append(result, value.startLayerIndex_);
    append(result, value.endLayerIndex_);
    append(result, value.distanceM_);
    append(result, value.traversedGrammageGPerCm2_);
    append(result, value.startDensityGPerCm3_);
    append(result, value.endDensityGPerCm3_);
    append(result, value.limitingRadiusM_);
    append(result, value.cutDepositedEnergyGeV_);
    append(result, value.observationSurfaceReachedBeforeCut_);
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome flatten(
      reference::PhotonTransportRecord const& value) {
    Outcome result;
    append(result, value.interaction);
    append(result, value.start);
    append(result, value.end);
    append(result, value.input_index);
    append(result, value.limit);
    append(result, value.start_layer_index);
    append(result, value.end_layer_index);
    append(result, value.distance_m);
    append(result, value.traversed_grammage_g_per_cm2);
    append(result, value.start_density_g_per_cm3);
    append(result, value.end_density_g_per_cm3);
    append(result, value.limiting_radius_m);
    append(result, value.cut_deposited_energy_GeV);
    append(result, value.observation_surface_reached_before_cut);
    return result;
  }
  static_assert(static_cast<int>(current::PhotonTransportLimit::Interaction) == 0);
  static_assert(static_cast<int>(current::PhotonTransportLimit::LayerBoundary) == 1);
  static_assert(static_cast<int>(current::PhotonTransportLimit::ObservationSurface) == 2);
  static_assert(static_cast<int>(current::PhotonTransportLimit::EscapedEnvironment) == 3);
  static_assert(static_cast<int>(current::PhotonTransportLimit::ParticleCut) == 4);
  static_assert(static_cast<int>(current::PhotonTransportLimit::MaterialBoundary) == 5);
  CORSIKA_HOST_DEVICE inline reference::PhotonTransportRecord convert(
      current::PhotonTransportRecord const& value) {
    reference::PhotonTransportRecord result;
    result.interaction = continuous_test::convert(value.interaction_);
    result.start = continuous_test::convert(value.start_);
    result.end = continuous_test::convert(value.end_);
    result.input_index = value.inputIndex_;
    result.limit = static_cast<reference::PhotonTransportLimit>(value.limit_);
    result.start_layer_index = value.startLayerIndex_;
    result.end_layer_index = value.endLayerIndex_;
    result.distance_m = value.distanceM_;
    result.traversed_grammage_g_per_cm2 = value.traversedGrammageGPerCm2_;
    result.start_density_g_per_cm3 = value.startDensityGPerCm3_;
    result.end_density_g_per_cm3 = value.endDensityGPerCm3_;
    result.limiting_radius_m = value.limitingRadiusM_;
    result.cut_deposited_energy_GeV = value.cutDepositedEnergyGeV_;
    result.observation_surface_reached_before_cut =
        value.observationSurfaceReachedBeforeCut_;
    return result;
  }
  inline current::PhotonTransportRecord make_photon_payload(std::size_t seed) {
    current::PhotonTransportRecord result;
    // Synthetic storage values, not a physical transport event.
    auto& interaction = result.interaction_;
    auto& particle = interaction.particle_;
    particle.pid_ = seed % 2 ? -13 : 22;
    particle.mediumId_ = 7;
    particle.environmentNode_ = -3 + static_cast<int>(seed % 12);
    particle.generation_ = 31;
    particle.reserved_ = 0xa5a5u;
    particle.energyGeV_ = 1.e3 + seed;
    particle.timeS_ = 1.e-6 * seed;
    particle.weight_ = .5 + seed;
    particle.historyId_ = 0x8000000000000000ULL + seed;
    particle.parentHistoryId_ = 0xfedcba9876543210ULL - seed;
    particle.stepId_ = 0x12345678abcdefULL + seed;
    for (int axis = 0; axis < 3; ++axis) {
      particle.positionM_[axis] = .25 * axis - 7. * seed;
      particle.direction_[axis] = axis == 2 ? -1. : 0.;
    }
    interaction.inputIndex_ = 0x8000000012345678ULL + seed;
    interaction.status_ = current::InteractionStatus::RequiresReselection;
    interaction.deferredPhotonFallbackReason_ = 28;
    interaction.deferredPhotonTableStatus_ = 4;
    interaction.interactionVertexReached_ = 1;
    interaction.processId_ = 987;
    interaction.componentHash_ = 0xfedcba9876543210ULL;
    interaction.totalRateCm2PerG_ = 2.;
    interaction.vertexTotalRateCm2PerG_ = 3.;
    interaction.interactionGrammageGPerCm2_ = 4.;
    interaction.massDensityGPerCm3_ = .0012;
    interaction.particleMassGeV_ = .1;
    interaction.decayDistanceM_ = 500.;
    interaction.decayUniform_ = .2;
    interaction.decayDrawId_ = 19;
    interaction.energyFraction_ = .4;
    interaction.distanceUniform_ = .1;
    interaction.processUniform_ = .3;
    interaction.lossQuantile_ = .6;
    interaction.distanceDrawId_ = 31;
    interaction.processDrawId_ = 37;
    interaction.lossDrawId_ = 41;
    interaction.proposalSelectionUniform_ = .7;
    interaction.processRandomProcessId_ = 43;
    interaction.proposalSelectionRandomProcessId_ = 47;
    interaction.proposalSelectionDrawId_ = 53;
    result.start_ = particle;
    result.end_ = particle;
    result.end_.energyGeV_ -= 1.;
    result.end_.positionM_[2] += 2.;
    result.end_.direction_[1] = .3;
    result.end_.timeS_ += 3.e-9;
    result.end_.stepId_ += 1;
    result.end_.environmentNode_ = 101;
    result.inputIndex_ = static_cast<std::uint64_t>(3 + seed + 1);
    result.limit_ = static_cast<current::PhotonTransportLimit>(seed % 6);
    result.startLayerIndex_ = static_cast<std::int32_t>(5 + seed + 1);
    result.endLayerIndex_ = static_cast<std::int32_t>(6 + seed + 1);
    result.distanceM_ = (seed % 2 ? -1. : 1.) * (7.25 + seed);
    result.traversedGrammageGPerCm2_ = (seed % 2 ? -1. : 1.) * (8.25 + seed);
    result.startDensityGPerCm3_ = (seed % 2 ? -1. : 1.) * (9.25 + seed);
    result.endDensityGPerCm3_ = (seed % 2 ? -1. : 1.) * (10.25 + seed);
    result.limitingRadiusM_ = (seed % 2 ? -1. : 1.) * (11.25 + seed);
    result.cutDepositedEnergyGeV_ = (seed % 2 ? -1. : 1.) * (12.25 + seed);
    result.observationSurfaceReachedBeforeCut_ =
        static_cast<std::uint32_t>(13 + seed + 1);
    return result;
  }
  static_assert(sizeof(current::LeptonTransportRecord) ==
                sizeof(reference::LeptonTransportRecord));
  static_assert(alignof(current::LeptonTransportRecord) ==
                alignof(reference::LeptonTransportRecord));
  static_assert(offsetof(current::LeptonTransportRecord, interaction_) ==
                offsetof(reference::LeptonTransportRecord, interaction));
  static_assert(offsetof(current::LeptonTransportRecord, start_) ==
                offsetof(reference::LeptonTransportRecord, start));
  static_assert(offsetof(current::LeptonTransportRecord, end_) ==
                offsetof(reference::LeptonTransportRecord, end));
  static_assert(offsetof(current::LeptonTransportRecord, inputIndex_) ==
                offsetof(reference::LeptonTransportRecord, input_index));
  static_assert(offsetof(current::LeptonTransportRecord, limit_) ==
                offsetof(reference::LeptonTransportRecord, limit));
  static_assert(offsetof(current::LeptonTransportRecord, startLayerIndex_) ==
                offsetof(reference::LeptonTransportRecord, start_layer_index));
  static_assert(offsetof(current::LeptonTransportRecord, endLayerIndex_) ==
                offsetof(reference::LeptonTransportRecord, end_layer_index));
  static_assert(offsetof(current::LeptonTransportRecord, distanceM_) ==
                offsetof(reference::LeptonTransportRecord, distance_m));
  static_assert(offsetof(current::LeptonTransportRecord, traversedGrammageGPerCm2_) ==
                offsetof(reference::LeptonTransportRecord, traversed_grammage_g_per_cm2));
  static_assert(offsetof(current::LeptonTransportRecord,
                         continuousStepGrammageGPerCm2_) ==
                offsetof(reference::LeptonTransportRecord,
                         continuous_step_grammage_g_per_cm2));
  static_assert(offsetof(current::LeptonTransportRecord, startDensityGPerCm3_) ==
                offsetof(reference::LeptonTransportRecord, start_density_g_per_cm3));
  static_assert(offsetof(current::LeptonTransportRecord, endDensityGPerCm3_) ==
                offsetof(reference::LeptonTransportRecord, end_density_g_per_cm3));
  static_assert(offsetof(current::LeptonTransportRecord, limitingRadiusM_) ==
                offsetof(reference::LeptonTransportRecord, limiting_radius_m));
  static_assert(offsetof(current::LeptonTransportRecord, continuousDepositedEnergyGeV_) ==
                offsetof(reference::LeptonTransportRecord,
                         continuous_deposited_energy_GeV));
  static_assert(offsetof(current::LeptonTransportRecord, cutDepositedEnergyGeV_) ==
                offsetof(reference::LeptonTransportRecord, cut_deposited_energy_GeV));
  static_assert(offsetof(current::LeptonTransportRecord,
                         observationSurfaceReachedBeforeCut_) ==
                offsetof(reference::LeptonTransportRecord,
                         observation_surface_reached_before_cut));
  static_assert(offsetof(current::LeptonTransportRecord, multipleScatteringApplied_) ==
                offsetof(reference::LeptonTransportRecord, multiple_scattering_applied));
  static_assert(offsetof(current::LeptonTransportRecord, multipleScatteringStatus_) ==
                offsetof(reference::LeptonTransportRecord, multiple_scattering_status));
  static_assert(offsetof(current::LeptonTransportRecord, multipleScatteringIterations_) ==
                offsetof(reference::LeptonTransportRecord,
                         multiple_scattering_iterations));
  static_assert(offsetof(current::LeptonTransportRecord, multipleScatteringAngleRad_) ==
                offsetof(reference::LeptonTransportRecord,
                         multiple_scattering_angle_rad));
  static_assert(offsetof(current::LeptonTransportRecord,
                         multipleScatteringFirstUniform_) ==
                offsetof(reference::LeptonTransportRecord,
                         multiple_scattering_first_uniform));
  static_assert(offsetof(current::LeptonTransportRecord,
                         multipleScatteringSecondUniform_) ==
                offsetof(reference::LeptonTransportRecord,
                         multiple_scattering_second_uniform));
  static_assert(offsetof(current::LeptonTransportRecord,
                         multipleScatteringAzimuthUniform_) ==
                offsetof(reference::LeptonTransportRecord,
                         multiple_scattering_azimuth_uniform));
  static_assert(offsetof(current::LeptonTransportRecord, magneticBendingApplied_) ==
                offsetof(reference::LeptonTransportRecord, magnetic_bending_applied));
  static_assert(offsetof(current::LeptonTransportRecord, magneticStepStatus_) ==
                offsetof(reference::LeptonTransportRecord, magnetic_step_status));
  static_assert(offsetof(current::LeptonTransportRecord, magneticStepLimitM_) ==
                offsetof(reference::LeptonTransportRecord, magnetic_step_limit_m));
  static_assert(offsetof(current::LeptonTransportRecord, magneticGyroradiusM_) ==
                offsetof(reference::LeptonTransportRecord, magnetic_gyroradius_m));
  static_assert(offsetof(current::LeptonTransportRecord, magneticBendParameter_) ==
                offsetof(reference::LeptonTransportRecord, magnetic_bend_parameter));
  static_assert(offsetof(current::LeptonTransportRecord, magneticChordLengthM_) ==
                offsetof(reference::LeptonTransportRecord, magnetic_chord_length_m));
  CORSIKA_HOST_DEVICE inline Outcome flatten(
      current::LeptonTransportRecord const& value) {
    Outcome result;
    append(result, value.interaction_);
    append(result, value.start_);
    append(result, value.end_);
    append(result, value.inputIndex_);
    append(result, value.limit_);
    append(result, value.startLayerIndex_);
    append(result, value.endLayerIndex_);
    append(result, value.distanceM_);
    append(result, value.traversedGrammageGPerCm2_);
    append(result, value.continuousStepGrammageGPerCm2_);
    append(result, value.startDensityGPerCm3_);
    append(result, value.endDensityGPerCm3_);
    append(result, value.limitingRadiusM_);
    append(result, value.continuousDepositedEnergyGeV_);
    append(result, value.cutDepositedEnergyGeV_);
    append(result, value.observationSurfaceReachedBeforeCut_);
    append(result, value.multipleScatteringApplied_);
    append(result, value.multipleScatteringStatus_);
    append(result, value.multipleScatteringIterations_);
    append(result, value.multipleScatteringAngleRad_);
    append(result, value.multipleScatteringFirstUniform_);
    append(result, value.multipleScatteringSecondUniform_);
    append(result, value.multipleScatteringAzimuthUniform_);
    append(result, value.magneticBendingApplied_);
    append(result, value.magneticStepStatus_);
    append(result, value.magneticStepLimitM_);
    append(result, value.magneticGyroradiusM_);
    append(result, value.magneticBendParameter_);
    append(result, value.magneticChordLengthM_);
    return result;
  }
  CORSIKA_HOST_DEVICE inline Outcome flatten(
      reference::LeptonTransportRecord const& value) {
    Outcome result;
    append(result, value.interaction);
    append(result, value.start);
    append(result, value.end);
    append(result, value.input_index);
    append(result, value.limit);
    append(result, value.start_layer_index);
    append(result, value.end_layer_index);
    append(result, value.distance_m);
    append(result, value.traversed_grammage_g_per_cm2);
    append(result, value.continuous_step_grammage_g_per_cm2);
    append(result, value.start_density_g_per_cm3);
    append(result, value.end_density_g_per_cm3);
    append(result, value.limiting_radius_m);
    append(result, value.continuous_deposited_energy_GeV);
    append(result, value.cut_deposited_energy_GeV);
    append(result, value.observation_surface_reached_before_cut);
    append(result, value.multiple_scattering_applied);
    append(result, value.multiple_scattering_status);
    append(result, value.multiple_scattering_iterations);
    append(result, value.multiple_scattering_angle_rad);
    append(result, value.multiple_scattering_first_uniform);
    append(result, value.multiple_scattering_second_uniform);
    append(result, value.multiple_scattering_azimuth_uniform);
    append(result, value.magnetic_bending_applied);
    append(result, value.magnetic_step_status);
    append(result, value.magnetic_step_limit_m);
    append(result, value.magnetic_gyroradius_m);
    append(result, value.magnetic_bend_parameter);
    append(result, value.magnetic_chord_length_m);
    return result;
  }
  static_assert(static_cast<int>(current::LeptonTransportLimit::InteractionCandidate) ==
                0);
  static_assert(static_cast<int>(current::LeptonTransportLimit::ContinuousStep) == 1);
  static_assert(static_cast<int>(current::LeptonTransportLimit::ParticleCut) == 2);
  static_assert(static_cast<int>(current::LeptonTransportLimit::LayerBoundary) == 3);
  static_assert(static_cast<int>(current::LeptonTransportLimit::ObservationSurface) == 4);
  static_assert(static_cast<int>(current::LeptonTransportLimit::EscapedEnvironment) == 5);
  static_assert(static_cast<int>(current::LeptonTransportLimit::MagneticStep) == 6);
  static_assert(static_cast<int>(current::LeptonTransportLimit::DecayCandidate) == 7);
  static_assert(static_cast<int>(current::LeptonTransportLimit::MaterialBoundary) == 8);
  CORSIKA_HOST_DEVICE inline reference::LeptonTransportRecord convert(
      current::LeptonTransportRecord const& value) {
    reference::LeptonTransportRecord result;
    result.interaction = continuous_test::convert(value.interaction_);
    result.start = continuous_test::convert(value.start_);
    result.end = continuous_test::convert(value.end_);
    result.input_index = value.inputIndex_;
    result.limit = static_cast<reference::LeptonTransportLimit>(value.limit_);
    result.start_layer_index = value.startLayerIndex_;
    result.end_layer_index = value.endLayerIndex_;
    result.distance_m = value.distanceM_;
    result.traversed_grammage_g_per_cm2 = value.traversedGrammageGPerCm2_;
    result.continuous_step_grammage_g_per_cm2 = value.continuousStepGrammageGPerCm2_;
    result.start_density_g_per_cm3 = value.startDensityGPerCm3_;
    result.end_density_g_per_cm3 = value.endDensityGPerCm3_;
    result.limiting_radius_m = value.limitingRadiusM_;
    result.continuous_deposited_energy_GeV = value.continuousDepositedEnergyGeV_;
    result.cut_deposited_energy_GeV = value.cutDepositedEnergyGeV_;
    result.observation_surface_reached_before_cut =
        value.observationSurfaceReachedBeforeCut_;
    result.multiple_scattering_applied = value.multipleScatteringApplied_;
    result.multiple_scattering_status = value.multipleScatteringStatus_;
    result.multiple_scattering_iterations = value.multipleScatteringIterations_;
    result.multiple_scattering_angle_rad = value.multipleScatteringAngleRad_;
    result.multiple_scattering_first_uniform = value.multipleScatteringFirstUniform_;
    result.multiple_scattering_second_uniform = value.multipleScatteringSecondUniform_;
    result.multiple_scattering_azimuth_uniform = value.multipleScatteringAzimuthUniform_;
    result.magnetic_bending_applied = value.magneticBendingApplied_;
    result.magnetic_step_status = value.magneticStepStatus_;
    result.magnetic_step_limit_m = value.magneticStepLimitM_;
    result.magnetic_gyroradius_m = value.magneticGyroradiusM_;
    result.magnetic_bend_parameter = value.magneticBendParameter_;
    result.magnetic_chord_length_m = value.magneticChordLengthM_;
    return result;
  }
  inline current::LeptonTransportRecord make_lepton_payload(std::size_t seed) {
    current::LeptonTransportRecord result;
    // Synthetic storage values, not a physical transport event.
    auto& interaction = result.interaction_;
    auto& particle = interaction.particle_;
    particle.pid_ = seed % 2 ? -13 : 22;
    particle.mediumId_ = 7;
    particle.environmentNode_ = -3 + static_cast<int>(seed % 12);
    particle.generation_ = 31;
    particle.reserved_ = 0xa5a5u;
    particle.energyGeV_ = 1.e3 + seed;
    particle.timeS_ = 1.e-6 * seed;
    particle.weight_ = .5 + seed;
    particle.historyId_ = 0x8000000000000000ULL + seed;
    particle.parentHistoryId_ = 0xfedcba9876543210ULL - seed;
    particle.stepId_ = 0x12345678abcdefULL + seed;
    for (int axis = 0; axis < 3; ++axis) {
      particle.positionM_[axis] = .25 * axis - 7. * seed;
      particle.direction_[axis] = axis == 2 ? -1. : 0.;
    }
    interaction.inputIndex_ = 0x8000000012345678ULL + seed;
    interaction.status_ = current::InteractionStatus::RequiresReselection;
    interaction.deferredPhotonFallbackReason_ = 28;
    interaction.deferredPhotonTableStatus_ = 4;
    interaction.interactionVertexReached_ = 1;
    interaction.processId_ = 987;
    interaction.componentHash_ = 0xfedcba9876543210ULL;
    interaction.totalRateCm2PerG_ = 2.;
    interaction.vertexTotalRateCm2PerG_ = 3.;
    interaction.interactionGrammageGPerCm2_ = 4.;
    interaction.massDensityGPerCm3_ = .0012;
    interaction.particleMassGeV_ = .1;
    interaction.decayDistanceM_ = 500.;
    interaction.decayUniform_ = .2;
    interaction.decayDrawId_ = 19;
    interaction.energyFraction_ = .4;
    interaction.distanceUniform_ = .1;
    interaction.processUniform_ = .3;
    interaction.lossQuantile_ = .6;
    interaction.distanceDrawId_ = 31;
    interaction.processDrawId_ = 37;
    interaction.lossDrawId_ = 41;
    interaction.proposalSelectionUniform_ = .7;
    interaction.processRandomProcessId_ = 43;
    interaction.proposalSelectionRandomProcessId_ = 47;
    interaction.proposalSelectionDrawId_ = 53;
    result.start_ = particle;
    result.end_ = particle;
    result.end_.energyGeV_ -= 1.;
    result.end_.positionM_[2] += 2.;
    result.end_.direction_[1] = .3;
    result.end_.timeS_ += 3.e-9;
    result.end_.stepId_ += 1;
    result.end_.environmentNode_ = 101;
    result.inputIndex_ = static_cast<std::uint64_t>(3 + seed + 1);
    result.limit_ = static_cast<current::LeptonTransportLimit>(seed % 9);
    result.startLayerIndex_ = static_cast<std::int32_t>(5 + seed + 1);
    result.endLayerIndex_ = static_cast<std::int32_t>(6 + seed + 1);
    result.distanceM_ = (seed % 2 ? -1. : 1.) * (7.25 + seed);
    result.traversedGrammageGPerCm2_ = (seed % 2 ? -1. : 1.) * (8.25 + seed);
    result.continuousStepGrammageGPerCm2_ = (seed % 2 ? -1. : 1.) * (9.25 + seed);
    result.startDensityGPerCm3_ = (seed % 2 ? -1. : 1.) * (10.25 + seed);
    result.endDensityGPerCm3_ = (seed % 2 ? -1. : 1.) * (11.25 + seed);
    result.limitingRadiusM_ = (seed % 2 ? -1. : 1.) * (12.25 + seed);
    result.continuousDepositedEnergyGeV_ = (seed % 2 ? -1. : 1.) * (13.25 + seed);
    result.cutDepositedEnergyGeV_ = (seed % 2 ? -1. : 1.) * (14.25 + seed);
    result.observationSurfaceReachedBeforeCut_ =
        static_cast<std::uint32_t>(15 + seed + 1);
    result.multipleScatteringApplied_ = static_cast<std::uint16_t>(16 + seed + 1);
    result.multipleScatteringStatus_ = static_cast<std::uint16_t>(17 + seed + 1);
    result.multipleScatteringIterations_ = static_cast<std::uint32_t>(18 + seed + 1);
    result.multipleScatteringAngleRad_ = (seed % 2 ? -1. : 1.) * (19.25 + seed);
    result.multipleScatteringFirstUniform_ = (seed % 2 ? -1. : 1.) * (20.25 + seed);
    result.multipleScatteringSecondUniform_ = (seed % 2 ? -1. : 1.) * (21.25 + seed);
    result.multipleScatteringAzimuthUniform_ = (seed % 2 ? -1. : 1.) * (22.25 + seed);
    result.magneticBendingApplied_ = static_cast<std::uint32_t>(23 + seed + 1);
    result.magneticStepStatus_ = static_cast<std::uint32_t>(24 + seed + 1);
    result.magneticStepLimitM_ = (seed % 2 ? -1. : 1.) * (25.25 + seed);
    result.magneticGyroradiusM_ = (seed % 2 ? -1. : 1.) * (26.25 + seed);
    result.magneticBendParameter_ = (seed % 2 ? -1. : 1.) * (27.25 + seed);
    result.magneticChordLengthM_ = (seed % 2 ? -1. : 1.) * (28.25 + seed);
    return result;
  }
  template <typename TRecord>
  void check_roundtrip(std::vector<TRecord> const& inputs) {
    auto data = upload(inputs);
    Kokkos::View<TRecord*, memory_space_type> copied("transport_copy", inputs.size());
    Kokkos::View<Outcome*, memory_space_type> inspected("transport_fields",
                                                        inputs.size());
    Kokkos::parallel_for(
        "transport_roundtrip",
        Kokkos::RangePolicy<execution_space_type>(0, inputs.size()),
        KOKKOS_LAMBDA(std::size_t index) {
          copied(index) = data(index);
          inspected(index) = flatten(data(index));
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, copied);
    auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, inspected);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      INFO("Transport storage case " << index);
      auto const expected = flatten(convert(inputs[index]));
      check(flatten(inputs[index]), expected);
      check(flatten(host(index)), expected);
      check(values(index), expected);
    }
  }
} // namespace transport_test
