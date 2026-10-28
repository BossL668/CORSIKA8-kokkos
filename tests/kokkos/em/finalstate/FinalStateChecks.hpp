/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Exhaustive ABI/default/populated-field checks, generated from production records.
#include "FinalStateReference.hpp"
#include <corsika/kokkos/em/FirstInteractionRecord.hpp>
#include <corsika/kokkos/em/PhotonFinalStateRecord.hpp>
#include <corsika/kokkos/em/LeptonFinalStateRecord.hpp>
#include <corsika/kokkos/em/PhotonPairLpmSuppressionRecord.hpp>
#include <corsika/kokkos/em/BremsLpmSuppressionRecord.hpp>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <cstddef>
namespace final_state_test {
  namespace current = corsika::kokkos;
  namespace reference = final_state_reference;
  struct Payload {
    double real_[64]{};
    std::uint64_t integer_[64]{};
    unsigned reals_{};
    unsigned integers_{};
    KOKKOS_INLINE_FUNCTION void append(Payload const& other) {
      for (unsigned index = 0; index < other.reals_; ++index)
        real_[reals_++] = other.real_[index];
      for (unsigned index = 0; index < other.integers_; ++index)
        integer_[integers_++] = other.integer_[index];
    }
  };
  KOKKOS_INLINE_FUNCTION Payload flatten(current::ParticleState const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::ParticleState& value, std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::EmParticleState const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::EmParticleState& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(current::FirstInteractionRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::FirstInteractionRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload
  flatten(reference::GpuFirstInteractionSnapshot const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::GpuFirstInteractionSnapshot& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(current::PhotonFinalStateRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::PhotonFinalStateRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::PhotonFinalStateRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::PhotonFinalStateRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(current::LeptonFinalStateRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::LeptonFinalStateRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::BremsFinalStateRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::BremsFinalStateRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload
  flatten(current::PhotonPairLpmSuppressionRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::PhotonPairLpmSuppressionRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload
  flatten(reference::PhotonPairLpmSuppressionRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::PhotonPairLpmSuppressionRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(current::BremsLpmSuppressionRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::BremsLpmSuppressionRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload
  flatten(reference::BremsLpmSuppressionRecord const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::BremsLpmSuppressionRecord& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(current::EmThinningConfig const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::EmThinningConfig& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::EmThinningConfig const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::EmThinningConfig& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(current::EmThinningResult const& value);
  KOKKOS_INLINE_FUNCTION void populate(current::EmThinningResult& value,
                                       std::size_t seed);
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::EmThinningResult const& value);
  KOKKOS_INLINE_FUNCTION void populate(reference::EmThinningResult& value,
                                       std::size_t seed);
  static_assert(sizeof(current::ParticleState) == sizeof(reference::EmParticleState));
  static_assert(alignof(current::ParticleState) == alignof(reference::EmParticleState));
  static_assert(offsetof(current::ParticleState, pid_) ==
                offsetof(reference::EmParticleState, pid));
  static_assert(offsetof(current::ParticleState, mediumId_) ==
                offsetof(reference::EmParticleState, medium_id));
  static_assert(offsetof(current::ParticleState, generation_) ==
                offsetof(reference::EmParticleState, generation));
  static_assert(offsetof(current::ParticleState, reserved_) ==
                offsetof(reference::EmParticleState, reserved));
  static_assert(offsetof(current::ParticleState, energyGeV_) ==
                offsetof(reference::EmParticleState, energy_GeV));
  static_assert(offsetof(current::ParticleState, positionM_) ==
                offsetof(reference::EmParticleState, position_m));
  static_assert(offsetof(current::ParticleState, direction_) ==
                offsetof(reference::EmParticleState, direction));
  static_assert(offsetof(current::ParticleState, timeS_) ==
                offsetof(reference::EmParticleState, time_s));
  static_assert(offsetof(current::ParticleState, weight_) ==
                offsetof(reference::EmParticleState, weight));
  static_assert(offsetof(current::ParticleState, historyId_) ==
                offsetof(reference::EmParticleState, history_id));
  static_assert(offsetof(current::ParticleState, parentHistoryId_) ==
                offsetof(reference::EmParticleState, parent_history_id));
  static_assert(offsetof(current::ParticleState, stepId_) ==
                offsetof(reference::EmParticleState, step_id));
  static_assert(offsetof(current::ParticleState, environmentNode_) ==
                offsetof(reference::EmParticleState, environment_node));
  KOKKOS_INLINE_FUNCTION Payload flatten(current::ParticleState const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.pid_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.mediumId_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.generation_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.reserved_);
    output.real_[output.reals_++] = value.energyGeV_;
    output.real_[output.reals_++] = value.positionM_[0];
    output.real_[output.reals_++] = value.positionM_[1];
    output.real_[output.reals_++] = value.positionM_[2];
    output.real_[output.reals_++] = value.direction_[0];
    output.real_[output.reals_++] = value.direction_[1];
    output.real_[output.reals_++] = value.direction_[2];
    output.real_[output.reals_++] = value.timeS_;
    output.real_[output.reals_++] = value.weight_;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.historyId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.parentHistoryId_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.stepId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.environmentNode_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::ParticleState& value, std::size_t seed) {
    value.pid_ = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 0);
    value.mediumId_ = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 1);
    value.generation_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 2);
    value.reserved_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
    value.energyGeV_ = ((seed + 4) % 2 ? -1. : 1.) * static_cast<double>(seed + 5) / 16.;
    value.positionM_[0] =
        ((seed + 5) % 2 ? -1. : 1.) * static_cast<double>(seed + 6) / 16.;
    value.positionM_[1] =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.positionM_[2] =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.direction_[0] =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.direction_[1] =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.direction_[2] =
        ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.timeS_ = ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.weight_ = ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.historyId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 9);
    value.parentHistoryId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 10);
    value.stepId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 11);
    value.environmentNode_ = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 12);
  }
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::EmParticleState const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.pid);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.medium_id);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.generation);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.reserved);
    output.real_[output.reals_++] = value.energy_GeV;
    output.real_[output.reals_++] = value.position_m[0];
    output.real_[output.reals_++] = value.position_m[1];
    output.real_[output.reals_++] = value.position_m[2];
    output.real_[output.reals_++] = value.direction[0];
    output.real_[output.reals_++] = value.direction[1];
    output.real_[output.reals_++] = value.direction[2];
    output.real_[output.reals_++] = value.time_s;
    output.real_[output.reals_++] = value.weight;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.history_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.parent_history_id);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.step_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.environment_node);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::EmParticleState& value,
                                       std::size_t seed) {
    value.pid = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 0);
    value.medium_id = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 1);
    value.generation = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 2);
    value.reserved = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
    value.energy_GeV = ((seed + 4) % 2 ? -1. : 1.) * static_cast<double>(seed + 5) / 16.;
    value.position_m[0] =
        ((seed + 5) % 2 ? -1. : 1.) * static_cast<double>(seed + 6) / 16.;
    value.position_m[1] =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.position_m[2] =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.direction[0] =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.direction[1] =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.direction[2] =
        ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.time_s = ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.weight = ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.history_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 9);
    value.parent_history_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 10);
    value.step_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 11);
    value.environment_node = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 12);
  }
  static_assert(sizeof(current::FirstInteractionRecord) ==
                sizeof(reference::GpuFirstInteractionSnapshot));
  static_assert(alignof(current::FirstInteractionRecord) ==
                alignof(reference::GpuFirstInteractionSnapshot));
  static_assert(offsetof(current::FirstInteractionRecord, parentAtVertex_) ==
                offsetof(reference::GpuFirstInteractionSnapshot, parent_at_vertex));
  static_assert(offsetof(current::FirstInteractionRecord, secondaries_) ==
                offsetof(reference::GpuFirstInteractionSnapshot, secondaries));
  static_assert(offsetof(current::FirstInteractionRecord, processId_) ==
                offsetof(reference::GpuFirstInteractionSnapshot, process_id));
  static_assert(offsetof(current::FirstInteractionRecord, secondaryCount_) ==
                offsetof(reference::GpuFirstInteractionSnapshot, secondary_count));
  KOKKOS_INLINE_FUNCTION Payload flatten(current::FirstInteractionRecord const& value) {
    Payload output;
    output.append(flatten(value.parentAtVertex_));
    output.append(flatten(value.secondaries_[0]));
    output.append(flatten(value.secondaries_[1]));
    output.append(flatten(value.secondaries_[2]));
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.processId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondaryCount_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::FirstInteractionRecord& value,
                                       std::size_t seed) {
    populate(value.parentAtVertex_, seed + 1);
    populate(value.secondaries_[0], seed + 2);
    populate(value.secondaries_[1], seed + 3);
    populate(value.secondaries_[2], seed + 4);
    value.processId_ = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 2);
    value.secondaryCount_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
  }
  KOKKOS_INLINE_FUNCTION Payload
  flatten(reference::GpuFirstInteractionSnapshot const& value) {
    Payload output;
    output.append(flatten(value.parent_at_vertex));
    output.append(flatten(value.secondaries[0]));
    output.append(flatten(value.secondaries[1]));
    output.append(flatten(value.secondaries[2]));
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.process_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondary_count);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::GpuFirstInteractionSnapshot& value,
                                       std::size_t seed) {
    populate(value.parent_at_vertex, seed + 1);
    populate(value.secondaries[0], seed + 2);
    populate(value.secondaries[1], seed + 3);
    populate(value.secondaries[2], seed + 4);
    value.process_id = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 2);
    value.secondary_count = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
  }
  static_assert(sizeof(current::PhotonFinalStateRecord) ==
                sizeof(reference::PhotonFinalStateRecord));
  static_assert(alignof(current::PhotonFinalStateRecord) ==
                alignof(reference::PhotonFinalStateRecord));
  static_assert(offsetof(current::PhotonFinalStateRecord, inputIndex_) ==
                offsetof(reference::PhotonFinalStateRecord, input_index));
  static_assert(offsetof(current::PhotonFinalStateRecord, parentHistoryId_) ==
                offsetof(reference::PhotonFinalStateRecord, parent_history_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, secondaryOffset_) ==
                offsetof(reference::PhotonFinalStateRecord, secondary_offset));
  static_assert(offsetof(current::PhotonFinalStateRecord, secondaryCount_) ==
                offsetof(reference::PhotonFinalStateRecord, secondary_count));
  static_assert(offsetof(current::PhotonFinalStateRecord, processId_) ==
                offsetof(reference::PhotonFinalStateRecord, process_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, energySplitFraction_) ==
                offsetof(reference::PhotonFinalStateRecord, energy_split_fraction));
  static_assert(offsetof(current::PhotonFinalStateRecord, splitUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, split_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, azimuthUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, azimuth_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, electronPolarUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, electron_polar_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, positronPolarUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, positron_polar_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, lpmSurvivalProbability_) ==
                offsetof(reference::PhotonFinalStateRecord, lpm_survival_probability));
  static_assert(offsetof(current::PhotonFinalStateRecord, lpmUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, lpm_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, splitDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, split_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, azimuthDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, azimuth_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, electronPolarDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, electron_polar_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, positronPolarDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, positron_polar_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, lpmDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, lpm_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, thinningStatus_) ==
                offsetof(reference::PhotonFinalStateRecord, thinning_status));
  static_assert(offsetof(current::PhotonFinalStateRecord, thinningKeepMask_) ==
                offsetof(reference::PhotonFinalStateRecord, thinning_keep_mask));
  static_assert(offsetof(current::PhotonFinalStateRecord, thinningFirstUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, thinning_first_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, thinningSecondUniform_) ==
                offsetof(reference::PhotonFinalStateRecord, thinning_second_uniform));
  static_assert(offsetof(current::PhotonFinalStateRecord, thinningFirstDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, thinning_first_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord, thinningSecondDrawId_) ==
                offsetof(reference::PhotonFinalStateRecord, thinning_second_draw_id));
  static_assert(offsetof(current::PhotonFinalStateRecord,
                         weightedMassConventionCorrectionGeV_) ==
                offsetof(reference::PhotonFinalStateRecord,
                         weighted_mass_convention_correction_GeV));
  KOKKOS_INLINE_FUNCTION Payload flatten(current::PhotonFinalStateRecord const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.inputIndex_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.parentHistoryId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondaryOffset_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondaryCount_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.processId_);
    output.real_[output.reals_++] = value.energySplitFraction_;
    output.real_[output.reals_++] = value.splitUniform_;
    output.real_[output.reals_++] = value.azimuthUniform_;
    output.real_[output.reals_++] = value.electronPolarUniform_;
    output.real_[output.reals_++] = value.positronPolarUniform_;
    output.real_[output.reals_++] = value.lpmSurvivalProbability_;
    output.real_[output.reals_++] = value.lpmUniform_;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.splitDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.azimuthDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.electronPolarDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.positronPolarDrawId_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.lpmDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningStatus_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningKeepMask_);
    output.real_[output.reals_++] = value.thinningFirstUniform_;
    output.real_[output.reals_++] = value.thinningSecondUniform_;
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningFirstDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningSecondDrawId_);
    output.real_[output.reals_++] = value.weightedMassConventionCorrectionGeV_;
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::PhotonFinalStateRecord& value,
                                       std::size_t seed) {
    value.inputIndex_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 0);
    value.parentHistoryId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.secondaryOffset_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.secondaryCount_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
    value.processId_ = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 4);
    value.energySplitFraction_ =
        ((seed + 5) % 2 ? -1. : 1.) * static_cast<double>(seed + 6) / 16.;
    value.splitUniform_ =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.azimuthUniform_ =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.electronPolarUniform_ =
        ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.positronPolarUniform_ =
        ((seed + 9) % 2 ? -1. : 1.) * static_cast<double>(seed + 10) / 16.;
    value.lpmSurvivalProbability_ =
        ((seed + 10) % 2 ? -1. : 1.) * static_cast<double>(seed + 11) / 16.;
    value.lpmUniform_ =
        ((seed + 11) % 2 ? -1. : 1.) * static_cast<double>(seed + 12) / 16.;
    value.splitDrawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 12);
    value.azimuthDrawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 13);
    value.electronPolarDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 14);
    value.positronPolarDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 15);
    value.lpmDrawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 16);
    value.thinningStatus_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 17);
    value.thinningKeepMask_ =
        static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 18);
    value.thinningFirstUniform_ =
        ((seed + 19) % 2 ? -1. : 1.) * static_cast<double>(seed + 20) / 16.;
    value.thinningSecondUniform_ =
        ((seed + 20) % 2 ? -1. : 1.) * static_cast<double>(seed + 21) / 16.;
    value.thinningFirstDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 21);
    value.thinningSecondDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 22);
    value.weightedMassConventionCorrectionGeV_ =
        ((seed + 23) % 2 ? -1. : 1.) * static_cast<double>(seed + 24) / 16.;
  }
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::PhotonFinalStateRecord const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.input_index);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.parent_history_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondary_offset);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondary_count);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.process_id);
    output.real_[output.reals_++] = value.energy_split_fraction;
    output.real_[output.reals_++] = value.split_uniform;
    output.real_[output.reals_++] = value.azimuth_uniform;
    output.real_[output.reals_++] = value.electron_polar_uniform;
    output.real_[output.reals_++] = value.positron_polar_uniform;
    output.real_[output.reals_++] = value.lpm_survival_probability;
    output.real_[output.reals_++] = value.lpm_uniform;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.split_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.azimuth_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.electron_polar_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.positron_polar_draw_id);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.lpm_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_status);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_keep_mask);
    output.real_[output.reals_++] = value.thinning_first_uniform;
    output.real_[output.reals_++] = value.thinning_second_uniform;
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_first_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_second_draw_id);
    output.real_[output.reals_++] = value.weighted_mass_convention_correction_GeV;
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::PhotonFinalStateRecord& value,
                                       std::size_t seed) {
    value.input_index = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 0);
    value.parent_history_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.secondary_offset = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.secondary_count = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
    value.process_id = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 4);
    value.energy_split_fraction =
        ((seed + 5) % 2 ? -1. : 1.) * static_cast<double>(seed + 6) / 16.;
    value.split_uniform =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.azimuth_uniform =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.electron_polar_uniform =
        ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.positron_polar_uniform =
        ((seed + 9) % 2 ? -1. : 1.) * static_cast<double>(seed + 10) / 16.;
    value.lpm_survival_probability =
        ((seed + 10) % 2 ? -1. : 1.) * static_cast<double>(seed + 11) / 16.;
    value.lpm_uniform =
        ((seed + 11) % 2 ? -1. : 1.) * static_cast<double>(seed + 12) / 16.;
    value.split_draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 12);
    value.azimuth_draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 13);
    value.electron_polar_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 14);
    value.positron_polar_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 15);
    value.lpm_draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 16);
    value.thinning_status = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 17);
    value.thinning_keep_mask =
        static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 18);
    value.thinning_first_uniform =
        ((seed + 19) % 2 ? -1. : 1.) * static_cast<double>(seed + 20) / 16.;
    value.thinning_second_uniform =
        ((seed + 20) % 2 ? -1. : 1.) * static_cast<double>(seed + 21) / 16.;
    value.thinning_first_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 21);
    value.thinning_second_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 22);
    value.weighted_mass_convention_correction_GeV =
        ((seed + 23) % 2 ? -1. : 1.) * static_cast<double>(seed + 24) / 16.;
  }
  static_assert(sizeof(current::LeptonFinalStateRecord) ==
                sizeof(reference::BremsFinalStateRecord));
  static_assert(alignof(current::LeptonFinalStateRecord) ==
                alignof(reference::BremsFinalStateRecord));
  static_assert(offsetof(current::LeptonFinalStateRecord, inputIndex_) ==
                offsetof(reference::BremsFinalStateRecord, input_index));
  static_assert(offsetof(current::LeptonFinalStateRecord, parentHistoryId_) ==
                offsetof(reference::BremsFinalStateRecord, parent_history_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, secondaryOffset_) ==
                offsetof(reference::BremsFinalStateRecord, secondary_offset));
  static_assert(offsetof(current::LeptonFinalStateRecord, secondaryCount_) ==
                offsetof(reference::BremsFinalStateRecord, secondary_count));
  static_assert(offsetof(current::LeptonFinalStateRecord, processId_) ==
                offsetof(reference::BremsFinalStateRecord, process_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, photonEnergyFraction_) ==
                offsetof(reference::BremsFinalStateRecord, photon_energy_fraction));
  static_assert(offsetof(current::LeptonFinalStateRecord, finalStateUniform_) ==
                offsetof(reference::BremsFinalStateRecord, final_state_uniform));
  static_assert(offsetof(current::LeptonFinalStateRecord, azimuthUniform_) ==
                offsetof(reference::BremsFinalStateRecord, azimuth_uniform));
  static_assert(offsetof(current::LeptonFinalStateRecord, auxiliaryUniform_) ==
                offsetof(reference::BremsFinalStateRecord, auxiliary_uniform));
  static_assert(offsetof(current::LeptonFinalStateRecord, lpmSurvivalProbability_) ==
                offsetof(reference::BremsFinalStateRecord, lpm_survival_probability));
  static_assert(offsetof(current::LeptonFinalStateRecord, lpmUniform_) ==
                offsetof(reference::BremsFinalStateRecord, lpm_uniform));
  static_assert(offsetof(current::LeptonFinalStateRecord, finalStateDrawId_) ==
                offsetof(reference::BremsFinalStateRecord, final_state_draw_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, azimuthDrawId_) ==
                offsetof(reference::BremsFinalStateRecord, azimuth_draw_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, auxiliaryDrawId_) ==
                offsetof(reference::BremsFinalStateRecord, auxiliary_draw_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, lpmDrawId_) ==
                offsetof(reference::BremsFinalStateRecord, lpm_draw_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, thinningStatus_) ==
                offsetof(reference::BremsFinalStateRecord, thinning_status));
  static_assert(offsetof(current::LeptonFinalStateRecord, thinningKeepMask_) ==
                offsetof(reference::BremsFinalStateRecord, thinning_keep_mask));
  static_assert(offsetof(current::LeptonFinalStateRecord, thinningFirstUniform_) ==
                offsetof(reference::BremsFinalStateRecord, thinning_first_uniform));
  static_assert(offsetof(current::LeptonFinalStateRecord, thinningSecondUniform_) ==
                offsetof(reference::BremsFinalStateRecord, thinning_second_uniform));
  static_assert(offsetof(current::LeptonFinalStateRecord, thinningFirstDrawId_) ==
                offsetof(reference::BremsFinalStateRecord, thinning_first_draw_id));
  static_assert(offsetof(current::LeptonFinalStateRecord, thinningSecondDrawId_) ==
                offsetof(reference::BremsFinalStateRecord, thinning_second_draw_id));
  static_assert(offsetof(current::LeptonFinalStateRecord,
                         weightedMassConventionCorrectionGeV_) ==
                offsetof(reference::BremsFinalStateRecord,
                         weighted_mass_convention_correction_GeV));
  KOKKOS_INLINE_FUNCTION Payload flatten(current::LeptonFinalStateRecord const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.inputIndex_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.parentHistoryId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondaryOffset_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondaryCount_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.processId_);
    output.real_[output.reals_++] = value.photonEnergyFraction_;
    output.real_[output.reals_++] = value.finalStateUniform_;
    output.real_[output.reals_++] = value.azimuthUniform_;
    output.real_[output.reals_++] = value.auxiliaryUniform_;
    output.real_[output.reals_++] = value.lpmSurvivalProbability_;
    output.real_[output.reals_++] = value.lpmUniform_;
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.finalStateDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.azimuthDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.auxiliaryDrawId_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.lpmDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningStatus_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningKeepMask_);
    output.real_[output.reals_++] = value.thinningFirstUniform_;
    output.real_[output.reals_++] = value.thinningSecondUniform_;
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningFirstDrawId_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinningSecondDrawId_);
    output.real_[output.reals_++] = value.weightedMassConventionCorrectionGeV_;
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::LeptonFinalStateRecord& value,
                                       std::size_t seed) {
    value.inputIndex_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 0);
    value.parentHistoryId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.secondaryOffset_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.secondaryCount_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
    value.processId_ = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 4);
    value.photonEnergyFraction_ =
        ((seed + 5) % 2 ? -1. : 1.) * static_cast<double>(seed + 6) / 16.;
    value.finalStateUniform_ =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.azimuthUniform_ =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.auxiliaryUniform_ =
        ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.lpmSurvivalProbability_ =
        ((seed + 9) % 2 ? -1. : 1.) * static_cast<double>(seed + 10) / 16.;
    value.lpmUniform_ =
        ((seed + 10) % 2 ? -1. : 1.) * static_cast<double>(seed + 11) / 16.;
    value.finalStateDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 11);
    value.azimuthDrawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 12);
    value.auxiliaryDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 13);
    value.lpmDrawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 14);
    value.thinningStatus_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 15);
    value.thinningKeepMask_ =
        static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 16);
    value.thinningFirstUniform_ =
        ((seed + 17) % 2 ? -1. : 1.) * static_cast<double>(seed + 18) / 16.;
    value.thinningSecondUniform_ =
        ((seed + 18) % 2 ? -1. : 1.) * static_cast<double>(seed + 19) / 16.;
    value.thinningFirstDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 19);
    value.thinningSecondDrawId_ =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 20);
    value.weightedMassConventionCorrectionGeV_ =
        ((seed + 21) % 2 ? -1. : 1.) * static_cast<double>(seed + 22) / 16.;
  }
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::BremsFinalStateRecord const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.input_index);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.parent_history_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondary_offset);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.secondary_count);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.process_id);
    output.real_[output.reals_++] = value.photon_energy_fraction;
    output.real_[output.reals_++] = value.final_state_uniform;
    output.real_[output.reals_++] = value.azimuth_uniform;
    output.real_[output.reals_++] = value.auxiliary_uniform;
    output.real_[output.reals_++] = value.lpm_survival_probability;
    output.real_[output.reals_++] = value.lpm_uniform;
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.final_state_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.azimuth_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.auxiliary_draw_id);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.lpm_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_status);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_keep_mask);
    output.real_[output.reals_++] = value.thinning_first_uniform;
    output.real_[output.reals_++] = value.thinning_second_uniform;
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_first_draw_id);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.thinning_second_draw_id);
    output.real_[output.reals_++] = value.weighted_mass_convention_correction_GeV;
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::BremsFinalStateRecord& value,
                                       std::size_t seed) {
    value.input_index = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 0);
    value.parent_history_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.secondary_offset = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.secondary_count = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
    value.process_id = static_cast<std::int32_t>(0xfedcba9876543210ULL + seed + 4);
    value.photon_energy_fraction =
        ((seed + 5) % 2 ? -1. : 1.) * static_cast<double>(seed + 6) / 16.;
    value.final_state_uniform =
        ((seed + 6) % 2 ? -1. : 1.) * static_cast<double>(seed + 7) / 16.;
    value.azimuth_uniform =
        ((seed + 7) % 2 ? -1. : 1.) * static_cast<double>(seed + 8) / 16.;
    value.auxiliary_uniform =
        ((seed + 8) % 2 ? -1. : 1.) * static_cast<double>(seed + 9) / 16.;
    value.lpm_survival_probability =
        ((seed + 9) % 2 ? -1. : 1.) * static_cast<double>(seed + 10) / 16.;
    value.lpm_uniform =
        ((seed + 10) % 2 ? -1. : 1.) * static_cast<double>(seed + 11) / 16.;
    value.final_state_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 11);
    value.azimuth_draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 12);
    value.auxiliary_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 13);
    value.lpm_draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 14);
    value.thinning_status = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 15);
    value.thinning_keep_mask =
        static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 16);
    value.thinning_first_uniform =
        ((seed + 17) % 2 ? -1. : 1.) * static_cast<double>(seed + 18) / 16.;
    value.thinning_second_uniform =
        ((seed + 18) % 2 ? -1. : 1.) * static_cast<double>(seed + 19) / 16.;
    value.thinning_first_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 19);
    value.thinning_second_draw_id =
        static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 20);
    value.weighted_mass_convention_correction_GeV =
        ((seed + 21) % 2 ? -1. : 1.) * static_cast<double>(seed + 22) / 16.;
  }
  static_assert(sizeof(current::PhotonPairLpmSuppressionRecord) ==
                sizeof(reference::PhotonPairLpmSuppressionRecord));
  static_assert(alignof(current::PhotonPairLpmSuppressionRecord) ==
                alignof(reference::PhotonPairLpmSuppressionRecord));
  static_assert(offsetof(current::PhotonPairLpmSuppressionRecord, particle_) ==
                offsetof(reference::PhotonPairLpmSuppressionRecord, particle));
  static_assert(offsetof(current::PhotonPairLpmSuppressionRecord, inputIndex_) ==
                offsetof(reference::PhotonPairLpmSuppressionRecord, input_index));
  static_assert(offsetof(current::PhotonPairLpmSuppressionRecord, componentHash_) ==
                offsetof(reference::PhotonPairLpmSuppressionRecord, component_hash));
  static_assert(offsetof(current::PhotonPairLpmSuppressionRecord, survivalProbability_) ==
                offsetof(reference::PhotonPairLpmSuppressionRecord,
                         survival_probability));
  static_assert(offsetof(current::PhotonPairLpmSuppressionRecord, uniform_) ==
                offsetof(reference::PhotonPairLpmSuppressionRecord, uniform));
  static_assert(offsetof(current::PhotonPairLpmSuppressionRecord, drawId_) ==
                offsetof(reference::PhotonPairLpmSuppressionRecord, draw_id));
  KOKKOS_INLINE_FUNCTION Payload
  flatten(current::PhotonPairLpmSuppressionRecord const& value) {
    Payload output;
    output.append(flatten(value.particle_));
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.inputIndex_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.componentHash_);
    output.real_[output.reals_++] = value.survivalProbability_;
    output.real_[output.reals_++] = value.uniform_;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.drawId_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::PhotonPairLpmSuppressionRecord& value,
                                       std::size_t seed) {
    populate(value.particle_, seed + 1);
    value.inputIndex_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.componentHash_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.survivalProbability_ =
        ((seed + 3) % 2 ? -1. : 1.) * static_cast<double>(seed + 4) / 16.;
    value.uniform_ = ((seed + 4) % 2 ? -1. : 1.) * static_cast<double>(seed + 5) / 16.;
    value.drawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 5);
  }
  KOKKOS_INLINE_FUNCTION Payload
  flatten(reference::PhotonPairLpmSuppressionRecord const& value) {
    Payload output;
    output.append(flatten(value.particle));
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.input_index);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.component_hash);
    output.real_[output.reals_++] = value.survival_probability;
    output.real_[output.reals_++] = value.uniform;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.draw_id);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::PhotonPairLpmSuppressionRecord& value,
                                       std::size_t seed) {
    populate(value.particle, seed + 1);
    value.input_index = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.component_hash = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.survival_probability =
        ((seed + 3) % 2 ? -1. : 1.) * static_cast<double>(seed + 4) / 16.;
    value.uniform = ((seed + 4) % 2 ? -1. : 1.) * static_cast<double>(seed + 5) / 16.;
    value.draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 5);
  }
  static_assert(sizeof(current::BremsLpmSuppressionRecord) ==
                sizeof(reference::BremsLpmSuppressionRecord));
  static_assert(alignof(current::BremsLpmSuppressionRecord) ==
                alignof(reference::BremsLpmSuppressionRecord));
  static_assert(offsetof(current::BremsLpmSuppressionRecord, particle_) ==
                offsetof(reference::BremsLpmSuppressionRecord, particle));
  static_assert(offsetof(current::BremsLpmSuppressionRecord, inputIndex_) ==
                offsetof(reference::BremsLpmSuppressionRecord, input_index));
  static_assert(offsetof(current::BremsLpmSuppressionRecord, componentHash_) ==
                offsetof(reference::BremsLpmSuppressionRecord, component_hash));
  static_assert(offsetof(current::BremsLpmSuppressionRecord, survivalProbability_) ==
                offsetof(reference::BremsLpmSuppressionRecord, survival_probability));
  static_assert(offsetof(current::BremsLpmSuppressionRecord, uniform_) ==
                offsetof(reference::BremsLpmSuppressionRecord, uniform));
  static_assert(offsetof(current::BremsLpmSuppressionRecord, drawId_) ==
                offsetof(reference::BremsLpmSuppressionRecord, draw_id));
  KOKKOS_INLINE_FUNCTION Payload
  flatten(current::BremsLpmSuppressionRecord const& value) {
    Payload output;
    output.append(flatten(value.particle_));
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.inputIndex_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.componentHash_);
    output.real_[output.reals_++] = value.survivalProbability_;
    output.real_[output.reals_++] = value.uniform_;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.drawId_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::BremsLpmSuppressionRecord& value,
                                       std::size_t seed) {
    populate(value.particle_, seed + 1);
    value.inputIndex_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.componentHash_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.survivalProbability_ =
        ((seed + 3) % 2 ? -1. : 1.) * static_cast<double>(seed + 4) / 16.;
    value.uniform_ = ((seed + 4) % 2 ? -1. : 1.) * static_cast<double>(seed + 5) / 16.;
    value.drawId_ = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 5);
  }
  KOKKOS_INLINE_FUNCTION Payload
  flatten(reference::BremsLpmSuppressionRecord const& value) {
    Payload output;
    output.append(flatten(value.particle));
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.input_index);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.component_hash);
    output.real_[output.reals_++] = value.survival_probability;
    output.real_[output.reals_++] = value.uniform;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.draw_id);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::BremsLpmSuppressionRecord& value,
                                       std::size_t seed) {
    populate(value.particle, seed + 1);
    value.input_index = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 1);
    value.component_hash = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 2);
    value.survival_probability =
        ((seed + 3) % 2 ? -1. : 1.) * static_cast<double>(seed + 4) / 16.;
    value.uniform = ((seed + 4) % 2 ? -1. : 1.) * static_cast<double>(seed + 5) / 16.;
    value.draw_id = static_cast<std::uint64_t>(0xfedcba9876543210ULL + seed + 5);
  }
  static_assert(sizeof(current::EmThinningConfig) == sizeof(reference::EmThinningConfig));
  static_assert(alignof(current::EmThinningConfig) ==
                alignof(reference::EmThinningConfig));
  static_assert(offsetof(current::EmThinningConfig, thresholdGeV_) ==
                offsetof(reference::EmThinningConfig, threshold_GeV));
  static_assert(offsetof(current::EmThinningConfig, maximumWeight_) ==
                offsetof(reference::EmThinningConfig, maximum_weight));
  static_assert(offsetof(current::EmThinningConfig, enabled_) ==
                offsetof(reference::EmThinningConfig, enabled));
  static_assert(offsetof(current::EmThinningConfig, eraseZeroWeight_) ==
                offsetof(reference::EmThinningConfig, erase_zero_weight));
  KOKKOS_INLINE_FUNCTION Payload flatten(current::EmThinningConfig const& value) {
    Payload output;
    output.real_[output.reals_++] = value.thresholdGeV_;
    output.real_[output.reals_++] = value.maximumWeight_;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.enabled_);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.eraseZeroWeight_);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::EmThinningConfig& value,
                                       std::size_t seed) {
    value.thresholdGeV_ =
        ((seed + 0) % 2 ? -1. : 1.) * static_cast<double>(seed + 1) / 16.;
    value.maximumWeight_ =
        ((seed + 1) % 2 ? -1. : 1.) * static_cast<double>(seed + 2) / 16.;
    value.enabled_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 2);
    value.eraseZeroWeight_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
  }
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::EmThinningConfig const& value) {
    Payload output;
    output.real_[output.reals_++] = value.threshold_GeV;
    output.real_[output.reals_++] = value.maximum_weight;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.enabled);
    output.integer_[output.integers_++] =
        static_cast<std::uint64_t>(value.erase_zero_weight);
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::EmThinningConfig& value,
                                       std::size_t seed) {
    value.threshold_GeV =
        ((seed + 0) % 2 ? -1. : 1.) * static_cast<double>(seed + 1) / 16.;
    value.maximum_weight =
        ((seed + 1) % 2 ? -1. : 1.) * static_cast<double>(seed + 2) / 16.;
    value.enabled = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 2);
    value.erase_zero_weight =
        static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 3);
  }
  static_assert(sizeof(current::EmThinningResult) == sizeof(reference::EmThinningResult));
  static_assert(alignof(current::EmThinningResult) ==
                alignof(reference::EmThinningResult));
  static_assert(offsetof(current::EmThinningResult, status_) ==
                offsetof(reference::EmThinningResult, status));
  static_assert(offsetof(current::EmThinningResult, keepMask_) ==
                offsetof(reference::EmThinningResult, keep_mask));
  static_assert(offsetof(current::EmThinningResult, firstWeight_) ==
                offsetof(reference::EmThinningResult, first_weight));
  static_assert(offsetof(current::EmThinningResult, secondWeight_) ==
                offsetof(reference::EmThinningResult, second_weight));
  KOKKOS_INLINE_FUNCTION Payload flatten(current::EmThinningResult const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.status_);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.keepMask_);
    output.real_[output.reals_++] = value.firstWeight_;
    output.real_[output.reals_++] = value.secondWeight_;
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(current::EmThinningResult& value,
                                       std::size_t seed) {
    value.status_ =
        static_cast<current::EmThinningStatus>(0xfedcba9876543210ULL + seed + 0);
    value.keepMask_ = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 1);
    value.firstWeight_ =
        ((seed + 2) % 2 ? -1. : 1.) * static_cast<double>(seed + 3) / 16.;
    value.secondWeight_ =
        ((seed + 3) % 2 ? -1. : 1.) * static_cast<double>(seed + 4) / 16.;
  }
  KOKKOS_INLINE_FUNCTION Payload flatten(reference::EmThinningResult const& value) {
    Payload output;
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.status);
    output.integer_[output.integers_++] = static_cast<std::uint64_t>(value.keep_mask);
    output.real_[output.reals_++] = value.first_weight;
    output.real_[output.reals_++] = value.second_weight;
    return output;
  }
  KOKKOS_INLINE_FUNCTION void populate(reference::EmThinningResult& value,
                                       std::size_t seed) {
    value.status =
        static_cast<reference::EmThinningStatus>(0xfedcba9876543210ULL + seed + 0);
    value.keep_mask = static_cast<std::uint32_t>(0xfedcba9876543210ULL + seed + 1);
    value.first_weight =
        ((seed + 2) % 2 ? -1. : 1.) * static_cast<double>(seed + 3) / 16.;
    value.second_weight =
        ((seed + 3) % 2 ? -1. : 1.) * static_cast<double>(seed + 4) / 16.;
  }
} // namespace final_state_test
