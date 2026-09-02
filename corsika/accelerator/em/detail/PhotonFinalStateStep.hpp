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
#include <corsika/accelerator/em/PhotonFinalStateRandomDomains.hpp>
#include <corsika/gpu/em/EmThinning.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/PhotonPairFinalState.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::accelerator::em::detail {

  /**
   * Backend-neutral intermediate state for one photon interaction.
   *
   * Classification is deliberately separate from output allocation.  A
   * Kokkos parallel_scan and the native CUDA CUB scan can therefore assign
   * the same stable child offsets without atomically appending particles.
   */
  struct PhotonFinalStateParameters {
    std::int32_t process_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double split_fraction{};
    double split_uniform{};
    std::uint64_t split_draw_id{PhotonPairSplitDrawId};
    double azimuth_uniform{};
    double electron_polar_uniform{};
    double positron_polar_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    double thinning_first_weight{};
    double thinning_second_weight{};
  };

  struct PhotonFinalStateClassification {
    PhotonFinalStateParameters parameters{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t child_count{};
    std::uint32_t record_flag{};
    std::uint32_t fallback_flag{};
    std::uint32_t continuation_flag{};
    std::uint32_t suppression_flag{};
    std::uint32_t photon_pair_flag{};
    std::uint32_t compton_flag{};
    std::uint32_t photoelectric_flag{};
  };

  struct PhotonFinalStateMaterialization {
    gpu::em::PhotonPairFinalStateRecord record{};
    gpu::em::EmParticleState secondaries[2]{};
    gpu::em::ProposalFallbackEvent fallback{};
    gpu::em::EmInteractionRecord continuation{};
    gpu::em::PhotonPairLpmSuppressionRecord suppression{};
    gpu::em::GpuFirstInteractionSnapshot first_interaction{};
    std::uint32_t secondary_count{};
    std::uint32_t has_record{};
    std::uint32_t has_fallback{};
    std::uint32_t has_continuation{};
    std::uint32_t has_suppression{};
    std::uint32_t has_first_interaction{};
    std::uint32_t error{};
  };

  C8_ACCELERATOR_INLINE_FUNCTION inline bool photonFinite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double photonSqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(value);
#else
    return std::sqrt(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double photonCos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::cos(value);
#else
    return std::cos(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double photonSin(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sin(value);
#else
    return std::sin(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double photonFabs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double photonFmod(
      double left, double right) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fmod(left, right);
#else
    return std::fmod(left, right);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double photonMaximum(
      double left, double right) {
    return left > right ? left : right;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double boundedCosine(double value) {
    return photonMaximum(-1., value < 1. ? value : 1.);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double sauterCosine(
      double energy_GeV, double uniform) {
    auto const mass_squared =
        gpu::em::ElectronMassGeV * gpu::em::ElectronMassGeV;
    auto const momentum = photonSqrt(photonMaximum(
        0., energy_GeV * energy_GeV - mass_squared));
    auto const coordinate = 2. * uniform - 1.;
    auto const energy_minus_momentum =
        mass_squared / (energy_GeV + momentum);
    auto const one_minus_cosine =
        energy_minus_momentum * (1. - coordinate) /
        (momentum * coordinate + energy_GeV);
    return boundedCosine(1. - one_minus_cosine);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool comptonCosines(
      double energy_GeV, double v, double& photon_cosine,
      double& electron_cosine) {
    auto const scattered_energy = energy_GeV * (1. - v);
    auto const electron_momentum_squared =
        2. * v * energy_GeV * gpu::em::ElectronMassGeV +
        v * v * energy_GeV * energy_GeV;
    if (!photonFinite(energy_GeV) || !(energy_GeV > 0.) ||
        !photonFinite(v) || !(v > 0.) || !(v < 1.) ||
        !(scattered_energy > 0.) || !(electron_momentum_squared > 0.)) {
      return false;
    }
    photon_cosine =
        1. - (v * gpu::em::ElectronMassGeV) / scattered_energy;
    electron_cosine =
        v * (energy_GeV + gpu::em::ElectronMassGeV) /
        photonSqrt(electron_momentum_squared);
    auto constexpr tolerance = 1.e-12;
    if (!photonFinite(photon_cosine) || !photonFinite(electron_cosine) ||
        photon_cosine < -1. - tolerance || photon_cosine > 1. + tolerance ||
        electron_cosine < -1. - tolerance ||
        electron_cosine > 1. + tolerance) {
      return false;
    }
    photon_cosine = boundedCosine(photon_cosine);
    electron_cosine = boundedCosine(electron_cosine);
    return true;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool photoelectricEnergy(
      gpu::em::PhotonPairLpmSnapshot const& snapshot,
      std::uint64_t component_hash, double parent_energy_GeV,
      double& electron_total_energy_GeV, double& kinetic_fraction) {
    if (!photonFinite(parent_energy_GeV) || !(parent_energy_GeV > 0.) ||
        snapshot.component_count == 0 ||
        snapshot.component_count > gpu::em::MaxPhotonPairLpmComponents ||
        !photonFinite(snapshot.fine_structure_constant) ||
        !(snapshot.fine_structure_constant > 0.)) {
      return false;
    }
    double nuclear_charge = 0.;
    for (std::uint32_t index = 0; index < snapshot.component_count; ++index) {
      if (snapshot.components[index].component_hash == component_hash) {
        nuclear_charge = snapshot.components[index].nuclear_charge;
        break;
      }
    }
    if (!photonFinite(nuclear_charge) || !(nuclear_charge > 0.)) return false;
    auto const z_alpha = nuclear_charge * snapshot.fine_structure_constant;
    auto const binding_energy_GeV =
        z_alpha * z_alpha * gpu::em::ElectronMassGeV / 2.;
    auto const kinetic_energy_GeV = parent_energy_GeV - binding_energy_GeV;
    if (!photonFinite(binding_energy_GeV) ||
        !photonFinite(kinetic_energy_GeV) || !(kinetic_energy_GeV >= 0.)) {
      return false;
    }
    electron_total_energy_GeV =
        gpu::em::ElectronMassGeV + kinetic_energy_GeV;
    kinetic_fraction = kinetic_energy_GeV / parent_energy_GeV;
    return photonFinite(electron_total_energy_GeV) &&
           electron_total_energy_GeV >= gpu::em::ElectronMassGeV &&
           photonFinite(kinetic_fraction) && kinetic_fraction >= 0. &&
           kinetic_fraction <= 1.;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline void deflectPhotonChild(
      double const input[3], double cosine, double azimuth,
      double output[3]) {
    auto const transverse = photonSqrt(photonMaximum(
        0., input[0] * input[0] + input[1] * input[1]));
    double cosine_phi = 1.;
    double sine_phi = 0.;
    if (transverse > 0.) {
      cosine_phi = input[0] / transverse;
      sine_phi = input[1] / transverse;
    }
    auto const cosine_theta = input[2];
    auto const sine_theta = transverse;
    double const rotation_x[3]{cosine_theta * cosine_phi,
                               cosine_theta * sine_phi, -sine_theta};
    double const rotation_y[3]{-sine_phi, cosine_phi, 0.};
    auto const sine = photonSqrt(
        photonMaximum(0., (1. - cosine) * (1. + cosine)));
    auto const local_x = sine * photonCos(azimuth);
    auto const local_y = sine * photonSin(azimuth);
    auto local_z = photonSqrt(photonMaximum(
        0., 1. - local_x * local_x - local_y * local_y));
    if (cosine < 0.) local_z = -local_z;
    for (int axis = 0; axis < 3; ++axis) {
      output[axis] = local_z * input[axis] + local_x * rotation_x[axis] +
                     local_y * rotation_y[axis];
    }
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  invalidPhotonFinalState(
      gpu::em::EmInteractionRecord const& interaction,
      PhotonFinalStateParameters const& parameters = {}) {
    auto event = gpu::em::makeProcessFallbackEvent(
        interaction, gpu::em::ProposalFallbackReason::InvalidFinalState);
    if (parameters.split_uniform > 0.) {
      event.final_state_uniform = parameters.split_uniform;
      event.final_state_draw_id = parameters.split_draw_id;
    }
    return event;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool applyPhotonTwoChildThinning(
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmParticleState const& parent, std::int32_t process_id,
      double first_energy_GeV, double second_energy_GeV,
      std::uint64_t random_seed, std::uint64_t shower_id,
      PhotonFinalStateParameters& sample, std::uint32_t& child_count) {
    gpu::em::RandomNumberKey first_key{
        random_seed, shower_id, parent.history_id, parent.step_id,
        static_cast<std::uint32_t>(process_id),
        gpu::em::EmThinningFirstDrawId};
    auto second_key = first_key;
    second_key.draw_id = gpu::em::EmThinningSecondDrawId;
    sample.thinning_first_uniform = gpu::em::uniformOpen01(first_key);
    sample.thinning_second_uniform = gpu::em::uniformOpen01(second_key);
    auto const result = gpu::em::applyEmThinning(
        thinning, parent.energy_GeV, parent.weight, first_energy_GeV,
        second_energy_GeV, sample.thinning_first_uniform,
        sample.thinning_second_uniform);
    if (result.status == gpu::em::EmThinningStatus::InvalidInput) return false;
    sample.thinning_status = static_cast<std::uint32_t>(result.status);
    sample.thinning_keep_mask = result.keep_mask;
    sample.thinning_first_weight = result.first_weight;
    sample.thinning_second_weight = result.second_weight;
    child_count = (result.keep_mask & 0x1U ? 1U : 0U) +
                  (result.keep_mask & 0x2U ? 1U : 0U);
    return true;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline PhotonFinalStateClassification
  classifyPhotonFinalState(
      gpu::em::tables::FlatRateTableView const& table,
      gpu::em::PhotonPairLpmSnapshot const& lpm_snapshot,
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmInteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id) {
    PhotonFinalStateClassification result{};
    if (interaction.status ==
        gpu::em::EmInteractionStatus::NoDiscreteInteraction) {
      result.continuation_flag = 1;
      return result;
    }
    if (interaction.status != gpu::em::EmInteractionStatus::Selected) {
      result.fallback = invalidPhotonFinalState(interaction);
      result.fallback_flag = 1;
      return result;
    }
    auto const capability = gpu::em::gpuProcessCapability(
        interaction.particle.pid, interaction.process_id);
    if (capability != gpu::em::GpuProcessCapability::PhotonPair &&
        capability != gpu::em::GpuProcessCapability::Compton &&
        capability != gpu::em::GpuProcessCapability::Photoelectric) {
      result.fallback = gpu::em::makeProcessFallbackEvent(
          interaction, gpu::em::processFallbackReason(capability));
      result.fallback_flag = 1;
      return result;
    }
    auto const& parent = interaction.particle;
    if (capability == gpu::em::GpuProcessCapability::Compton) {
      double photon_cosine = 0.;
      double electron_cosine = 0.;
      if (parent.generation == 0xffffffffU ||
          parent.step_id == 0xffffffffffffffffULL ||
          !comptonCosines(parent.energy_GeV, interaction.energy_fraction,
                          photon_cosine, electron_cosine)) {
        result.fallback = invalidPhotonFinalState(interaction);
        result.fallback_flag = 1;
        return result;
      }
      auto& sample = result.parameters;
      sample.process_id = gpu::em::ComptonProcessId;
      sample.split_fraction = interaction.energy_fraction;
      sample.split_uniform = interaction.loss_quantile;
      gpu::em::RandomNumberKey const azimuth_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(gpu::em::ComptonProcessId),
          ComptonAzimuthDrawId};
      sample.azimuth_uniform = gpu::em::uniformOpen01(azimuth_key);
      if (!applyPhotonTwoChildThinning(
              thinning, parent, gpu::em::ComptonProcessId,
              parent.energy_GeV * (1. - sample.split_fraction),
              gpu::em::ElectronMassGeV +
                  parent.energy_GeV * sample.split_fraction,
              random_seed, shower_id, sample, result.child_count)) {
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      result.record_flag = 1;
      result.compton_flag = 1;
      return result;
    }
    if (capability == gpu::em::GpuProcessCapability::Photoelectric) {
      double electron_energy_GeV = 0.;
      double kinetic_fraction = 0.;
      if (parent.generation == 0xffffffffU ||
          !photonFinite(interaction.energy_fraction) ||
          photonFabs(interaction.energy_fraction - 1.) > 1.e-12 ||
          !photoelectricEnergy(lpm_snapshot, interaction.component_hash,
                               parent.energy_GeV, electron_energy_GeV,
                               kinetic_fraction)) {
        result.fallback = invalidPhotonFinalState(interaction);
        result.fallback_flag = 1;
        return result;
      }
      auto& sample = result.parameters;
      sample.process_id = gpu::em::PhotoelectricProcessId;
      sample.split_fraction = kinetic_fraction;
      sample.split_uniform = interaction.loss_quantile;
      result.child_count = 1;
      result.record_flag = 1;
      result.photoelectric_flag = 1;
      return result;
    }
    if (!photonFinite(parent.energy_GeV) ||
        parent.energy_GeV < 2. * gpu::em::ElectronMassGeV ||
        parent.generation == 0xffffffffU ||
        parent.step_id == 0xffffffffffffffffULL) {
      result.fallback = invalidPhotonFinalState(interaction);
      result.fallback_flag = 1;
      return result;
    }

    auto& sample = result.parameters;
    sample.process_id = gpu::em::PhotonPairProcessId;
    gpu::em::RandomNumberKey const split_key{
        random_seed, shower_id, parent.history_id, parent.step_id,
        static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
        PhotonPairSplitDrawId};
    sample.split_uniform = gpu::em::uniformOpen01(split_key);
    sample.split_draw_id = PhotonPairSplitDrawId;
    gpu::em::tables::TableQuery const split_query{
        gpu::em::tables::TableQueryKind::LossFraction, parent.pid,
        gpu::em::PhotonPairFinalStateProcessId, interaction.component_hash,
        parent.energy_GeV * 1000., sample.split_uniform};
    auto const split = gpu::em::tables::executeTableQuery(table, split_query);
    bool split_ready = false;
    if (split.status == gpu::em::tables::TableLookupStatus::Success) {
      split_ready = gpu::em::decodePhotonPairNormalizedSplit(
          parent.energy_GeV, split.value, sample.split_fraction);
      if (!split_ready) {
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
    } else if (
        split.status ==
            gpu::em::tables::TableLookupStatus::LossEnergyOutOfRange ||
        (table.physics_source == 1u &&
         split.status == gpu::em::tables::TableLookupStatus::ColumnNotFound)) {
      for (std::uint32_t attempt = 0;
           attempt < PhotonPairAnalyticMaximumAttempts; ++attempt) {
        gpu::em::RandomNumberKey candidate_key{
            random_seed, shower_id, parent.history_id, parent.step_id,
            static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
            PhotonPairAnalyticCandidateDrawIdBase + attempt};
        auto acceptance_key = candidate_key;
        acceptance_key.draw_id =
            PhotonPairAnalyticAcceptanceDrawIdBase + attempt;
        sample.split_uniform = gpu::em::uniformOpen01(candidate_key);
        sample.split_draw_id = candidate_key.draw_id;
        auto const trial = gpu::em::photonPairFinalStateTrial(
            lpm_snapshot, interaction.component_hash,
            parent.energy_GeV * 1000., sample.split_uniform,
            gpu::em::uniformOpen01(acceptance_key));
        if (trial.status != gpu::em::PhotonPairFinalStateStatus::Success) {
          result.fallback = invalidPhotonFinalState(interaction, sample);
          result.fallback_flag = 1;
          return result;
        }
        if (trial.accepted != 0) {
          sample.split_fraction = trial.split_fraction;
          split_ready = true;
          break;
        }
      }
      if (!split_ready) {
        result.fallback = invalidPhotonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
    } else {
      auto event = gpu::em::makeTableFallbackEvent(
          parent, split_query, split, PhotonPairSplitDrawId,
          interaction.input_index);
      event.energy_fraction = interaction.energy_fraction;
      event.process_id = interaction.process_id;
      auto const native_selection =
          interaction.proposal_selection_random_process_id != 0u;
      event.selection_uniform =
          native_selection ? interaction.proposal_selection_uniform
                           : interaction.process_uniform;
      event.loss_quantile = interaction.loss_quantile;
      event.outer_acceptance_uniform = interaction.process_uniform;
      event.outer_acceptance_random_process_id =
          interaction.process_random_process_id;
      event.outer_acceptance_draw_id = interaction.process_draw_id;
      event.random_process_id =
          native_selection ? interaction.proposal_selection_random_process_id
                           : 0u;
      event.random_draw_id =
          native_selection ? interaction.proposal_selection_draw_id
                           : interaction.loss_draw_id;
      event.final_state_uniform = sample.split_uniform;
      event.final_state_draw_id = PhotonPairSplitDrawId;
      result.fallback = event;
      result.fallback_flag = 1;
      return result;
    }
    auto const electron_energy = parent.energy_GeV * sample.split_fraction;
    auto const positron_energy = parent.energy_GeV - electron_energy;
    if (!photonFinite(sample.split_fraction) ||
        !(sample.split_fraction > 0.) || !(sample.split_fraction < 1.) ||
        electron_energy < gpu::em::ElectronMassGeV ||
        positron_energy < gpu::em::ElectronMassGeV) {
      result.fallback = invalidPhotonFinalState(interaction, sample);
      result.fallback_flag = 1;
      return result;
    }
    gpu::em::RandomNumberKey angle_key{
        random_seed, shower_id, parent.history_id, parent.step_id,
        static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
        PhotonPairAzimuthDrawId};
    sample.azimuth_uniform = gpu::em::uniformOpen01(angle_key);
    angle_key.draw_id = PhotonPairElectronPolarDrawId;
    sample.electron_polar_uniform = gpu::em::uniformOpen01(angle_key);
    angle_key.draw_id = PhotonPairPositronPolarDrawId;
    sample.positron_polar_uniform = gpu::em::uniformOpen01(angle_key);
    auto const lpm = gpu::em::photonPairLpmSuppressionFactor(
        lpm_snapshot, interaction.component_hash, parent.energy_GeV * 1000.,
        sample.split_fraction, interaction.mass_density_g_per_cm3);
    if (lpm.status != gpu::em::PhotonPairLpmStatus::Success) {
      auto const reason =
          lpm.status == gpu::em::PhotonPairLpmStatus::InvalidInput
              ? gpu::em::ProposalFallbackReason::InvalidMassDensity
              : gpu::em::ProposalFallbackReason::LpmParametersUnavailable;
      result.fallback = gpu::em::makeProcessFallbackEvent(interaction, reason);
      result.fallback_flag = 1;
      return result;
    }
    sample.lpm_survival_probability = lpm.survival_probability;
    gpu::em::RandomNumberKey const lpm_key{
        random_seed, shower_id, parent.history_id, parent.step_id,
        static_cast<std::uint32_t>(gpu::em::PhotonPairProcessId),
        PhotonPairLpmDrawId};
    sample.lpm_uniform = gpu::em::uniformOpen01(lpm_key);
    if (sample.lpm_uniform > sample.lpm_survival_probability) {
      result.suppression_flag = 1;
      return result;
    }
    if (!applyPhotonTwoChildThinning(
            thinning, parent, gpu::em::PhotonPairProcessId, electron_energy,
            positron_energy, random_seed, shower_id, sample,
            result.child_count)) {
      result.fallback = invalidPhotonFinalState(interaction, sample);
      result.fallback_flag = 1;
      return result;
    }
    result.record_flag = 1;
    result.photon_pair_flag = 1;
    return result;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline void fillFirstInteraction(
      gpu::em::EmParticleState const& parent, std::int32_t process_id,
      gpu::em::EmParticleState const& first,
      gpu::em::EmParticleState const* second,
      PhotonFinalStateMaterialization& output) {
    if (parent.generation != 0) return;
    output.has_first_interaction = 1;
    output.first_interaction.parent_at_vertex = parent;
    output.first_interaction.process_id = process_id;
    output.first_interaction.secondary_count = second == nullptr ? 1U : 2U;
    output.first_interaction.secondaries[0] = first;
    output.first_interaction.secondaries[0].weight = parent.weight;
    if (second != nullptr) {
      output.first_interaction.secondaries[1] = *second;
      output.first_interaction.secondaries[1].weight = parent.weight;
    }
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline PhotonFinalStateMaterialization
  materializePhotonFinalState(
      gpu::em::EmInteractionRecord const& interaction,
      PhotonFinalStateClassification const& classification,
      std::uint64_t child_offset, std::uint64_t first_history_id) {
    PhotonFinalStateMaterialization output{};
    if (classification.fallback_flag != 0) {
      output.fallback = classification.fallback;
      output.has_fallback = 1;
      return output;
    }
    if (classification.continuation_flag != 0) {
      output.continuation = interaction;
      output.has_continuation = 1;
      return output;
    }
    if (classification.suppression_flag != 0) {
      auto parent = interaction.particle;
      ++parent.step_id;
      auto const& sample = classification.parameters;
      output.suppression = {
          parent, interaction.input_index, interaction.component_hash,
          sample.lpm_survival_probability, sample.lpm_uniform,
          PhotonPairLpmDrawId};
      output.has_suppression = 1;
      return output;
    }
    if (classification.record_flag == 0) return output;

    auto const& parent = interaction.particle;
    auto const& sample = classification.parameters;
    auto const azimuth =
        sample.azimuth_uniform * 6.283185307179586476925286766559;
    if (sample.process_id == gpu::em::ComptonProcessId) {
      double photon_cosine = 0.;
      double electron_cosine = 0.;
      if (!comptonCosines(parent.energy_GeV, sample.split_fraction,
                          photon_cosine, electron_cosine)) {
        output.error = 1;
        return output;
      }
      auto photon = parent;
      photon.energy_GeV = parent.energy_GeV * (1. - sample.split_fraction);
      photon.parent_history_id = parent.history_id;
      photon.generation = parent.generation + 1;
      photon.step_id = 0;
      photon.reserved = 0;
      photon.weight = sample.thinning_first_weight;
      deflectPhotonChild(parent.direction, photon_cosine, azimuth,
                         photon.direction);
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Electron);
      electron.energy_GeV = gpu::em::ElectronMassGeV +
                            parent.energy_GeV * sample.split_fraction;
      electron.parent_history_id = parent.history_id;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      electron.weight = sample.thinning_second_weight;
      deflectPhotonChild(
          parent.direction, electron_cosine,
          photonFmod(azimuth + 3.1415926535897932384626433832795,
                     6.283185307179586476925286766559),
          electron.direction);
      fillFirstInteraction(parent, gpu::em::ComptonProcessId, photon,
                           &electron, output);
      if ((sample.thinning_keep_mask & 0x1U) != 0) {
        photon.history_id = first_history_id + child_offset +
                            output.secondary_count;
        output.secondaries[output.secondary_count++] = photon;
      }
      if ((sample.thinning_keep_mask & 0x2U) != 0) {
        electron.history_id = first_history_id + child_offset +
                              output.secondary_count;
        output.secondaries[output.secondary_count++] = electron;
      }
      output.record = {
          interaction.input_index, parent.history_id, child_offset,
          output.secondary_count, gpu::em::ComptonProcessId,
          sample.split_fraction, interaction.loss_quantile,
          sample.azimuth_uniform, 0., 0., 0., 0., interaction.loss_draw_id,
          ComptonAzimuthDrawId, 0, 0, 0};
    } else if (sample.process_id == gpu::em::PhotoelectricProcessId) {
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Electron);
      electron.energy_GeV = gpu::em::ElectronMassGeV +
                            parent.energy_GeV * sample.split_fraction;
      electron.parent_history_id = parent.history_id;
      electron.history_id = first_history_id + child_offset;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      fillFirstInteraction(parent, gpu::em::PhotoelectricProcessId, electron,
                           nullptr, output);
      output.secondaries[0] = electron;
      output.secondary_count = 1;
      output.record = {
          interaction.input_index, parent.history_id, child_offset, 1,
          gpu::em::PhotoelectricProcessId, sample.split_fraction,
          interaction.loss_quantile, 0., 0., 0., 0., 0.,
          interaction.loss_draw_id, 0, 0, 0, 0};
    } else {
      auto const electron_energy =
          parent.energy_GeV * sample.split_fraction;
      auto const positron_energy = parent.energy_GeV - electron_energy;
      auto const electron_cosine =
          sauterCosine(electron_energy, sample.electron_polar_uniform);
      auto const positron_cosine =
          sauterCosine(positron_energy, sample.positron_polar_uniform);
      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Electron);
      electron.energy_GeV = electron_energy;
      electron.parent_history_id = parent.history_id;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      electron.weight = sample.thinning_first_weight;
      deflectPhotonChild(parent.direction, electron_cosine, azimuth,
                         electron.direction);
      auto positron = parent;
      positron.pid = static_cast<std::int32_t>(gpu::em::EmPid::Positron);
      positron.energy_GeV = positron_energy;
      positron.parent_history_id = parent.history_id;
      positron.generation = parent.generation + 1;
      positron.step_id = 0;
      positron.reserved = 0;
      positron.weight = sample.thinning_second_weight;
      deflectPhotonChild(
          parent.direction, positron_cosine,
          photonFmod(azimuth + 3.1415926535897932384626433832795,
                     6.283185307179586476925286766559),
          positron.direction);
      fillFirstInteraction(parent, gpu::em::PhotonPairProcessId, electron,
                           &positron, output);
      if ((sample.thinning_keep_mask & 0x1U) != 0) {
        electron.history_id = first_history_id + child_offset +
                              output.secondary_count;
        output.secondaries[output.secondary_count++] = electron;
      }
      if ((sample.thinning_keep_mask & 0x2U) != 0) {
        positron.history_id = first_history_id + child_offset +
                              output.secondary_count;
        output.secondaries[output.secondary_count++] = positron;
      }
      output.record = {
          interaction.input_index, parent.history_id, child_offset,
          output.secondary_count, gpu::em::PhotonPairProcessId,
          sample.split_fraction, sample.split_uniform,
          sample.azimuth_uniform, sample.electron_polar_uniform,
          sample.positron_polar_uniform, sample.lpm_survival_probability,
          sample.lpm_uniform, sample.split_draw_id, PhotonPairAzimuthDrawId,
          PhotonPairElectronPolarDrawId, PhotonPairPositronPolarDrawId,
          PhotonPairLpmDrawId};
    }
    output.record.thinning_status = sample.thinning_status;
    output.record.thinning_keep_mask = sample.thinning_keep_mask;
    output.record.thinning_first_uniform = sample.thinning_first_uniform;
    output.record.thinning_second_uniform = sample.thinning_second_uniform;
    output.record.thinning_first_draw_id = gpu::em::EmThinningFirstDrawId;
    output.record.thinning_second_draw_id = gpu::em::EmThinningSecondDrawId;
    output.has_record = 1;
    return output;
  }

} // namespace corsika::accelerator::em::detail
