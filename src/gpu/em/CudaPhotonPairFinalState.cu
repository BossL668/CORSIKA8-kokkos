/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <corsika/gpu/em/CudaPhotonPairFinalState.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr unsigned int ClassificationThreadsPerBlock = 64;

    struct PhotonPairParameters {
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

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    __device__ double boundedCosine(double value) {
      return ::fmax(-1., ::fmin(1., value));
    }

    __device__ double sauterCosine(double energy_GeV, double uniform) {
      auto const mass_squared =
          ElectronMassGeV * ElectronMassGeV;
      auto const momentum =
          ::sqrt(::fmax(0., energy_GeV * energy_GeV -
                                mass_squared));
      auto const coordinate = 2. * uniform - 1.;
      auto const energy_minus_momentum =
          mass_squared / (energy_GeV + momentum);
      auto const one_minus_cosine =
          energy_minus_momentum * (1. - coordinate) /
          (momentum * coordinate + energy_GeV);
      return boundedCosine(1. - one_minus_cosine);
    }

    __device__ bool comptonCosines(
        double energy_GeV, double v,
        double& photon_cosine,
        double& electron_cosine) {
      auto const scattered_energy =
          energy_GeV * (1. - v);
      auto const electron_momentum_squared =
          2. * v * energy_GeV * ElectronMassGeV +
          v * v * energy_GeV * energy_GeV;
      if (!::isfinite(energy_GeV) ||
          !(energy_GeV > 0.) || !::isfinite(v) ||
          !(v > 0.) || !(v < 1.) ||
          !(scattered_energy > 0.) ||
          !(electron_momentum_squared > 0.)) {
        return false;
      }
      photon_cosine =
          1. - (v * ElectronMassGeV) /
                   scattered_energy;
      electron_cosine =
          v * (energy_GeV + ElectronMassGeV) /
          ::sqrt(electron_momentum_squared);
      auto constexpr tolerance = 1.e-12;
      if (!::isfinite(photon_cosine) ||
          !::isfinite(electron_cosine) ||
          photon_cosine < -1. - tolerance ||
          photon_cosine > 1. + tolerance ||
          electron_cosine < -1. - tolerance ||
          electron_cosine > 1. + tolerance) {
        return false;
      }
      photon_cosine = boundedCosine(photon_cosine);
      electron_cosine = boundedCosine(electron_cosine);
      return true;
    }

    __device__ bool photoelectricEnergy(
        PhotonPairLpmSnapshot const& snapshot,
        std::uint64_t component_hash,
        double parent_energy_GeV,
        double& electron_total_energy_GeV,
        double& kinetic_fraction) {
      if (!::isfinite(parent_energy_GeV) ||
          !(parent_energy_GeV > 0.) ||
          snapshot.component_count == 0 ||
          snapshot.component_count >
              MaxPhotonPairLpmComponents ||
          !::isfinite(snapshot.fine_structure_constant) ||
          !(snapshot.fine_structure_constant > 0.)) {
        return false;
      }
      double nuclear_charge = 0.;
      for (std::uint32_t index = 0;
           index < snapshot.component_count; ++index) {
        if (snapshot.components[index].component_hash ==
            component_hash) {
          nuclear_charge =
              snapshot.components[index].nuclear_charge;
          break;
        }
      }
      if (!::isfinite(nuclear_charge) ||
          !(nuclear_charge > 0.)) {
        return false;
      }
      auto const z_alpha =
          nuclear_charge *
          snapshot.fine_structure_constant;
      auto const binding_energy_GeV =
          z_alpha * z_alpha * ElectronMassGeV / 2.;
      auto const kinetic_energy_GeV =
          parent_energy_GeV - binding_energy_GeV;
      if (!::isfinite(binding_energy_GeV) ||
          !::isfinite(kinetic_energy_GeV) ||
          !(kinetic_energy_GeV >= 0.)) {
        return false;
      }
      electron_total_energy_GeV =
          ElectronMassGeV + kinetic_energy_GeV;
      kinetic_fraction =
          kinetic_energy_GeV / parent_energy_GeV;
      return ::isfinite(electron_total_energy_GeV) &&
             electron_total_energy_GeV >= ElectronMassGeV &&
             ::isfinite(kinetic_fraction) &&
             kinetic_fraction >= 0. &&
             kinetic_fraction <= 1.;
    }

    __device__ void deflect(double const input[3], double cosine,
                            double azimuth, double output[3]) {
      auto const transverse =
          ::sqrt(::fmax(0., input[0] * input[0] +
                                input[1] * input[1]));
      double cosine_phi = 1.;
      double sine_phi = 0.;
      if (transverse > 0.) {
        cosine_phi = input[0] / transverse;
        sine_phi = input[1] / transverse;
      }
      auto const cosine_theta = input[2];
      auto const sine_theta = transverse;
      double const rotation_x[3]{
          cosine_theta * cosine_phi,
          cosine_theta * sine_phi,
          -sine_theta};
      double const rotation_y[3]{-sine_phi, cosine_phi, 0.};

      auto const sine =
          ::sqrt(::fmax(0., (1. - cosine) * (1. + cosine)));
      auto const local_x = sine * ::cos(azimuth);
      auto const local_y = sine * ::sin(azimuth);
      auto local_z = ::sqrt(::fmax(
          0., 1. - local_x * local_x - local_y * local_y));
      if (cosine < 0.) {
        local_z = -local_z;
      }
      for (int axis = 0; axis < 3; ++axis) {
        output[axis] = local_z * input[axis] +
                       local_x * rotation_x[axis] +
                       local_y * rotation_y[axis];
      }
    }

    __device__ ProposalFallbackEvent invalidFinalState(
        EmInteractionRecord const& interaction,
        PhotonPairParameters const& parameters = {}) {
      auto event = makeProcessFallbackEvent(
          interaction, ProposalFallbackReason::InvalidFinalState);
      if (parameters.split_uniform > 0.) {
        event.final_state_uniform = parameters.split_uniform;
        event.final_state_draw_id = parameters.split_draw_id;
      }
      return event;
    }

    __device__ bool applyTwoChildThinning(
        EmThinningConfig const& thinning,
        EmParticleState const& parent, std::int32_t process_id,
        double first_energy_GeV, double second_energy_GeV,
        std::uint64_t random_seed, std::uint64_t shower_id,
        PhotonPairParameters& sample,
        std::uint32_t& child_count) {
      RandomNumberKey first_key{
          random_seed, shower_id, parent.history_id,
          parent.step_id, static_cast<std::uint32_t>(process_id),
          EmThinningFirstDrawId};
      RandomNumberKey second_key = first_key;
      second_key.draw_id = EmThinningSecondDrawId;
      sample.thinning_first_uniform = uniformOpen01(first_key);
      sample.thinning_second_uniform = uniformOpen01(second_key);
      auto const result = applyEmThinning(
          thinning, parent.energy_GeV, parent.weight,
          first_energy_GeV, second_energy_GeV,
          sample.thinning_first_uniform,
          sample.thinning_second_uniform);
      if (result.status == EmThinningStatus::InvalidInput) {
        return false;
      }
      sample.thinning_status =
          static_cast<std::uint32_t>(result.status);
      sample.thinning_keep_mask = result.keep_mask;
      sample.thinning_first_weight = result.first_weight;
      sample.thinning_second_weight = result.second_weight;
      child_count =
          (result.keep_mask & 0x1U ? 1U : 0U) +
          (result.keep_mask & 0x2U ? 1U : 0U);
      return true;
    }

    __device__ void captureFirstInteraction(
        detail::DeviceFirstInteractionCapture const& capture,
        EmParticleState const& parent, std::int32_t process_id,
        EmParticleState const& first,
        EmParticleState const* second = nullptr) {
      if (parent.generation != 0 || capture.snapshot == nullptr ||
          capture.candidate_count == nullptr) {
        return;
      }
      auto const candidate = atomicAdd(capture.candidate_count, 1U);
      if (candidate != 0) {
        return;
      }
      GpuFirstInteractionSnapshot snapshot{};
      snapshot.parent_at_vertex = parent;
      snapshot.process_id = process_id;
      snapshot.secondary_count = second == nullptr ? 1U : 2U;
      snapshot.secondaries[0] = first;
      snapshot.secondaries[0].weight = parent.weight;
      if (second != nullptr) {
        snapshot.secondaries[1] = *second;
        snapshot.secondaries[1].weight = parent.weight;
      }
      *capture.snapshot = snapshot;
    }

    __global__ void classifyPhotonPairFinalStatesKernel(
        tables::FlatRateTableView table,
        PhotonPairLpmSnapshot lpm_snapshot,
        EmThinningConfig thinning,
        EmInteractionRecord const* interactions, std::size_t count,
        std::uint32_t const* device_input_count,
        std::uint64_t random_seed, std::uint64_t shower_id,
        PhotonPairParameters* parameters,
        ProposalFallbackEvent* raw_fallbacks,
        std::uint32_t* child_counts,
        std::uint32_t* record_flags,
        std::uint32_t* fallback_flags,
        std::uint32_t* continuation_flags,
        std::uint32_t* suppression_flags,
        std::uint32_t* photon_pair_flags,
        std::uint32_t* compton_flags,
        std::uint32_t* photoelectric_flags) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }

      child_counts[index] = 0;
      record_flags[index] = 0;
      fallback_flags[index] = 0;
      continuation_flags[index] = 0;
      suppression_flags[index] = 0;
      photon_pair_flags[index] = 0;
      compton_flags[index] = 0;
      photoelectric_flags[index] = 0;
      if (device_input_count != nullptr &&
          index >= *device_input_count) {
        return;
      }
      auto const& interaction = interactions[index];
      if (interaction.status ==
          EmInteractionStatus::NoDiscreteInteraction) {
        continuation_flags[index] = 1;
        return;
      }
      if (interaction.status != EmInteractionStatus::Selected) {
        raw_fallbacks[index] = invalidFinalState(interaction);
        fallback_flags[index] = 1;
        return;
      }

      auto const capability = gpuProcessCapability(
          interaction.particle.pid, interaction.process_id);
      if (capability != GpuProcessCapability::PhotonPair &&
          capability != GpuProcessCapability::Compton &&
          capability != GpuProcessCapability::Photoelectric) {
        raw_fallbacks[index] = makeProcessFallbackEvent(
            interaction, processFallbackReason(capability));
        fallback_flags[index] = 1;
        return;
      }

      auto const& parent = interaction.particle;
      if (capability == GpuProcessCapability::Compton) {
        double photon_cosine = 0.;
        double electron_cosine = 0.;
        if (parent.generation == 0xffffffffU ||
            parent.step_id == 0xffffffffffffffffULL ||
            !comptonCosines(
                parent.energy_GeV,
                interaction.energy_fraction,
                photon_cosine, electron_cosine)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction);
          fallback_flags[index] = 1;
          return;
        }
        PhotonPairParameters sample{};
        sample.process_id = ComptonProcessId;
        sample.split_fraction =
            interaction.energy_fraction;
        sample.split_uniform =
            interaction.loss_quantile;
        RandomNumberKey const azimuth_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(ComptonProcessId),
            ComptonAzimuthDrawId};
        sample.azimuth_uniform =
            uniformOpen01(azimuth_key);
        std::uint32_t kept_children = 0;
        if (!applyTwoChildThinning(
                thinning, parent, ComptonProcessId,
                parent.energy_GeV *
                    (1. - sample.split_fraction),
                ElectronMassGeV +
                    parent.energy_GeV *
                        sample.split_fraction,
                random_seed, shower_id, sample,
                kept_children)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          fallback_flags[index] = 1;
          return;
        }
        parameters[index] = sample;
        child_counts[index] = kept_children;
        record_flags[index] = 1;
        compton_flags[index] = 1;
        return;
      }

      if (capability ==
          GpuProcessCapability::Photoelectric) {
        double electron_energy_GeV = 0.;
        double kinetic_fraction = 0.;
        if (parent.generation == 0xffffffffU ||
            !::isfinite(interaction.energy_fraction) ||
            ::fabs(interaction.energy_fraction - 1.) >
                1.e-12 ||
            !photoelectricEnergy(
                lpm_snapshot, interaction.component_hash,
                parent.energy_GeV, electron_energy_GeV,
                kinetic_fraction)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction);
          fallback_flags[index] = 1;
          return;
        }
        PhotonPairParameters sample{};
        sample.process_id = PhotoelectricProcessId;
        sample.split_fraction = kinetic_fraction;
        sample.split_uniform =
            interaction.loss_quantile;
        parameters[index] = sample;
        child_counts[index] = 1;
        record_flags[index] = 1;
        photoelectric_flags[index] = 1;
        return;
      }

      if (!::isfinite(parent.energy_GeV) ||
          parent.energy_GeV < 2. * ElectronMassGeV ||
          parent.generation == 0xffffffffU ||
          parent.step_id == 0xffffffffffffffffULL) {
        raw_fallbacks[index] = invalidFinalState(interaction);
        fallback_flags[index] = 1;
        return;
      }

      PhotonPairParameters sample{};
      sample.process_id = PhotonPairProcessId;
      RandomNumberKey const split_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(PhotonPairProcessId),
          PhotonPairSplitDrawId};
      sample.split_uniform = uniformOpen01(split_key);
      sample.split_draw_id = PhotonPairSplitDrawId;
      tables::TableQuery const split_query{
          tables::TableQueryKind::LossFraction, parent.pid,
          PhotonPairFinalStateProcessId, interaction.component_hash,
          parent.energy_GeV * 1000., sample.split_uniform};
      auto const split =
          tables::executeTableQuery(table, split_query);
      bool split_ready = false;
      if (split.status == tables::TableLookupStatus::Success) {
        split_ready = decodePhotonPairNormalizedSplit(
            parent.energy_GeV, split.value,
            sample.split_fraction);
        if (!split_ready) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          fallback_flags[index] = 1;
          return;
        }
      } else if (
          split.status ==
              tables::TableLookupStatus::LossEnergyOutOfRange ||
          (table.physics_source == 1u &&
           split.status ==
               tables::TableLookupStatus::ColumnNotFound)) {
        // The legacy c8emrt payload contains an artificial
        // PhotonPairFinalStateProcessId inverse-CDF column.  That column is
        // deliberately not part of PROPOSAL's native interaction splines.
        // In proposal-native mode use the already validated device
        // rejection sampler and the exported LPM parameters directly.  The
        // same sampler is also the established out-of-domain path for the
        // legacy auxiliary column.
        for (std::uint32_t attempt = 0;
             attempt < PhotonPairAnalyticMaximumAttempts; ++attempt) {
          RandomNumberKey candidate_key{
              random_seed, shower_id, parent.history_id,
              parent.step_id,
              static_cast<std::uint32_t>(PhotonPairProcessId),
              PhotonPairAnalyticCandidateDrawIdBase + attempt};
          auto acceptance_key = candidate_key;
          acceptance_key.draw_id =
              PhotonPairAnalyticAcceptanceDrawIdBase + attempt;
          sample.split_uniform = uniformOpen01(candidate_key);
          sample.split_draw_id = candidate_key.draw_id;
          auto const trial = photonPairFinalStateTrial(
              lpm_snapshot, interaction.component_hash,
              parent.energy_GeV * 1000.,
              sample.split_uniform,
              uniformOpen01(acceptance_key));
          if (trial.status !=
              PhotonPairFinalStateStatus::Success) {
            raw_fallbacks[index] =
                invalidFinalState(interaction, sample);
            fallback_flags[index] = 1;
            return;
          }
          if (trial.accepted != 0) {
            sample.split_fraction = trial.split_fraction;
            split_ready = true;
            break;
          }
        }
        if (!split_ready) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          fallback_flags[index] = 1;
          return;
        }
      } else {
        auto event = makeTableFallbackEvent(
            parent, split_query, split, PhotonPairSplitDrawId,
            interaction.input_index);
        event.energy_fraction = interaction.energy_fraction;
        event.process_id = interaction.process_id;
        auto const native_selection =
            interaction.proposal_selection_random_process_id != 0u;
        event.selection_uniform =
            native_selection
                ? interaction.proposal_selection_uniform
                : interaction.process_uniform;
        event.loss_quantile = interaction.loss_quantile;
        event.outer_acceptance_uniform = interaction.process_uniform;
        event.outer_acceptance_random_process_id =
            interaction.process_random_process_id;
        event.outer_acceptance_draw_id = interaction.process_draw_id;
        event.random_process_id =
            native_selection
                ? interaction.proposal_selection_random_process_id
                : 0u;
        event.random_draw_id =
            native_selection
                ? interaction.proposal_selection_draw_id
                : interaction.loss_draw_id;
        event.final_state_uniform = sample.split_uniform;
        event.final_state_draw_id = PhotonPairSplitDrawId;
        raw_fallbacks[index] = event;
        fallback_flags[index] = 1;
        return;
      }
      auto const electron_energy =
          parent.energy_GeV * sample.split_fraction;
      auto const positron_energy =
          parent.energy_GeV - electron_energy;
      if (!::isfinite(sample.split_fraction) ||
          !(sample.split_fraction > 0.) ||
          !(sample.split_fraction < 1.) ||
          electron_energy < ElectronMassGeV ||
          positron_energy < ElectronMassGeV) {
        raw_fallbacks[index] =
            invalidFinalState(interaction, sample);
        fallback_flags[index] = 1;
        return;
      }

      RandomNumberKey angle_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(PhotonPairProcessId),
          PhotonPairAzimuthDrawId};
      sample.azimuth_uniform = uniformOpen01(angle_key);
      angle_key.draw_id = PhotonPairElectronPolarDrawId;
      sample.electron_polar_uniform = uniformOpen01(angle_key);
      angle_key.draw_id = PhotonPairPositronPolarDrawId;
      sample.positron_polar_uniform = uniformOpen01(angle_key);

      auto const lpm = photonPairLpmSuppressionFactor(
          lpm_snapshot, interaction.component_hash,
          parent.energy_GeV * 1000., sample.split_fraction,
          interaction.mass_density_g_per_cm3);
      if (lpm.status != PhotonPairLpmStatus::Success) {
        auto const reason =
            lpm.status == PhotonPairLpmStatus::InvalidInput
                ? ProposalFallbackReason::InvalidMassDensity
                : ProposalFallbackReason::
                      LpmParametersUnavailable;
        raw_fallbacks[index] =
            makeProcessFallbackEvent(interaction, reason);
        fallback_flags[index] = 1;
        return;
      }
      sample.lpm_survival_probability =
          lpm.survival_probability;
      RandomNumberKey const lpm_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(PhotonPairProcessId),
          PhotonPairLpmDrawId};
      sample.lpm_uniform = uniformOpen01(lpm_key);
      if (sample.lpm_uniform >
          sample.lpm_survival_probability) {
        parameters[index] = sample;
        suppression_flags[index] = 1;
        return;
      }
      std::uint32_t kept_children = 0;
      if (!applyTwoChildThinning(
              thinning, parent, PhotonPairProcessId,
              electron_energy, positron_energy, random_seed,
              shower_id, sample, kept_children)) {
        raw_fallbacks[index] =
            invalidFinalState(interaction, sample);
        fallback_flags[index] = 1;
        return;
      }
      parameters[index] = sample;
      child_counts[index] = kept_children;
      record_flags[index] = 1;
      photon_pair_flags[index] = 1;
    }

    __global__ void writePhotonPairFinalStatesKernel(
        EmInteractionRecord const* interactions,
        PhotonPairParameters const* parameters,
        ProposalFallbackEvent const* raw_fallbacks,
        std::uint32_t const* child_counts,
        std::uint32_t const* child_offsets,
        std::uint32_t const* record_flags,
        std::uint32_t const* record_offsets,
        std::uint32_t const* fallback_flags,
        std::uint32_t const* fallback_offsets,
        std::uint32_t const* continuation_flags,
        std::uint32_t const* continuation_offsets,
        std::uint32_t const* suppression_flags,
        std::uint32_t const* suppression_offsets,
        std::size_t count, std::uint64_t first_history_id,
        PhotonPairFinalStateRecord* compact_records,
        EmParticleState* compact_secondaries,
        ProposalFallbackEvent* compact_fallbacks,
        EmInteractionRecord* compact_continuations,
        PhotonPairLpmSuppressionRecord*
            compact_suppressions,
        std::uint32_t* error_flag,
        detail::DeviceFirstInteractionCapture first_interaction) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      if (fallback_flags[index] != 0) {
        compact_fallbacks[fallback_offsets[index]] =
            raw_fallbacks[index];
        return;
      }
      if (continuation_flags[index] != 0) {
        compact_continuations[continuation_offsets[index]] =
            interactions[index];
        return;
      }
      if (suppression_flags[index] != 0) {
        auto parent = interactions[index].particle;
        ++parent.step_id;
        auto const sample = parameters[index];
        compact_suppressions[suppression_offsets[index]] = {
            parent,
            interactions[index].input_index,
            interactions[index].component_hash,
            sample.lpm_survival_probability,
            sample.lpm_uniform,
            PhotonPairLpmDrawId};
        return;
      }
      if (record_flags[index] == 0) {
        return;
      }

      auto const& interaction = interactions[index];
      auto const parent = interaction.particle;
      auto const sample = parameters[index];
      auto const child_offset = child_offsets[index];
      auto const record_offset = record_offsets[index];
      auto const azimuth =
          sample.azimuth_uniform *
          6.283185307179586476925286766559;

      if (sample.process_id == ComptonProcessId) {
        auto const v = sample.split_fraction;
        double photon_cosine = 0.;
        double electron_cosine = 0.;
        if (!comptonCosines(
                parent.energy_GeV, v, photon_cosine,
                electron_cosine)) {
          atomicCAS(error_flag, 0U, 1U);
          return;
        }
        auto photon = parent;
        photon.energy_GeV =
            parent.energy_GeV * (1. - v);
        photon.parent_history_id = parent.history_id;
        photon.generation = parent.generation + 1;
        photon.step_id = 0;
        photon.reserved = 0;
        photon.weight = sample.thinning_first_weight;
        deflect(parent.direction, photon_cosine, azimuth,
                photon.direction);

        auto electron = parent;
        electron.pid =
            static_cast<std::int32_t>(EmPid::Electron);
        electron.energy_GeV =
            ElectronMassGeV + parent.energy_GeV * v;
        electron.parent_history_id = parent.history_id;
        electron.generation = parent.generation + 1;
        electron.step_id = 0;
        electron.reserved = 0;
        electron.weight = sample.thinning_second_weight;
        deflect(
            parent.direction, electron_cosine,
            ::fmod(
                azimuth +
                    3.1415926535897932384626433832795,
                6.283185307179586476925286766559),
            electron.direction);

        captureFirstInteraction(
            first_interaction, parent, ComptonProcessId,
            photon, &electron);

        std::uint32_t written = 0;
        if ((sample.thinning_keep_mask & 0x1U) != 0) {
          photon.history_id =
              first_history_id + child_offset + written;
          compact_secondaries[child_offset + written] = photon;
          ++written;
        }
        if ((sample.thinning_keep_mask & 0x2U) != 0) {
          electron.history_id =
              first_history_id + child_offset + written;
          compact_secondaries[child_offset + written] =
              electron;
          ++written;
        }
        compact_records[record_offset] =
            PhotonPairFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                written,
                ComptonProcessId,
                v,
                interaction.loss_quantile,
                sample.azimuth_uniform,
                0.,
                0.,
                0.,
                0.,
                interaction.loss_draw_id,
                ComptonAzimuthDrawId,
                0,
                0,
                0};
        auto& output_record = compact_records[record_offset];
        output_record.thinning_status = sample.thinning_status;
        output_record.thinning_keep_mask =
            sample.thinning_keep_mask;
        output_record.thinning_first_uniform =
            sample.thinning_first_uniform;
        output_record.thinning_second_uniform =
            sample.thinning_second_uniform;
        output_record.thinning_first_draw_id =
            EmThinningFirstDrawId;
        output_record.thinning_second_draw_id =
            EmThinningSecondDrawId;
        return;
      }

      if (sample.process_id == PhotoelectricProcessId) {
        auto electron = parent;
        electron.pid =
            static_cast<std::int32_t>(EmPid::Electron);
        electron.energy_GeV =
            ElectronMassGeV +
            parent.energy_GeV * sample.split_fraction;
        electron.parent_history_id = parent.history_id;
        electron.history_id =
            first_history_id + child_offset;
        electron.generation = parent.generation + 1;
        electron.step_id = 0;
        electron.reserved = 0;
        captureFirstInteraction(
            first_interaction, parent, PhotoelectricProcessId,
            electron);
        compact_secondaries[child_offset] = electron;
        compact_records[record_offset] =
            PhotonPairFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                1,
                PhotoelectricProcessId,
                sample.split_fraction,
                interaction.loss_quantile,
                0.,
                0.,
                0.,
                0.,
                0.,
                interaction.loss_draw_id,
                0,
                0,
                0,
                0};
        return;
      }

      auto const electron_energy =
          parent.energy_GeV * sample.split_fraction;
      auto const positron_energy =
          parent.energy_GeV - electron_energy;
      auto const electron_cosine = sauterCosine(
          electron_energy, sample.electron_polar_uniform);
      auto const positron_cosine = sauterCosine(
          positron_energy, sample.positron_polar_uniform);

      auto electron = parent;
      electron.pid = static_cast<std::int32_t>(EmPid::Electron);
      electron.energy_GeV = electron_energy;
      electron.parent_history_id = parent.history_id;
      electron.generation = parent.generation + 1;
      electron.step_id = 0;
      electron.reserved = 0;
      electron.weight = sample.thinning_first_weight;
      deflect(parent.direction, electron_cosine, azimuth,
              electron.direction);

      auto positron = parent;
      positron.pid = static_cast<std::int32_t>(EmPid::Positron);
      positron.energy_GeV = positron_energy;
      positron.parent_history_id = parent.history_id;
      positron.generation = parent.generation + 1;
      positron.step_id = 0;
      positron.reserved = 0;
      positron.weight = sample.thinning_second_weight;
      deflect(parent.direction, positron_cosine,
              ::fmod(azimuth + 3.1415926535897932384626433832795,
                     6.283185307179586476925286766559),
              positron.direction);

      captureFirstInteraction(
          first_interaction, parent, PhotonPairProcessId,
          electron, &positron);

      std::uint32_t written = 0;
      if ((sample.thinning_keep_mask & 0x1U) != 0) {
        electron.history_id =
            first_history_id + child_offset + written;
        compact_secondaries[child_offset + written] = electron;
        ++written;
      }
      if ((sample.thinning_keep_mask & 0x2U) != 0) {
        positron.history_id =
            first_history_id + child_offset + written;
        compact_secondaries[child_offset + written] = positron;
        ++written;
      }
      compact_records[record_offset] = PhotonPairFinalStateRecord{
          interaction.input_index,
          parent.history_id,
          child_offset,
          written,
          PhotonPairProcessId,
          sample.split_fraction,
          sample.split_uniform,
          sample.azimuth_uniform,
          sample.electron_polar_uniform,
          sample.positron_polar_uniform,
          sample.lpm_survival_probability,
          sample.lpm_uniform,
          sample.split_draw_id,
          PhotonPairAzimuthDrawId,
          PhotonPairElectronPolarDrawId,
          PhotonPairPositronPolarDrawId,
          PhotonPairLpmDrawId};
      auto& output_record = compact_records[record_offset];
      output_record.thinning_status = sample.thinning_status;
      output_record.thinning_keep_mask =
          sample.thinning_keep_mask;
      output_record.thinning_first_uniform =
          sample.thinning_first_uniform;
      output_record.thinning_second_uniform =
          sample.thinning_second_uniform;
      output_record.thinning_first_draw_id =
          EmThinningFirstDrawId;
      output_record.thinning_second_draw_id =
          EmThinningSecondDrawId;
    }

    __global__ void finalizePhotonFinalStateSummaryKernel(
        std::uint32_t const* scan_arrays,
        std::size_t count,
        std::uint32_t const* device_input_count,
        bool thinning_enabled,
        std::uint64_t first_secondary_history_id,
        std::uint32_t* summary) {
      if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
      }
      auto const last = count - 1;
      auto compact_count =
          [&](std::size_t pair) -> std::uint64_t {
        return static_cast<std::uint64_t>(
                   scan_arrays[(2 * pair) * count + last]) +
               scan_arrays[(2 * pair + 1) * count + last];
      };
      auto const secondary_count = compact_count(0);
      auto const record_count = compact_count(1);
      auto const fallback_count = compact_count(2);
      auto const continuation_count = compact_count(3);
      auto const suppression_count = compact_count(4);
      auto const photon_pair_count = compact_count(5);
      auto const compton_count = compact_count(6);
      auto const photoelectric_count = compact_count(7);
      auto const input_count =
          device_input_count == nullptr
              ? static_cast<std::uint64_t>(count)
              : static_cast<std::uint64_t>(
                    *device_input_count);

      summary[
          detail::PhotonFinalStateSummaryLayout::
              SecondaryCount] =
          static_cast<std::uint32_t>(secondary_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::GpuCount] =
          static_cast<std::uint32_t>(record_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::
              FallbackCount] =
          static_cast<std::uint32_t>(fallback_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::
              ContinuationCount] =
          static_cast<std::uint32_t>(continuation_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::
              SuppressionCount] =
          static_cast<std::uint32_t>(suppression_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::
              PhotonPairCount] =
          static_cast<std::uint32_t>(photon_pair_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::
              ComptonCount] =
          static_cast<std::uint32_t>(compton_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::
              PhotoelectricCount] =
          static_cast<std::uint32_t>(photoelectric_count);
      summary[
          detail::PhotonFinalStateSummaryLayout::InputCount] =
          static_cast<std::uint32_t>(input_count);

      auto const expected_children =
          2 * (photon_pair_count + compton_count) +
          photoelectric_count;
      if (photon_pair_count + compton_count +
                  photoelectric_count !=
              record_count ||
          secondary_count > expected_children ||
          (!thinning_enabled &&
           secondary_count != expected_children) ||
          record_count + fallback_count +
                  continuation_count + suppression_count !=
              input_count) {
        atomicExch(
            summary +
                detail::PhotonFinalStateSummaryLayout::Error,
            2U);
      }
      if (secondary_count != 0 &&
          secondary_count - 1 >
              (~std::uint64_t{0}) -
                  first_secondary_history_id) {
        atomicExch(
            summary +
                detail::PhotonFinalStateSummaryLayout::Error,
            3U);
      }
    }

    std::size_t scanStorageBytes(std::uint32_t* input,
                                 std::uint32_t* output,
                                 std::size_t count) {
      std::size_t bytes = 0;
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    nullptr, bytes, input, output, count),
                "query photon-pair scan storage");
      return bytes;
    }

    void executeScan(void* temporary, std::size_t bytes,
                     std::uint32_t* input, std::uint32_t* output,
                     std::size_t count, char const* operation) {
      checkCuda(cub::DeviceScan::ExclusiveSum(
                    temporary, bytes, input, output, count),
                operation);
    }

  } // namespace

  namespace detail {

    void appendPhotonPairFinalStateWorkspace(
        WorkspaceSize& required, std::size_t count) {
      auto const scan_bytes =
          scanStorageBytes(nullptr, nullptr, count);
      required.add<PhotonPairParameters>(count);
      required.add<ProposalFallbackEvent>(count);
      required.add<std::uint32_t>(count); // child counts
      required.add<std::uint32_t>(count); // child offsets
      required.add<std::uint32_t>(count); // record flags
      required.add<std::uint32_t>(count); // record offsets
      required.add<std::uint32_t>(count); // fallback flags
      required.add<std::uint32_t>(count); // fallback offsets
      required.add<std::uint32_t>(count); // continuation flags
      required.add<std::uint32_t>(count); // continuation offsets
      required.add<std::uint32_t>(count); // suppression flags
      required.add<std::uint32_t>(count); // suppression offsets
      required.add<std::uint32_t>(count); // photon-pair flags
      required.add<std::uint32_t>(count); // photon-pair offsets
      required.add<std::uint32_t>(count); // Compton flags
      required.add<std::uint32_t>(count); // Compton offsets
      required.add<std::uint32_t>(count); // photoelectric flags
      required.add<std::uint32_t>(count); // photoelectric offsets
      required.add<PhotonPairFinalStateRecord>(count);
      required.add<EmParticleState>(2 * count);
      required.add<ProposalFallbackEvent>(count);
      required.add<EmInteractionRecord>(count);
      required.add<PhotonPairLpmSuppressionRecord>(count);
      required.add<std::uint32_t>(
          PhotonFinalStateSummaryLayout::Size);
      required.addBytes(scan_bytes);
    }

    DevicePhotonPairFinalStateBatch
    launchPhotonPairFinalStateOnDevice(
        tables::FlatRateTableView device_table,
        PhotonPairLpmSnapshot const& lpm_snapshot,
        EmThinningConfig const& thinning,
        EmInteractionRecord const* device_interactions,
        std::size_t count, std::uint64_t random_seed,
        std::uint64_t shower_id,
        std::uint64_t first_secondary_history_id,
        DeviceWorkspace& workspace,
        bool defer_count_download,
        DeviceTransportInteractionBatch*
            deferred_interactions,
        DeviceFirstInteractionCapture const*
            first_interaction) {
      if (count == 0 || device_interactions == nullptr) {
        throw std::invalid_argument(
            "device final-state stage requires interactions");
      }
      if (first_secondary_history_id == 0) {
        throw std::invalid_argument(
            "secondary history IDs must start above zero");
      }
      if (count >
          std::numeric_limits<std::uint32_t>::max() / 2) {
        throw std::length_error(
            "photon-pair batch exceeds 32-bit secondary offsets");
      }
      std::uint32_t const* device_input_count = nullptr;
      if (deferred_interactions != nullptr) {
        if (!deferred_interactions->count_deferred ||
            deferred_interactions->device_interaction_count ==
                nullptr ||
            deferred_interactions->interactions !=
                device_interactions ||
            deferred_interactions->input_count != count) {
          throw std::invalid_argument(
              "deferred photon transport interactions are inconsistent");
        }
        device_input_count =
            deferred_interactions->device_interaction_count;
      }
      auto const scan_bytes =
          scanStorageBytes(nullptr, nullptr, count);
      auto* parameters =
          workspace.acquire<PhotonPairParameters>(count);
      auto* raw_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* child_counts =
          workspace.acquire<std::uint32_t>(count);
      auto* child_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* record_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* record_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* fallback_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* fallback_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* continuation_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* continuation_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* suppression_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* suppression_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* photon_pair_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* photon_pair_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* compton_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* compton_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* photoelectric_flags =
          workspace.acquire<std::uint32_t>(count);
      auto* photoelectric_offsets =
          workspace.acquire<std::uint32_t>(count);
      auto* compact_records =
          workspace.acquire<PhotonPairFinalStateRecord>(count);
      auto* compact_secondaries =
          workspace.acquire<EmParticleState>(2 * count);
      auto* compact_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* compact_continuations =
          workspace.acquire<EmInteractionRecord>(count);
      auto* compact_suppressions =
          workspace.acquire<PhotonPairLpmSuppressionRecord>(count);
      auto* device_summary =
          workspace.acquire<std::uint32_t>(
              PhotonFinalStateSummaryLayout::Size);
      auto* error_flag =
          device_summary +
          PhotonFinalStateSummaryLayout::Error;
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);
      checkCuda(
          cudaMemset(
              device_summary, 0,
              PhotonFinalStateSummaryLayout::Size *
                  sizeof(std::uint32_t)),
          "clear photon final-state summary");

      auto const classification_block_count =
          (count + ClassificationThreadsPerBlock - 1) /
          ClassificationThreadsPerBlock;
      if (classification_block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "photon-pair classification launch is too large");
      }
      auto const classification_blocks =
          static_cast<unsigned int>(
              classification_block_count);
      auto const block_count =
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "photon-pair final-state write launch is too large");
      }
      auto const blocks = static_cast<unsigned int>(block_count);
      classifyPhotonPairFinalStatesKernel
          <<<classification_blocks,
             ClassificationThreadsPerBlock>>>(
          device_table, lpm_snapshot, thinning,
          device_interactions, count, device_input_count,
          random_seed, shower_id,
          parameters, raw_fallbacks,
          child_counts, record_flags, fallback_flags,
          continuation_flags,
          suppression_flags, photon_pair_flags,
          compton_flags, photoelectric_flags);
      checkCuda(cudaGetLastError(),
                "classify photon-pair final states launch");
      executeScan(scan_temporary, scan_bytes, child_counts,
                  child_offsets, count,
                  "scan photon-pair child counts");
      executeScan(scan_temporary, scan_bytes, record_flags,
                  record_offsets, count,
                  "scan photon final-state record flags");
      executeScan(scan_temporary, scan_bytes, fallback_flags,
                  fallback_offsets, count,
                  "scan final-state fallback flags");
      executeScan(scan_temporary, scan_bytes, continuation_flags,
                  continuation_offsets, count,
                  "scan continuation flags");
      executeScan(scan_temporary, scan_bytes, suppression_flags,
                  suppression_offsets, count,
                  "scan LPM suppression flags");
      executeScan(scan_temporary, scan_bytes, photon_pair_flags,
                  photon_pair_offsets, count,
                  "scan photon-pair success flags");
      executeScan(scan_temporary, scan_bytes, compton_flags,
                  compton_offsets, count,
                  "scan Compton success flags");
      executeScan(scan_temporary, scan_bytes,
                  photoelectric_flags, photoelectric_offsets,
                  count,
                  "scan photoelectric success flags");

      // The sixteen flag/offset arrays are consecutive in the arena.  Keep
      // their compact counts device-resident so the production endpoint
      // stage can consume them without a host synchronization.
      finalizePhotonFinalStateSummaryKernel<<<1, 1>>>(
          child_counts, count, device_input_count,
          thinning.enabled != 0,
          first_secondary_history_id, device_summary);
      checkCuda(
          cudaGetLastError(),
          "finalize photon final-state summary launch");

      writePhotonPairFinalStatesKernel<<<blocks, ThreadsPerBlock>>>(
          device_interactions, parameters, raw_fallbacks,
          child_counts, child_offsets, record_flags,
          record_offsets, fallback_flags,
          fallback_offsets, continuation_flags,
          continuation_offsets, suppression_flags,
          suppression_offsets, count, first_secondary_history_id,
          compact_records, compact_secondaries, compact_fallbacks,
          compact_continuations, compact_suppressions,
          error_flag,
          first_interaction == nullptr
              ? DeviceFirstInteractionCapture{}
              : *first_interaction);
      checkCuda(cudaGetLastError(),
                "write photon-pair final states launch");

      std::size_t secondary_count = 0;
      std::size_t record_count = 0;
      std::size_t fallback_count = 0;
      std::size_t continuation_count = 0;
      std::size_t suppression_count = 0;
      std::size_t photon_pair_count = 0;
      std::size_t compton_count = 0;
      std::size_t photoelectric_count = 0;
      if (!defer_count_download) {
        std::array<
            std::uint32_t,
            PhotonFinalStateSummaryLayout::Size>
            host_summary{};
        checkCuda(
            cudaMemcpy(
                host_summary.data(), device_summary,
                sizeof(host_summary),
                cudaMemcpyDeviceToHost),
            "download photon final-state summary");
        secondary_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    SecondaryCount];
        record_count =
            host_summary[
                PhotonFinalStateSummaryLayout::GpuCount];
        fallback_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    FallbackCount];
        continuation_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    ContinuationCount];
        suppression_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    SuppressionCount];
        photon_pair_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    PhotonPairCount];
        compton_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    ComptonCount];
        photoelectric_count =
            host_summary[
                PhotonFinalStateSummaryLayout::
                    PhotoelectricCount];
        auto const input_count =
            static_cast<std::size_t>(
                host_summary[
                    PhotonFinalStateSummaryLayout::
                        InputCount]);
        if (input_count != count) {
          throw std::runtime_error(
              "photon final-state input count changed unexpectedly");
        }
        auto const error =
            host_summary[
                PhotonFinalStateSummaryLayout::Error];
        if (error == 2U) {
          throw std::runtime_error(
              "photon final-state classification lost or duplicated an interaction");
        }
        if (error == 3U) {
          throw std::overflow_error(
              "photon-pair secondary history ID overflow");
        }
        if (error != 0U) {
          throw std::runtime_error(
              "photon final-state generation failed");
        }
      }
      return {
          count,
          record_count,
          photon_pair_count,
          compton_count,
          photoelectric_count,
          secondary_count,
          fallback_count,
          continuation_count,
          suppression_count,
          compact_records,
          compact_secondaries,
          compact_fallbacks,
          compact_continuations,
          compact_suppressions,
          error_flag,
          device_summary,
          defer_count_download};
    }

  } // namespace detail

  EmFinalStateBatchResult generatePhotonPairFinalStatesForValidation(
      tables::FlatRateTableView device_table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace) {
    EmFinalStateBatchResult result{};
    result.input_interactions = interactions.size();
    if (interactions.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "photon-pair final-state CUDA device must be non-negative");
    }
    if (first_secondary_history_id == 0) {
      throw std::invalid_argument(
          "secondary history IDs must start above zero");
    }
    if (interactions.size() >
        std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::length_error(
          "photon-pair batch exceeds 32-bit secondary offsets");
    }
    checkCuda(cudaSetDevice(device),
              "cudaSetDevice(photon-pair final state)");

    auto const count = interactions.size();
    detail::WorkspaceSize required;
    required.add<EmInteractionRecord>(count);
    detail::appendPhotonPairFinalStateWorkspace(required, count);
    workspace.prepare(required.bytes());

    auto* device_interactions =
        workspace.acquire<EmInteractionRecord>(count);
    checkCuda(cudaMemcpy(
                  device_interactions, interactions.data(),
                  count * sizeof(EmInteractionRecord),
                  cudaMemcpyHostToDevice),
              "upload final-state interactions");

    auto const batch =
        detail::launchPhotonPairFinalStateOnDevice(
            device_table, lpm_snapshot, thinning,
            device_interactions, count, random_seed, shower_id,
            first_secondary_history_id, workspace);
    result.gpu_interactions = batch.gpu_interaction_count;
    result.photon_pair_interactions =
        batch.photon_pair_interaction_count;
    result.compton_interactions =
        batch.compton_interaction_count;
    result.photoelectric_interactions =
        batch.photoelectric_interaction_count;
    result.final_state_records.resize(result.gpu_interactions);
    result.secondaries.resize(batch.secondary_count);
    result.fallback_events.resize(batch.fallback_count);
    result.continuations.resize(batch.continuation_count);
    result.lpm_suppressed.resize(batch.suppression_count);
    if (!result.final_state_records.empty()) {
      checkCuda(cudaMemcpy(
                    result.final_state_records.data(),
                    batch.records,
                    result.final_state_records.size() *
                        sizeof(PhotonPairFinalStateRecord),
                    cudaMemcpyDeviceToHost),
                "download photon-pair final-state records");
    }
    if (!result.secondaries.empty()) {
      checkCuda(cudaMemcpy(
                    result.secondaries.data(), batch.secondaries,
                    result.secondaries.size() *
                        sizeof(EmParticleState),
                    cudaMemcpyDeviceToHost),
                "download photon-pair secondaries");
    }
    if (!result.fallback_events.empty()) {
      checkCuda(cudaMemcpy(
                    result.fallback_events.data(),
                    batch.fallbacks,
                    result.fallback_events.size() *
                        sizeof(ProposalFallbackEvent),
                    cudaMemcpyDeviceToHost),
                "download final-state fallbacks");
    }
    if (!result.continuations.empty()) {
      checkCuda(cudaMemcpy(
                    result.continuations.data(),
                    batch.continuations,
                    result.continuations.size() *
                        sizeof(EmInteractionRecord),
                    cudaMemcpyDeviceToHost),
                "download no-interaction continuations");
    }
    if (!result.lpm_suppressed.empty()) {
      checkCuda(cudaMemcpy(
                    result.lpm_suppressed.data(),
                    batch.suppressions,
                    result.lpm_suppressed.size() *
                        sizeof(PhotonPairLpmSuppressionRecord),
                    cudaMemcpyDeviceToHost),
                "download LPM-suppressed photons");
    }
    return result;
  }

} // namespace corsika::gpu::em
