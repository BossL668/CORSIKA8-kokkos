/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <cub/device/device_scan.cuh>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <corsika/gpu/em/CudaBremsFinalState.hpp>
#include <corsika/gpu/em/EpairFinalState.hpp>
#include <corsika/gpu/em/EpairLpm.hpp>
#include <corsika/gpu/em/Philox.hpp>
#include <corsika/gpu/em/ProcessCapabilities.hpp>
#include <corsika/gpu/em/ProposalFallback.hpp>
#include <corsika/gpu/em/detail/DeviceBatchStages.hpp>

namespace corsika::gpu::em {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr unsigned int ClassificationThreadsPerBlock = 64;
    constexpr double TwoPi =
        6.283185307179586476925286766559005768;
    constexpr double Pi =
        3.141592653589793238462643383279502884;
    constexpr double ProposalHalfPrecision = 1.e-5;

    struct BremsParameters {
      std::int32_t process_id{};
      std::uint32_t thinning_status{};
      std::uint32_t thinning_keep_mask{0x3U};
      double final_state_uniform{};
      double energy_split_fraction{};
      double azimuth_uniform{};
      double auxiliary_uniform{};
      double lpm_survival_probability{};
      double lpm_uniform{};
      double thinning_first_uniform{};
      double thinning_second_uniform{};
      double thinning_first_weight{};
      double thinning_second_weight{};
    };

    struct BremsCompactionValue {
      std::uint32_t child_count{};
      std::uint32_t record_count{};
      std::uint32_t fallback_count{};
      std::uint32_t continuation_count{};
      std::uint32_t suppression_count{};

      __host__ __device__ BremsCompactionValue operator+(
          BremsCompactionValue const& other) const {
        return {
            child_count + other.child_count,
            record_count + other.record_count,
            fallback_count + other.fallback_count,
            continuation_count + other.continuation_count,
            suppression_count + other.suppression_count};
      }
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

    /**
     * Add one contribution per currently active lane with a single global
     * atomic operation.  All callers use this for exact integer diagnostics;
     * the active mask may be a process-specific subset of a divergent warp.
     */
    __device__ void warpAggregatedIncrement(
        std::uint32_t* counter) {
      auto const active = __activemask();
      auto const leader = __ffs(active) - 1;
      auto const lane =
          static_cast<int>(threadIdx.x) &
          (warpSize - 1);
      if (lane == leader) {
        atomicAdd(
            counter,
            static_cast<std::uint32_t>(__popc(active)));
      }
    }

    /**
     * Prepare the component-only part of PROPOSAL's BremsLPM formula once per
     * CUDA block.  The original scalar implementation recomputes two pow()
     * calls and log(s1) for every bremsstrahlung vertex even though all
     * particles in the launch share this immutable snapshot.
     */
    __device__ void prepareBremsLpmBlock(
        BremsLpmSnapshot const& snapshot,
        BremsLpmPreparedSnapshot& prepared) {
      if (threadIdx.x == 0) {
        prepared.component_count = snapshot.component_count;
        prepared.valid =
            snapshot.component_count > 0 &&
                    snapshot.component_count <=
                        MaxBremsLpmComponents &&
                    ::isfinite(
                        snapshot
                            .baseline_mass_density_g_per_cm3) &&
                    snapshot.baseline_mass_density_g_per_cm3 >
                        0. &&
                    ::isfinite(
                        snapshot.molecular_density_per_cm3) &&
                    snapshot.molecular_density_per_cm3 > 0. &&
                    ::isfinite(snapshot.sum_charge) &&
                    snapshot.sum_charge > 0. &&
                    ::isfinite(snapshot.e_lpm_MeV) &&
                    snapshot.e_lpm_MeV > 0. &&
                    ::isfinite(snapshot.lepton_mass_MeV) &&
                    snapshot.lepton_mass_MeV > 0. &&
                    ::isfinite(snapshot.electron_mass_MeV) &&
                    snapshot.electron_mass_MeV > 0. &&
                    ::isfinite(snapshot.muon_mass_MeV) &&
                    snapshot.muon_mass_MeV > 0. &&
                    ::isfinite(
                        snapshot
                            .classical_electron_radius_cm) &&
                    snapshot.classical_electron_radius_cm > 0. &&
                    ::isfinite(
                        snapshot.fine_structure_constant) &&
                    snapshot.fine_structure_constant > 0.
                ? 1U
                : 0U;
      }
      __syncthreads();

      auto const component_index =
          static_cast<std::uint32_t>(threadIdx.x);
      if (prepared.valid != 0 &&
          component_index < prepared.component_count) {
        auto const& component =
            snapshot.components[component_index];
        if (!::isfinite(component.nuclear_charge) ||
            !(component.nuclear_charge > 0.) ||
            !::isfinite(component.atomic_mass_number) ||
            !(component.atomic_mass_number > 0.) ||
            !::isfinite(component.radiation_log_constant) ||
            !(component.radiation_log_constant > 0.)) {
          atomicExch(&prepared.valid, 0U);
        } else {
          constexpr double SqrtTwo =
              1.414213562373095048801688724209698079;
          auto const z_to_minus_third =
              ::pow(component.nuclear_charge, -1. / 3.);
          auto const d_n =
              1.54 *
              ::pow(component.atomic_mass_number, 0.27);
          auto const nuclear_auxiliary =
              snapshot.lepton_mass_MeV /
              snapshot.muon_mass_MeV * d_n;
          auto s1 =
              snapshot.electron_mass_MeV /
              (snapshot.lepton_mass_MeV *
               z_to_minus_third *
               component.radiation_log_constant);
          s1 =
              s1 * s1 *
              (1. +
               nuclear_auxiliary * nuclear_auxiliary) *
              SqrtTwo;
          prepared.components[component_index] = {
              component.component_hash, s1, ::log(s1)};
        }
      }
      __syncthreads();
    }

    __device__ BremsLpmResult
    bremsLpmSuppressionFactorPrepared(
        BremsLpmSnapshot const& snapshot,
        BremsLpmPreparedSnapshot const& prepared,
        std::uint64_t component_hash, double energy_MeV,
        double v, double local_mass_density_g_per_cm3) {
      if (prepared.valid == 0) {
        return {
            BremsLpmStatus::InvalidSnapshot, 0, 0.};
      }
      if (!::isfinite(energy_MeV) ||
          !(energy_MeV > snapshot.lepton_mass_MeV) ||
          !::isfinite(v) || !(v > 0.) || !(v < 1.) ||
          !::isfinite(local_mass_density_g_per_cm3) ||
          !(local_mass_density_g_per_cm3 > 0.)) {
        return {BremsLpmStatus::InvalidInput, 0, 0.};
      }

      BremsLpmPreparedComponent component{};
      bool found = false;
      for (std::uint32_t component_index = 0;
           component_index < prepared.component_count;
           ++component_index) {
        if (prepared.components[component_index]
                .component_hash == component_hash) {
          component =
              prepared.components[component_index];
          found = true;
          break;
        }
      }
      if (!found) {
        return {
            BremsLpmStatus::ComponentNotFound, 0, 0.};
      }

      constexpr double Fi1 = 1.54954;
      constexpr double G1 = 0.710390;
      constexpr double G2 = 0.904912;
      auto const density_correction =
          local_mass_density_g_per_cm3 /
          snapshot.baseline_mass_density_g_per_cm3;
      auto const sp =
          0.125 *
          ::sqrt(
              snapshot.e_lpm_MeV * v /
              (density_correction * energy_MeV *
               (1. - v)));
      auto const h =
          ::log(sp) / component.logarithm_s1;
      double xi = 1.;
      if (sp < component.s1) {
        xi = 2.;
      } else if (sp < 1.) {
        xi =
            1. + h -
            0.08 * (1. - h) *
                (1. - (1. - h) * (1. - h)) /
                component.logarithm_s1;
      }

      auto gamma =
          snapshot.classical_electron_radius_cm *
          snapshot.electron_mass_MeV /
          (snapshot.fine_structure_constant *
           snapshot.lepton_mass_MeV * v);
      gamma =
          1. +
          4. * Pi * snapshot.sum_charge *
              snapshot.classical_electron_radius_cm *
              gamma * gamma *
              snapshot.molecular_density_per_cm3 *
              density_correction;
      auto const s = sp / ::sqrt(xi) * gamma;
      auto const s2 = s * s;

      double fi = 0.;
      if (s < Fi1) {
        fi =
            1. -
            ::exp(
                -6. * s * (1. + (3. - Pi) * s) +
                s2 * s /
                    (0.623 + 0.796 * s + 0.658 * s2));
      } else {
        fi = 1. - 0.012 / (s2 * s2);
      }

      double g = 0.;
      if (s < G1) {
        auto const psi =
            1. -
            ::exp(
                -4. * s -
                8. * s2 /
                    (1. + 3.936 * s + 4.97 * s2 -
                     0.05 * s2 * s +
                     7.50 * s2 * s2));
        g = 3. * psi - 2. * fi;
      } else if (s < G2) {
        g = 36. * s2 / (36. * s2 + 1.);
      } else {
        g = 1. - 0.022 / (s2 * s2);
      }

      auto const probability =
          ((xi / 3.) *
           (v * v * g / (gamma * gamma) +
            2. * (1. + (1. - v) * (1. - v)) * fi /
                gamma)) /
          ((4. / 3.) * (1. - v) + v * v);
      if (!::isfinite(probability) || probability < 0.) {
        return {
            BremsLpmStatus::NonFiniteResult, 0, 0.};
      }
      return {
          BremsLpmStatus::Success, 0, probability};
    }

    __global__ void prepareBremsLpmSnapshotKernel(
        BremsLpmSnapshot snapshot,
        BremsLpmPreparedSnapshot* output) {
      __shared__ BremsLpmPreparedSnapshot prepared;
      prepareBremsLpmBlock(snapshot, prepared);
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = prepared;
      }
    }

    __device__ void deflect(
        double const input[3], double cosine, double azimuth,
        double output[3]) {
      auto const transverse =
          ::sqrt(::fmax(
              0., input[0] * input[0] +
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
      double const rotation_y[3]{
          -sine_phi, cosine_phi, 0.};

      auto const bounded_cosine =
          ::fmax(-1., ::fmin(1., cosine));
      auto const sine =
          ::sqrt(::fmax(
              0., (1. - bounded_cosine) *
                      (1. + bounded_cosine)));
      auto const local_x = sine * ::cos(azimuth);
      auto const local_y = sine * ::sin(azimuth);
      auto local_z = ::sqrt(::fmax(
          0., 1. - local_x * local_x -
                  local_y * local_y));
      if (bounded_cosine < 0.) {
        local_z = -local_z;
      }
      for (int axis = 0; axis < 3; ++axis) {
        output[axis] =
            local_z * input[axis] +
            local_x * rotation_x[axis] +
            local_y * rotation_y[axis];
      }
    }

    __device__ double annihilationFunctionTerm(
        double a1, double a2, double rho) {
      return a1 * ::log(rho) + a2 / rho - rho;
    }

    __device__ int sign(double value) {
      return (0. < value) - (value < 0.);
    }

    /**
     * Device-equivalent implementation of PROPOSAL 7.6.2
     * HeitlerAnnihilation::CalculateRho and MathMethods::Bisection.
     *
     * PROPOSAL returns the lower member of the final bracket, so this function
     * deliberately does the same instead of returning the midpoint.
     */
    __device__ bool sampleAnnihilationRho(
        double energy_GeV, double mass_GeV, double uniform,
        double& rho) {
      auto const gamma = energy_GeV / mass_GeV;
      if (!::isfinite(gamma) || !(gamma > 1.) ||
          !::isfinite(uniform) || !(uniform > 0.) ||
          !(uniform < 1.)) {
        return false;
      }
      auto const auxiliary =
          ::sqrt((gamma - 1.) / (gamma + 1.));
      auto lower = 0.5 * (1. - auxiliary);
      auto upper = 0.5 * (1. + auxiliary);
      auto const a2 = 1. / ((gamma + 1.) * (gamma + 1.));
      auto const a1 = 1. + 2. * gamma * a2;
      auto const lower_term =
          annihilationFunctionTerm(a1, a2, lower);
      auto const upper_term =
          annihilationFunctionTerm(a1, a2, upper);
      auto const evaluate =
          [&](double value) {
            return annihilationFunctionTerm(a1, a2, value) -
                   lower_term -
                   uniform * (upper_term - lower_term);
          };
      auto const initial_lower_value = evaluate(lower);
      auto const initial_upper_value = evaluate(upper);
      if (!::isfinite(initial_lower_value) ||
          !::isfinite(initial_upper_value) ||
          initial_lower_value * initial_upper_value > 0.) {
        return false;
      }
      auto const precision =
          lower * (energy_GeV * 1000.) *
          ProposalHalfPrecision;
      if (!::isfinite(precision) || !(precision > 0.)) {
        return false;
      }
      for (int iteration = 0; iteration <= 100; ++iteration) {
        auto const center = (lower + upper) / 2.;
        if (sign(evaluate(center)) == sign(evaluate(lower))) {
          lower = center;
        } else {
          upper = center;
        }
        if (::fabs(upper - lower) < precision) {
          rho = lower;
          return ::isfinite(rho) && rho > 0. && rho < 1.;
        }
      }
      rho = lower;
      return ::isfinite(rho) && rho > 0. && rho < 1.;
    }

    __device__ bool normalize(double vector[3]) {
      auto const norm_squared =
          vector[0] * vector[0] +
          vector[1] * vector[1] +
          vector[2] * vector[2];
      if (!::isfinite(norm_squared) || !(norm_squared > 0.)) {
        return false;
      }
      auto const inverse_norm = 1. / ::sqrt(norm_squared);
      for (int axis = 0; axis < 3; ++axis) {
        vector[axis] *= inverse_norm;
      }
      return ::isfinite(vector[0]) &&
             ::isfinite(vector[1]) &&
             ::isfinite(vector[2]);
    }

    __device__ double proposalEffectiveLossFraction(
        double energy_GeV, double sampled_fraction) {
      auto const energy_MeV = energy_GeV * 1000.;
      auto const loss_MeV = sampled_fraction * energy_MeV;
      return loss_MeV / energy_MeV;
    }

    __device__ ProposalFallbackEvent invalidFinalState(
        EmInteractionRecord const& interaction,
        BremsParameters const& parameters = {}) {
      auto event = makeProcessFallbackEvent(
          interaction,
          ProposalFallbackReason::InvalidFinalState);
      if (parameters.final_state_uniform > 0.) {
        event.final_state_uniform =
            parameters.final_state_uniform;
        event.final_state_draw_id =
            parameters.process_id == AnnihilationProcessId
                ? AnnihilationRhoDrawId
                : parameters.process_id ==
                          ElectronPairProcessId
                      ? EpairRhoDrawId
                      : BremsAzimuthDrawId;
      } else if (parameters.azimuth_uniform > 0.) {
        event.final_state_uniform =
            parameters.azimuth_uniform;
        event.final_state_draw_id =
            parameters.process_id == AnnihilationProcessId
                ? AnnihilationAzimuthDrawId
                : parameters.process_id ==
                          ElectronPairProcessId
                      ? EpairSignDrawId
                      : BremsAzimuthDrawId;
      }
      return event;
    }

    __device__ bool applyTwoChildThinning(
        EmThinningConfig const& thinning,
        EmParticleState const& parent, std::int32_t process_id,
        double first_energy_GeV, double second_energy_GeV,
        std::uint64_t random_seed, std::uint64_t shower_id,
        BremsParameters& sample,
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

    __global__ void classifyBremsFinalStatesKernel(
        BremsLpmSnapshot lpm_snapshot,
        BremsLpmPreparedSnapshot prepared_lpm_input,
        EmThinningConfig thinning,
        EmInteractionRecord const* interactions,
        std::size_t count,
        std::uint32_t const* device_input_count,
        std::uint64_t random_seed,
        std::uint64_t shower_id,
        BremsParameters* parameters,
        ProposalFallbackEvent* raw_fallbacks,
        BremsCompactionValue* classifications,
        std::uint32_t* summary) {
      __shared__ BremsLpmPreparedSnapshot prepared_lpm;
      if (threadIdx.x == 0) {
        prepared_lpm.valid = prepared_lpm_input.valid;
        prepared_lpm.component_count =
            prepared_lpm_input.component_count;
      }
      auto const prepared_component =
          static_cast<std::uint32_t>(threadIdx.x);
      if (prepared_component <
          prepared_lpm_input.component_count) {
        prepared_lpm.components[prepared_component] =
            prepared_lpm_input.components[prepared_component];
      }
      __syncthreads();

      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }

      auto& classification = classifications[index];
      classification = {};
      if (device_input_count != nullptr &&
          index >= *device_input_count) {
        return;
      }
      auto const& interaction = interactions[index];
      if (interaction.status ==
          EmInteractionStatus::NoDiscreteInteraction) {
        classification.continuation_count = 1;
        return;
      }
      if (interaction.status != EmInteractionStatus::Selected) {
        raw_fallbacks[index] =
            invalidFinalState(interaction);
        classification.fallback_count = 1;
        return;
      }

      auto const capability = gpuProcessCapability(
          interaction.particle.pid, interaction.process_id);
      if (capability !=
              GpuProcessCapability::Bremsstrahlung &&
          capability != GpuProcessCapability::Annihilation &&
          capability != GpuProcessCapability::Ionization &&
          capability != GpuProcessCapability::ElectronPair) {
        raw_fallbacks[index] = makeProcessFallbackEvent(
            interaction, processFallbackReason(capability));
        classification.fallback_count = 1;
        return;
      }

      auto const& parent = interaction.particle;
      auto const v = interaction.energy_fraction;
      auto const electron_mass_GeV =
          lpm_snapshot.lepton_mass_MeV / 1000.;
      auto const lepton_mass_GeV =
          interaction.particle_mass_GeV > 0.
              ? interaction.particle_mass_GeV
              : electron_mass_GeV;
      if (!::isfinite(parent.energy_GeV) ||
          !(parent.energy_GeV >
            lepton_mass_GeV) ||
          parent.generation == 0xffffffffU ||
          parent.step_id == 0xffffffffffffffffULL) {
        raw_fallbacks[index] =
            invalidFinalState(interaction);
        classification.fallback_count = 1;
        return;
      }

      BremsParameters sample{};
      sample.process_id = interaction.process_id;
      if (capability == GpuProcessCapability::Annihilation) {
        if (!::isfinite(v) || ::fabs(v - 1.) > 1.e-12) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          classification.fallback_count = 1;
          return;
        }
        RandomNumberKey const rho_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                AnnihilationProcessId),
            AnnihilationRhoDrawId};
        RandomNumberKey const azimuth_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                AnnihilationProcessId),
            AnnihilationAzimuthDrawId};
        sample.final_state_uniform = uniformOpen01(rho_key);
        sample.azimuth_uniform =
            uniformOpen01(azimuth_key);
        if (!sampleAnnihilationRho(
                parent.energy_GeV, lepton_mass_GeV,
                sample.final_state_uniform,
                sample.energy_split_fraction)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          classification.fallback_count = 1;
          return;
        }
        auto const total_photon_energy =
            parent.energy_GeV + lepton_mass_GeV;
        std::uint32_t kept_children = 0;
        if (!applyTwoChildThinning(
                thinning, parent, AnnihilationProcessId,
                total_photon_energy *
                    (1. - sample.energy_split_fraction),
                total_photon_energy *
                    sample.energy_split_fraction,
                random_seed, shower_id, sample,
                kept_children)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          classification.fallback_count = 1;
          return;
        }
        parameters[index] = sample;
        classification.child_count = kept_children;
        classification.record_count = 1;
        warpAggregatedIncrement(
            summary +
                detail::BremsFinalStateSummaryLayout::
                    AnnihilationCount);
        return;
      }

      if (capability == GpuProcessCapability::Ionization) {
        auto const effective_v =
            proposalEffectiveLossFraction(
                parent.energy_GeV, v);
        auto const outgoing_energy =
            parent.energy_GeV * (1. - effective_v);
        auto const delta_energy =
            parent.energy_GeV * effective_v +
            electron_mass_GeV;
        if (!::isfinite(effective_v) ||
            !(effective_v > 0.) ||
            !(effective_v < 1.) ||
            !::isfinite(outgoing_energy) ||
            outgoing_energy < lepton_mass_GeV ||
            !::isfinite(delta_energy) ||
            !(delta_energy > electron_mass_GeV)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          classification.fallback_count = 1;
          return;
        }
        RandomNumberKey const azimuth_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                IonizationProcessId),
            IonizationAzimuthDrawId};
        sample.azimuth_uniform =
            uniformOpen01(azimuth_key);
        std::uint32_t kept_children = 0;
        if (!applyTwoChildThinning(
                thinning, parent, IonizationProcessId,
                outgoing_energy, delta_energy, random_seed,
                shower_id, sample, kept_children)) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          classification.fallback_count = 1;
          return;
        }
        parameters[index] = sample;
        classification.child_count = kept_children;
        classification.record_count = 1;
        warpAggregatedIncrement(
            summary +
                detail::BremsFinalStateSummaryLayout::
                    IonizationCount);
        return;
      }

      if (capability == GpuProcessCapability::ElectronPair) {
        auto const effective_v =
            proposalEffectiveLossFraction(
                parent.energy_GeV, v);
        auto const loss_energy =
            parent.energy_GeV * effective_v;
        auto const surviving_energy =
            parent.energy_GeV - loss_energy;
        if (!::isfinite(effective_v) ||
            !(effective_v > 0.) || !(effective_v < 1.) ||
            !::isfinite(loss_energy) ||
            !(loss_energy >= 4. * lepton_mass_GeV) ||
            !::isfinite(surviving_energy) ||
            surviving_energy < lepton_mass_GeV) {
          raw_fallbacks[index] =
              invalidFinalState(interaction, sample);
          classification.fallback_count = 1;
          return;
        }
        RandomNumberKey const rho_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                ElectronPairProcessId),
            EpairRhoDrawId};
        RandomNumberKey const sign_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                ElectronPairProcessId),
            EpairSignDrawId};
        RandomNumberKey const direction_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                ElectronPairProcessId),
            EpairDirectionDrawId};
        sample.final_state_uniform = uniformOpen01(rho_key);
        sample.azimuth_uniform = uniformOpen01(sign_key);
        sample.auxiliary_uniform =
            uniformOpen01(direction_key);
        auto const rejection =
            sampleEpairRhoRejection(
            lpm_snapshot, interaction.component_hash,
            parent.energy_GeV * 1000., effective_v,
            sample.azimuth_uniform, rho_key);
        auto const& split = rejection.sample;
        if (rejection.trial_count != 0) {
          atomicAdd(
              summary +
                  detail::BremsFinalStateSummaryLayout::
                      ElectronPairRejectionTrials,
              rejection.trial_count);
        } else if (
            split.status == EpairFinalStateStatus::Success &&
            split.rho == 0.) {
          warpAggregatedIncrement(
              summary +
                  detail::BremsFinalStateSummaryLayout::
                      ElectronPairZeroWeightSamples);
        }
        if (split.status != EpairFinalStateStatus::Success) {
          warpAggregatedIncrement(
              summary +
                  detail::BremsFinalStateSummaryLayout::
                      ElectronPairRejectionFallbacks);
          if (split.status ==
              EpairFinalStateStatus::
                  RejectionEnvelopeExceeded) {
            warpAggregatedIncrement(
                summary +
                    detail::BremsFinalStateSummaryLayout::
                        ElectronPairEnvelopeViolations);
          }
          auto reason = ProposalFallbackReason::InvalidFinalState;
          if (split.status ==
                  EpairFinalStateStatus::ComponentNotFound ||
              split.status ==
                  EpairFinalStateStatus::InvalidSnapshot) {
            reason =
                ProposalFallbackReason::LpmParametersUnavailable;
          } else if (
              split.status ==
              EpairFinalStateStatus::RejectionEnvelopeExceeded) {
            reason = ProposalFallbackReason::
                EpairRejectionEnvelopeExceeded;
          }
          raw_fallbacks[index] =
              makeProcessFallbackEvent(interaction, reason);
          raw_fallbacks[index].diagnostic_status =
              static_cast<std::int32_t>(split.status);
          raw_fallbacks[index].diagnostic_value0 =
              split.rho_max;
          raw_fallbacks[index].diagnostic_value1 =
              static_cast<double>(rejection.trial_count);
          raw_fallbacks[index].final_state_uniform =
              sample.final_state_uniform;
          raw_fallbacks[index].final_state_draw_id =
              EpairRhoDrawId;
          classification.fallback_count = 1;
          return;
        }
        sample.energy_split_fraction = split.rho;
        auto const lpm = epairLpmSuppressionFactor(
            lpm_snapshot, parent.energy_GeV * 1000.,
            effective_v, split.rho * split.rho,
            interaction.mass_density_g_per_cm3);
        if (lpm.status != EpairLpmStatus::Success) {
          auto const reason =
              lpm.status == EpairLpmStatus::InvalidInput
                  ? ProposalFallbackReason::InvalidMassDensity
                  : ProposalFallbackReason::
                        LpmParametersUnavailable;
          raw_fallbacks[index] =
              makeProcessFallbackEvent(interaction, reason);
          raw_fallbacks[index].final_state_uniform =
              sample.final_state_uniform;
          raw_fallbacks[index].final_state_draw_id =
              EpairRhoDrawId;
          classification.fallback_count = 1;
          return;
        }
        sample.lpm_survival_probability =
            lpm.survival_probability;
        RandomNumberKey const lpm_key{
            random_seed, shower_id, parent.history_id,
            parent.step_id,
            static_cast<std::uint32_t>(
                ElectronPairProcessId),
            EpairLpmDrawId};
        sample.lpm_uniform = uniformOpen01(lpm_key);
        parameters[index] = sample;
        if (sample.lpm_uniform >
            sample.lpm_survival_probability) {
          classification.suppression_count = 1;
          warpAggregatedIncrement(
              summary +
                  detail::BremsFinalStateSummaryLayout::
                      ElectronPairSuppressionCount);
          return;
        }
        classification.child_count = 3;
        classification.record_count = 1;
        warpAggregatedIncrement(
            summary +
                detail::BremsFinalStateSummaryLayout::
                    ElectronPairCount);
        return;
      }

      auto const photon_energy = parent.energy_GeV * v;
      auto const lepton_energy =
          parent.energy_GeV - photon_energy;
      if (!::isfinite(v) || !(v > 0.) || !(v < 1.) ||
          !::isfinite(photon_energy) ||
          !(photon_energy > 0.) ||
          !::isfinite(lepton_energy) ||
          lepton_energy < lepton_mass_GeV) {
        raw_fallbacks[index] =
            invalidFinalState(interaction, sample);
        classification.fallback_count = 1;
        return;
      }
      RandomNumberKey const azimuth_key{
          random_seed, shower_id, parent.history_id,
          parent.step_id,
          static_cast<std::uint32_t>(BremsProcessId),
          BremsAzimuthDrawId};
      sample.azimuth_uniform = uniformOpen01(azimuth_key);
      auto const lpm = bremsLpmSuppressionFactorPrepared(
          lpm_snapshot, prepared_lpm,
          interaction.component_hash,
          parent.energy_GeV * 1000., v,
          interaction.mass_density_g_per_cm3);
      if (lpm.status != BremsLpmStatus::Success) {
        auto const reason =
            lpm.status == BremsLpmStatus::InvalidInput
                ? ProposalFallbackReason::InvalidMassDensity
                : ProposalFallbackReason::
                      LpmParametersUnavailable;
        raw_fallbacks[index] =
            makeProcessFallbackEvent(interaction, reason);
        raw_fallbacks[index].final_state_uniform =
            sample.azimuth_uniform;
        raw_fallbacks[index].final_state_draw_id =
            BremsAzimuthDrawId;
        classification.fallback_count = 1;
        return;
      }
      sample.lpm_survival_probability =
          lpm.survival_probability;
      RandomNumberKey const lpm_key{
          random_seed, shower_id, parent.history_id,
          parent.step_id,
          static_cast<std::uint32_t>(BremsProcessId),
          BremsLpmDrawId};
      sample.lpm_uniform = uniformOpen01(lpm_key);
      if (sample.lpm_uniform >
          sample.lpm_survival_probability) {
        parameters[index] = sample;
        classification.suppression_count = 1;
        warpAggregatedIncrement(
            summary +
                detail::BremsFinalStateSummaryLayout::
                    BremsSuppressionCount);
        return;
      }
      std::uint32_t kept_children = 0;
      if (!applyTwoChildThinning(
              thinning, parent, BremsProcessId, lepton_energy,
              photon_energy, random_seed, shower_id, sample,
              kept_children)) {
        raw_fallbacks[index] =
            invalidFinalState(interaction, sample);
        classification.fallback_count = 1;
        return;
      }
      parameters[index] = sample;
      classification.child_count = kept_children;
      classification.record_count = 1;
      warpAggregatedIncrement(
          summary +
              detail::BremsFinalStateSummaryLayout::BremsCount);
    }

    __global__ void writeBremsFinalStatesKernel(
        EmInteractionRecord const* interactions,
        BremsParameters const* parameters,
        ProposalFallbackEvent const* raw_fallbacks,
        BremsCompactionValue const* classifications,
        BremsCompactionValue const* offsets,
        std::size_t count, std::uint64_t first_history_id,
        double electron_mass_GeV,
        BremsFinalStateRecord* compact_records,
        EmParticleState* compact_secondaries,
        ProposalFallbackEvent* compact_fallbacks,
        EmInteractionRecord* compact_continuations,
        BremsLpmSuppressionRecord* compact_suppressions,
        std::uint32_t* error_flag) {
      auto const index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (index >= count) {
        return;
      }
      auto const classification = classifications[index];
      auto const offset = offsets[index];
      if (classification.fallback_count != 0) {
        compact_fallbacks[offset.fallback_count] =
            raw_fallbacks[index];
        return;
      }
      if (classification.continuation_count != 0) {
        compact_continuations[
            offset.continuation_count] =
            interactions[index];
        return;
      }
      if (classification.suppression_count != 0) {
        auto parent = interactions[index].particle;
        ++parent.step_id;
        auto const sample = parameters[index];
        compact_suppressions[offset.suppression_count] = {
            parent,
            interactions[index].input_index,
            interactions[index].component_hash,
            sample.lpm_survival_probability,
            sample.lpm_uniform,
            sample.process_id == ElectronPairProcessId
                ? EpairLpmDrawId
                : BremsLpmDrawId};
        return;
      }
      if (classification.record_count == 0) {
        return;
      }

      auto const& interaction = interactions[index];
      auto const parent = interaction.particle;
      auto const sample = parameters[index];
      auto const lepton_mass_GeV =
          interaction.particle_mass_GeV > 0.
              ? interaction.particle_mass_GeV
              : electron_mass_GeV;
      auto const child_offset = offset.child_count;
      auto const record_offset = offset.record_count;
      if (sample.process_id == AnnihilationProcessId) {
        auto const rho = sample.energy_split_fraction;
        auto const total_photon_energy =
            parent.energy_GeV + lepton_mass_GeV;
        auto const first_energy =
            total_photon_energy * (1. - rho);
        auto const second_energy =
            total_photon_energy * rho;
        auto const momentum =
            ::sqrt(
                (parent.energy_GeV + lepton_mass_GeV) *
                (parent.energy_GeV - lepton_mass_GeV));
        auto const first_cosine =
            (total_photon_energy * (1. - rho) -
             lepton_mass_GeV) /
            ((1. - rho) * momentum);
        auto const second_cosine =
            (total_photon_energy * rho -
             lepton_mass_GeV) /
            (rho * momentum);
        if (!::isfinite(first_energy) ||
            !::isfinite(second_energy) ||
            !(first_energy > 0.) || !(second_energy > 0.) ||
            !::isfinite(first_cosine) ||
            !::isfinite(second_cosine)) {
          atomicExch(error_flag, 1U);
          return;
        }
        auto const azimuth =
            sample.azimuth_uniform * TwoPi;
        double first_direction[3]{};
        double second_direction[3]{};
        deflect(
            parent.direction, first_cosine, azimuth,
            first_direction);
        auto second_azimuth = ::fmod(azimuth + Pi, TwoPi);
        deflect(
            parent.direction, second_cosine,
            second_azimuth, second_direction);

        auto first = parent;
        first.pid = static_cast<std::int32_t>(EmPid::Photon);
        first.energy_GeV = first_energy;
        first.parent_history_id = parent.history_id;
        first.generation = parent.generation + 1;
        first.step_id = 0;
        first.reserved = 0;
        first.weight = sample.thinning_first_weight;
        auto second = first;
        second.energy_GeV = second_energy;
        second.weight = sample.thinning_second_weight;
        for (int axis = 0; axis < 3; ++axis) {
          first.direction[axis] = first_direction[axis];
          second.direction[axis] = second_direction[axis];
        }
        std::uint32_t written = 0;
        if ((sample.thinning_keep_mask & 0x1U) != 0) {
          first.history_id =
              first_history_id + child_offset + written;
          compact_secondaries[child_offset + written] = first;
          ++written;
        }
        if ((sample.thinning_keep_mask & 0x2U) != 0) {
          second.history_id =
              first_history_id + child_offset + written;
          compact_secondaries[child_offset + written] = second;
          ++written;
        }
        compact_records[record_offset] =
            BremsFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                written,
                AnnihilationProcessId,
                rho,
                sample.final_state_uniform,
                sample.azimuth_uniform,
                0.,
                1.,
                0.,
                AnnihilationRhoDrawId,
                AnnihilationAzimuthDrawId,
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

      if (sample.process_id == IonizationProcessId) {
        auto const v = proposalEffectiveLossFraction(
            parent.energy_GeV,
            interaction.energy_fraction);
        auto const outgoing_energy =
            parent.energy_GeV * (1. - v);
        auto const delta_energy =
            parent.energy_GeV * v + electron_mass_GeV;
        auto const incoming_momentum =
            ::sqrt(
                (parent.energy_GeV + lepton_mass_GeV) *
                (parent.energy_GeV - lepton_mass_GeV));
        auto const outgoing_momentum =
            ::sqrt(
                (outgoing_energy + lepton_mass_GeV) *
                (outgoing_energy - lepton_mass_GeV));
        auto const delta_momentum =
            ::sqrt(
                (delta_energy + electron_mass_GeV) *
                (delta_energy - electron_mass_GeV));
        auto const outgoing_cosine =
            ((parent.energy_GeV + electron_mass_GeV) *
                 outgoing_energy -
             parent.energy_GeV * electron_mass_GeV -
             lepton_mass_GeV * lepton_mass_GeV) /
            (incoming_momentum * outgoing_momentum);
        auto const delta_cosine =
            ((parent.energy_GeV + electron_mass_GeV) *
                 delta_energy -
             parent.energy_GeV * electron_mass_GeV -
             electron_mass_GeV * electron_mass_GeV) /
            (incoming_momentum * delta_momentum);
        if (!::isfinite(outgoing_cosine) ||
            !::isfinite(delta_cosine)) {
          atomicExch(error_flag, 1U);
          return;
        }
        auto const azimuth =
            sample.azimuth_uniform * TwoPi;
        double outgoing_direction[3]{};
        double delta_direction[3]{};
        deflect(
            parent.direction, outgoing_cosine, azimuth,
            outgoing_direction);
        deflect(
            parent.direction, delta_cosine,
            ::fmod(azimuth + Pi, TwoPi),
            delta_direction);

        auto outgoing = parent;
        outgoing.energy_GeV = outgoing_energy;
        outgoing.parent_history_id = parent.history_id;
        outgoing.generation = parent.generation + 1;
        outgoing.step_id = 0;
        outgoing.reserved = 0;
        outgoing.weight = sample.thinning_first_weight;
        auto delta = outgoing;
        delta.pid =
            static_cast<std::int32_t>(EmPid::Electron);
        delta.energy_GeV = delta_energy;
        delta.weight = sample.thinning_second_weight;
        for (int axis = 0; axis < 3; ++axis) {
          outgoing.direction[axis] =
              outgoing_direction[axis];
          delta.direction[axis] = delta_direction[axis];
        }
        std::uint32_t written = 0;
        if ((sample.thinning_keep_mask & 0x1U) != 0) {
          outgoing.history_id =
              first_history_id + child_offset + written;
          compact_secondaries[child_offset + written] =
              outgoing;
          ++written;
        }
        if ((sample.thinning_keep_mask & 0x2U) != 0) {
          delta.history_id =
              first_history_id + child_offset + written;
          compact_secondaries[child_offset + written] = delta;
          ++written;
        }
        compact_records[record_offset] =
            BremsFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                written,
                IonizationProcessId,
                interaction.energy_fraction,
                0.,
                sample.azimuth_uniform,
                0.,
                1.,
                0.,
                0,
                IonizationAzimuthDrawId,
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

      if (sample.process_id == ElectronPairProcessId) {
        auto const v = proposalEffectiveLossFraction(
            parent.energy_GeV,
            interaction.energy_fraction);
        auto const loss_energy = parent.energy_GeV * v;
        auto const rho = sample.energy_split_fraction;
        auto const surviving_energy =
            parent.energy_GeV - loss_energy;
        auto const electron_energy =
            0.5 * loss_energy * (1. + rho);
        auto const positron_energy =
            0.5 * loss_energy * (1. - rho);
        if (!::isfinite(surviving_energy) ||
            surviving_energy < lepton_mass_GeV ||
            !::isfinite(electron_energy) ||
            electron_energy < lepton_mass_GeV ||
            !::isfinite(positron_energy) ||
            positron_energy < lepton_mass_GeV) {
          atomicExch(error_flag, 1U);
          return;
        }

        auto surviving = parent;
        surviving.energy_GeV = surviving_energy;
        surviving.parent_history_id = parent.history_id;
        surviving.history_id =
            first_history_id + child_offset;
        surviving.generation = parent.generation + 1;
        surviving.step_id = 0;
        surviving.reserved = 0;
        auto electron = surviving;
        electron.pid =
            static_cast<std::int32_t>(EmPid::Electron);
        electron.energy_GeV = electron_energy;
        electron.history_id =
            first_history_id + child_offset + 1;
        auto positron = surviving;
        positron.pid =
            static_cast<std::int32_t>(EmPid::Positron);
        positron.energy_GeV = positron_energy;
        positron.history_id =
            first_history_id + child_offset + 2;
        compact_secondaries[child_offset] = surviving;
        compact_secondaries[child_offset + 1] = electron;
        compact_secondaries[child_offset + 2] = positron;
        compact_records[record_offset] =
            BremsFinalStateRecord{
                interaction.input_index,
                parent.history_id,
                child_offset,
                3,
                ElectronPairProcessId,
                interaction.energy_fraction,
                sample.final_state_uniform,
                sample.azimuth_uniform,
                sample.auxiliary_uniform,
                sample.lpm_survival_probability,
                sample.lpm_uniform,
                EpairRhoDrawId,
                EpairSignDrawId,
                EpairDirectionDrawId,
                EpairLpmDrawId};
        return;
      }

      auto const photon_energy =
          parent.energy_GeV * interaction.energy_fraction;
      auto const lepton_energy =
          parent.energy_GeV - photon_energy;
      auto const azimuth =
          sample.azimuth_uniform * TwoPi;
      auto const photon_cosine =
          ::cos(lepton_mass_GeV / parent.energy_GeV);
      double photon_direction[3]{};
      deflect(
          parent.direction, photon_cosine, azimuth,
          photon_direction);

      auto lepton = parent;
      lepton.energy_GeV = lepton_energy;
      lepton.parent_history_id = parent.history_id;
      lepton.generation = parent.generation + 1;
      lepton.step_id = 0;
      lepton.reserved = 0;
      lepton.weight = sample.thinning_first_weight;
      auto const parent_momentum =
          ::sqrt(::fmax(
              0.,
              (parent.energy_GeV +
               lepton_mass_GeV) *
                  (parent.energy_GeV -
                   lepton_mass_GeV)));
      for (int axis = 0; axis < 3; ++axis) {
        lepton.direction[axis] =
            parent.direction[axis] * parent_momentum -
            photon_direction[axis] * photon_energy;
      }
      if (!normalize(lepton.direction)) {
        atomicExch(error_flag, 1U);
        return;
      }

      auto photon = parent;
      photon.pid =
          static_cast<std::int32_t>(EmPid::Photon);
      photon.energy_GeV = photon_energy;
      photon.parent_history_id = parent.history_id;
      photon.generation = parent.generation + 1;
      photon.step_id = 0;
      photon.reserved = 0;
      photon.weight = sample.thinning_second_weight;
      for (int axis = 0; axis < 3; ++axis) {
        photon.direction[axis] = photon_direction[axis];
      }

      std::uint32_t written = 0;
      if ((sample.thinning_keep_mask & 0x1U) != 0) {
        lepton.history_id =
            first_history_id + child_offset + written;
        compact_secondaries[child_offset + written] = lepton;
        ++written;
      }
      if ((sample.thinning_keep_mask & 0x2U) != 0) {
        photon.history_id =
            first_history_id + child_offset + written;
        compact_secondaries[child_offset + written] = photon;
        ++written;
      }
      compact_records[record_offset] =
          BremsFinalStateRecord{
              interaction.input_index,
              parent.history_id,
              child_offset,
              written,
              BremsProcessId,
              interaction.energy_fraction,
              0.,
              sample.azimuth_uniform,
              0.,
              sample.lpm_survival_probability,
              sample.lpm_uniform,
              0,
              BremsAzimuthDrawId,
              0,
              BremsLpmDrawId};
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

    __global__ void finalizeBremsFinalStateSummaryKernel(
        BremsCompactionValue const* classifications,
        BremsCompactionValue const* offsets,
        std::size_t count,
        std::uint32_t const* device_input_count,
        bool thinning_enabled,
        std::uint64_t first_secondary_history_id,
        std::uint32_t* summary) {
      if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
      }
      auto const last = count - 1;
      auto const final_counts =
          classifications[last] + offsets[last];
      auto const secondary_count =
          static_cast<std::uint64_t>(
              final_counts.child_count);
      auto const gpu_count =
          static_cast<std::uint64_t>(
              final_counts.record_count);
      auto const fallback_count =
          static_cast<std::uint64_t>(
              final_counts.fallback_count);
      auto const continuation_count =
          static_cast<std::uint64_t>(
              final_counts.continuation_count);
      auto const suppression_count =
          static_cast<std::uint64_t>(
              final_counts.suppression_count);
      auto const brems_count = static_cast<std::uint64_t>(
          summary[
              detail::BremsFinalStateSummaryLayout::BremsCount]);
      auto const annihilation_count =
          static_cast<std::uint64_t>(
              summary[
                  detail::BremsFinalStateSummaryLayout::
                      AnnihilationCount]);
      auto const ionization_count =
          static_cast<std::uint64_t>(
              summary[
                  detail::BremsFinalStateSummaryLayout::
                      IonizationCount]);
      auto const electron_pair_count =
          static_cast<std::uint64_t>(
              summary[
                  detail::BremsFinalStateSummaryLayout::
                      ElectronPairCount]);
      auto const brems_suppression_count =
          static_cast<std::uint64_t>(
              summary[
                  detail::BremsFinalStateSummaryLayout::
                      BremsSuppressionCount]);
      auto const electron_pair_suppression_count =
          static_cast<std::uint64_t>(
              summary[
                  detail::BremsFinalStateSummaryLayout::
                      ElectronPairSuppressionCount]);
      auto const input_count =
          device_input_count == nullptr
              ? static_cast<std::uint64_t>(count)
              : static_cast<std::uint64_t>(
                    *device_input_count);

      summary[
          detail::BremsFinalStateSummaryLayout::
              SecondaryCount] =
          static_cast<std::uint32_t>(secondary_count);
      summary[
          detail::BremsFinalStateSummaryLayout::GpuCount] =
          static_cast<std::uint32_t>(gpu_count);
      summary[
          detail::BremsFinalStateSummaryLayout::
              FallbackCount] =
          static_cast<std::uint32_t>(fallback_count);
      summary[
          detail::BremsFinalStateSummaryLayout::
              ContinuationCount] =
          static_cast<std::uint32_t>(continuation_count);
      summary[
          detail::BremsFinalStateSummaryLayout::
              SuppressionCount] =
          static_cast<std::uint32_t>(suppression_count);
      auto const expected_children =
          2 * (brems_count + annihilation_count +
               ionization_count) +
          3 * electron_pair_count;
      if (gpu_count + fallback_count +
                  continuation_count + suppression_count !=
              input_count ||
          brems_count + annihilation_count +
                  ionization_count + electron_pair_count !=
              gpu_count ||
          secondary_count > expected_children ||
          (!thinning_enabled &&
           secondary_count != expected_children) ||
          suppression_count !=
              brems_suppression_count +
                  electron_pair_suppression_count) {
        atomicExch(
            summary +
                detail::BremsFinalStateSummaryLayout::Error,
            2U);
      }
      if (secondary_count != 0 &&
          secondary_count - 1 >
              (~std::uint64_t{0}) -
                  first_secondary_history_id) {
        atomicExch(
            summary +
                detail::BremsFinalStateSummaryLayout::Error,
            3U);
      }
    }

    std::size_t scanStorageBytes(
        BremsCompactionValue* input,
        BremsCompactionValue* output,
        std::size_t count) {
      std::size_t bytes = 0;
      checkCuda(
          cub::DeviceScan::ExclusiveSum(
              nullptr, bytes, input, output, count),
          "query bremsstrahlung scan storage");
      return bytes;
    }

    void executeScan(
        void* temporary, std::size_t bytes,
        BremsCompactionValue* input,
        BremsCompactionValue* output,
        std::size_t count, char const* operation) {
      checkCuda(
          cub::DeviceScan::ExclusiveSum(
              temporary, bytes, input, output, count),
          operation);
    }

  } // namespace

  BremsLpmPreparedSnapshot prepareBremsLpmSnapshotForCuda(
      BremsLpmSnapshot const& snapshot, int device) {
    if (device < 0) {
      throw std::invalid_argument(
          "BremsLPM preparation requires a non-negative CUDA device");
    }
    checkCuda(
        cudaSetDevice(device),
        "cudaSetDevice(BremsLPM preparation)");
    BremsLpmPreparedSnapshot* device_prepared = nullptr;
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_prepared),
            sizeof(BremsLpmPreparedSnapshot)),
        "allocate prepared BremsLPM snapshot");
    try {
      prepareBremsLpmSnapshotKernel<<<1, 32>>>(
          snapshot, device_prepared);
      checkCuda(
          cudaGetLastError(),
          "launch BremsLPM snapshot preparation");
      BremsLpmPreparedSnapshot prepared{};
      checkCuda(
          cudaMemcpy(
              &prepared, device_prepared,
              sizeof(BremsLpmPreparedSnapshot),
              cudaMemcpyDeviceToHost),
          "download prepared BremsLPM snapshot");
      checkCuda(
          cudaFree(device_prepared),
          "free prepared BremsLPM snapshot");
      device_prepared = nullptr;
      if (prepared.valid == 0 ||
          prepared.component_count !=
              snapshot.component_count) {
        throw std::invalid_argument(
            "BremsLPM snapshot failed CUDA preparation");
      }
      return prepared;
    } catch (...) {
      if (device_prepared != nullptr) {
        cudaFree(device_prepared);
      }
      throw;
    }
  }

  namespace detail {

    void appendBremsFinalStateWorkspace(
        WorkspaceSize& required, std::size_t count) {
      auto const scan_bytes =
          scanStorageBytes(nullptr, nullptr, count);
      required.add<BremsParameters>(count);
      required.add<ProposalFallbackEvent>(count);
      // One vector scan computes all five stable compacted offsets: children,
      // accepted records, fallback, continuation and global suppression.
      required.add<BremsCompactionValue>(count);
      required.add<BremsCompactionValue>(count);
      required.add<BremsFinalStateRecord>(count);
      required.add<EmParticleState>(3 * count);
      required.add<ProposalFallbackEvent>(count);
      required.add<EmInteractionRecord>(count);
      required.add<BremsLpmSuppressionRecord>(count);
      required.add<std::uint32_t>(
          BremsFinalStateSummaryLayout::Size);
      required.addBytes(scan_bytes);
    }

    DeviceBremsFinalStateBatch launchBremsFinalStateOnDevice(
        BremsLpmSnapshot const& lpm_snapshot,
        BremsLpmPreparedSnapshot const& prepared_lpm,
        EmThinningConfig const& thinning,
        EmInteractionRecord const* device_interactions,
        std::size_t count, std::uint64_t random_seed,
        std::uint64_t shower_id,
        std::uint64_t first_secondary_history_id,
        DeviceWorkspace& workspace,
        bool defer_count_download,
        DeviceLeptonVertexSelectionBatch*
            deferred_vertex,
        LeptonPipelineStageEvents const* stage_events) {
      if (count == 0 || device_interactions == nullptr) {
        throw std::invalid_argument(
            "device bremsstrahlung final-state stage requires interactions");
      }
      if (first_secondary_history_id == 0) {
        throw std::invalid_argument(
            "bremsstrahlung secondary history IDs must start above zero");
      }
      if (prepared_lpm.valid == 0 ||
          prepared_lpm.component_count !=
              lpm_snapshot.component_count) {
        throw std::invalid_argument(
            "device bremsstrahlung final-state stage requires a prepared LPM snapshot");
      }
      if (count >
          std::numeric_limits<std::uint32_t>::max() / 3) {
        throw std::length_error(
            "bremsstrahlung batch exceeds 32-bit secondary offsets");
      }
      std::uint32_t const* device_input_count = nullptr;
      if (deferred_vertex != nullptr) {
        if (!deferred_vertex->counts_deferred ||
            deferred_vertex->device_summary == nullptr ||
            deferred_vertex->interactions !=
                device_interactions ||
            deferred_vertex->input_count != count) {
          throw std::invalid_argument(
              "deferred lepton vertex selection is inconsistent");
        }
        device_input_count =
            deferred_vertex->device_summary +
            LeptonVertexSummaryLayout::InteractionCount;
      }
      auto const scan_bytes =
          scanStorageBytes(nullptr, nullptr, count);
      auto* parameters =
          workspace.acquire<BremsParameters>(count);
      auto* raw_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* classifications =
          workspace.acquire<BremsCompactionValue>(count);
      auto* offsets =
          workspace.acquire<BremsCompactionValue>(count);
      auto* compact_records =
          workspace.acquire<BremsFinalStateRecord>(count);
      auto* compact_secondaries =
          workspace.acquire<EmParticleState>(3 * count);
      auto* compact_fallbacks =
          workspace.acquire<ProposalFallbackEvent>(count);
      auto* compact_continuations =
          workspace.acquire<EmInteractionRecord>(count);
      auto* compact_suppressions =
          workspace.acquire<BremsLpmSuppressionRecord>(count);
      auto* device_summary =
          workspace.acquire<std::uint32_t>(
              BremsFinalStateSummaryLayout::Size);
      auto* error_flag =
          device_summary +
          BremsFinalStateSummaryLayout::Error;
      auto* scan_temporary =
          workspace.acquireBytes(scan_bytes);
      checkCuda(
          cudaMemset(
              device_summary, 0,
              BremsFinalStateSummaryLayout::Size *
                  sizeof(std::uint32_t)),
          "clear bremsstrahlung final-state summary");

      auto const classification_block_count =
          (count + ClassificationThreadsPerBlock - 1) /
          ClassificationThreadsPerBlock;
      if (classification_block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "lepton final-state classification launch is too large");
      }
      auto const classification_blocks =
          static_cast<unsigned int>(
              classification_block_count);
      auto const block_count =
          (count + ThreadsPerBlock - 1) / ThreadsPerBlock;
      if (block_count >
          std::numeric_limits<unsigned int>::max()) {
        throw std::length_error(
            "lepton final-state write launch is too large");
      }
      auto const blocks =
          static_cast<unsigned int>(block_count);
      classifyBremsFinalStatesKernel
          <<<classification_blocks,
             ClassificationThreadsPerBlock>>>(
          lpm_snapshot, prepared_lpm, thinning,
          device_interactions, count,
          device_input_count, random_seed, shower_id,
          parameters, raw_fallbacks,
          classifications, device_summary);
      checkCuda(
          cudaGetLastError(),
          "classify bremsstrahlung final states launch");
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->
                    final_state_classification_done),
            "record lepton final-state classification stage");
      }
      executeScan(
          scan_temporary, scan_bytes, classifications,
          offsets, count,
          "scan lepton final-state compaction values");
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->final_state_scan_done),
            "record lepton final-state scan stage");
      }
      // The vector scan result and exact atomic process counters are combined
      // into the device summary, allowing the complete lepton pipeline to
      // defer its only host copy until endpoint compaction.
      finalizeBremsFinalStateSummaryKernel<<<1, 1>>>(
          classifications, offsets, count, device_input_count,
          thinning.enabled != 0,
          first_secondary_history_id, device_summary);
      checkCuda(
          cudaGetLastError(),
          "finalize lepton final-state summary launch");
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->final_state_summary_done),
            "record lepton final-state summary stage");
      }

      writeBremsFinalStatesKernel<<<blocks, ThreadsPerBlock>>>(
          device_interactions, parameters, raw_fallbacks,
          classifications, offsets,
          count,
          first_secondary_history_id,
          lpm_snapshot.lepton_mass_MeV / 1000.,
          compact_records,
          compact_secondaries, compact_fallbacks,
          compact_continuations, compact_suppressions,
          error_flag);
      checkCuda(
          cudaGetLastError(),
          "write bremsstrahlung final states launch");
      if (stage_events != nullptr) {
        checkCuda(
            cudaEventRecord(
                stage_events->final_state_write_done),
            "record lepton final-state write stage");
      }

      std::size_t secondary_count = 0;
      std::size_t gpu_count = 0;
      std::size_t fallback_count = 0;
      std::size_t continuation_count = 0;
      std::size_t suppression_count = 0;
      std::size_t brems_count = 0;
      std::size_t annihilation_count = 0;
      std::size_t ionization_count = 0;
      std::size_t electron_pair_count = 0;
      std::size_t brems_suppression_count = 0;
      std::size_t electron_pair_suppression_count = 0;
      std::size_t electron_pair_rejection_trials = 0;
      std::size_t electron_pair_zero_weight_samples = 0;
      std::size_t electron_pair_rejection_fallbacks = 0;
      std::size_t electron_pair_envelope_violations = 0;
      if (!defer_count_download) {
        std::array<
            std::uint32_t,
            BremsFinalStateSummaryLayout::Size>
            host_summary{};
        checkCuda(
            cudaMemcpy(
                host_summary.data(), device_summary,
                sizeof(host_summary),
                cudaMemcpyDeviceToHost),
            "download lepton final-state summary");
        secondary_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    SecondaryCount];
        gpu_count =
            host_summary[
                BremsFinalStateSummaryLayout::GpuCount];
        fallback_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    FallbackCount];
        continuation_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ContinuationCount];
        suppression_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    SuppressionCount];
        brems_count =
            host_summary[
                BremsFinalStateSummaryLayout::BremsCount];
        annihilation_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    AnnihilationCount];
        ionization_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    IonizationCount];
        electron_pair_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ElectronPairCount];
        brems_suppression_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    BremsSuppressionCount];
        electron_pair_suppression_count =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ElectronPairSuppressionCount];
        electron_pair_rejection_trials =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ElectronPairRejectionTrials];
        electron_pair_zero_weight_samples =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ElectronPairZeroWeightSamples];
        electron_pair_rejection_fallbacks =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ElectronPairRejectionFallbacks];
        electron_pair_envelope_violations =
            host_summary[
                BremsFinalStateSummaryLayout::
                    ElectronPairEnvelopeViolations];
        auto const error =
            host_summary[
                BremsFinalStateSummaryLayout::Error];
        if (error == 2U) {
          throw std::runtime_error(
              "lepton final-state classification lost or duplicated an interaction");
        }
        if (error == 3U) {
          throw std::overflow_error(
              "bremsstrahlung secondary history ID overflow");
        }
        if (error != 0U) {
          throw std::runtime_error(
              "bremsstrahlung final-state direction normalization failed");
        }
      }
      return {
          count,
          gpu_count,
          brems_count,
          annihilation_count,
          ionization_count,
          electron_pair_count,
          brems_count + brems_suppression_count,
          brems_suppression_count,
          electron_pair_count +
              electron_pair_suppression_count,
          electron_pair_suppression_count,
          electron_pair_rejection_trials,
          electron_pair_zero_weight_samples,
          electron_pair_rejection_fallbacks,
          electron_pair_envelope_violations,
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

  BremsFinalStateBatchResult generateBremsFinalStatesForValidation(
      BremsLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t random_seed, std::uint64_t shower_id,
      int device, std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace) {
    BremsFinalStateBatchResult result{};
    result.input_interactions = interactions.size();
    if (interactions.empty()) {
      return result;
    }
    if (device < 0) {
      throw std::invalid_argument(
          "bremsstrahlung final-state CUDA device must be non-negative");
    }
    if (first_secondary_history_id == 0) {
      throw std::invalid_argument(
          "bremsstrahlung secondary history IDs must start above zero");
    }
    checkCuda(
        cudaSetDevice(device),
        "cudaSetDevice(bremsstrahlung final state)");

    auto const count = interactions.size();
    detail::WorkspaceSize required;
    required.add<EmInteractionRecord>(count);
    detail::appendBremsFinalStateWorkspace(required, count);
    workspace.prepare(required.bytes());
    auto* device_interactions =
        workspace.acquire<EmInteractionRecord>(count);
    checkCuda(
        cudaMemcpy(
            device_interactions, interactions.data(),
            count * sizeof(EmInteractionRecord),
            cudaMemcpyHostToDevice),
        "upload bremsstrahlung interactions");

    auto const prepared_lpm =
        prepareBremsLpmSnapshotForCuda(
            lpm_snapshot, device);
    auto const batch = detail::launchBremsFinalStateOnDevice(
        lpm_snapshot, prepared_lpm, thinning,
        device_interactions, count,
        random_seed, shower_id, first_secondary_history_id,
        workspace);
    result.gpu_interactions =
        batch.gpu_interaction_count;
    result.brems_interactions =
        batch.brems_interaction_count;
    result.annihilation_interactions =
        batch.annihilation_interaction_count;
    result.ionization_interactions =
        batch.ionization_interaction_count;
    result.electron_pair_interactions =
        batch.electron_pair_interaction_count;
    result.brems_lpm_trials =
        batch.brems_lpm_trial_count;
    result.brems_lpm_suppressions =
        batch.brems_lpm_suppression_count;
    result.electron_pair_lpm_trials =
        batch.electron_pair_lpm_trial_count;
    result.electron_pair_lpm_suppressions =
        batch.electron_pair_lpm_suppression_count;
    result.electron_pair_rejection_trials =
        batch.electron_pair_rejection_trials;
    result.electron_pair_zero_weight_samples =
        batch.electron_pair_zero_weight_samples;
    result.electron_pair_rejection_fallbacks =
        batch.electron_pair_rejection_fallbacks;
    result.electron_pair_envelope_violations =
        batch.electron_pair_envelope_violations;
    result.final_state_records.resize(
        result.gpu_interactions);
    result.secondaries.resize(batch.secondary_count);
    result.fallback_events.resize(batch.fallback_count);
    result.continuations.resize(batch.continuation_count);
    result.lpm_suppressed.resize(batch.suppression_count);
    if (!result.final_state_records.empty()) {
      checkCuda(
          cudaMemcpy(
              result.final_state_records.data(), batch.records,
              result.final_state_records.size() *
                  sizeof(BremsFinalStateRecord),
              cudaMemcpyDeviceToHost),
          "download bremsstrahlung final-state records");
    }
    if (!result.secondaries.empty()) {
      checkCuda(
          cudaMemcpy(
              result.secondaries.data(), batch.secondaries,
              result.secondaries.size() *
                  sizeof(EmParticleState),
              cudaMemcpyDeviceToHost),
          "download bremsstrahlung secondaries");
    }
    if (!result.fallback_events.empty()) {
      checkCuda(
          cudaMemcpy(
              result.fallback_events.data(), batch.fallbacks,
              result.fallback_events.size() *
                  sizeof(ProposalFallbackEvent),
              cudaMemcpyDeviceToHost),
          "download bremsstrahlung fallbacks");
    }
    if (!result.continuations.empty()) {
      checkCuda(
          cudaMemcpy(
              result.continuations.data(),
              batch.continuations,
              result.continuations.size() *
                  sizeof(EmInteractionRecord),
              cudaMemcpyDeviceToHost),
          "download bremsstrahlung continuations");
    }
    if (!result.lpm_suppressed.empty()) {
      checkCuda(
          cudaMemcpy(
              result.lpm_suppressed.data(),
              batch.suppressions,
              result.lpm_suppressed.size() *
                  sizeof(BremsLpmSuppressionRecord),
              cudaMemcpyDeviceToHost),
          "download bremsstrahlung LPM suppressions");
    }
    return result;
  }

} // namespace corsika::gpu::em
