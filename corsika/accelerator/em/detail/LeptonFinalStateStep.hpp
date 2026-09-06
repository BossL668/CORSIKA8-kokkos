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
#include <corsika/accelerator/em/LeptonFinalStateRandomDomains.hpp>
#include <corsika/accelerator/em/common/BremsLpm.hpp>
#include <corsika/accelerator/em/common/EmThinning.hpp>
#include <corsika/accelerator/em/common/EpairFinalState.hpp>
#include <corsika/accelerator/em/common/EpairLpm.hpp>
#include <corsika/accelerator/em/common/Philox.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <corsika/accelerator/em/common/ProposalFallback.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>

namespace corsika::accelerator::em::detail {

  struct LeptonFinalStateParameters {
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

  struct LeptonFinalStateClassification {
    LeptonFinalStateParameters parameters{};
    gpu::em::ProposalFallbackEvent fallback{};
    std::uint32_t child_count{};
    std::uint32_t record_flag{};
    std::uint32_t fallback_flag{};
    std::uint32_t continuation_flag{};
    std::uint32_t suppression_flag{};
    std::uint32_t brems_flag{};
    std::uint32_t annihilation_flag{};
    std::uint32_t ionization_flag{};
    std::uint32_t electron_pair_flag{};
    std::uint32_t brems_suppression_flag{};
    std::uint32_t electron_pair_suppression_flag{};
    std::uint32_t electron_pair_rejection_trials{};
    std::uint32_t electron_pair_zero_weight_flag{};
    std::uint32_t electron_pair_rejection_fallback_flag{};
    std::uint32_t electron_pair_envelope_violation_flag{};
  };

  struct LeptonFinalStateMaterialization {
    gpu::em::BremsFinalStateRecord record{};
    gpu::em::EmParticleState secondaries[3]{};
    gpu::em::ProposalFallbackEvent fallback{};
    gpu::em::EmInteractionRecord continuation{};
    gpu::em::BremsLpmSuppressionRecord suppression{};
    gpu::em::GpuFirstInteractionSnapshot first_interaction{};
    std::uint32_t secondary_count{};
    std::uint32_t has_record{};
    std::uint32_t has_fallback{};
    std::uint32_t has_continuation{};
    std::uint32_t has_suppression{};
    std::uint32_t has_first_interaction{};
    std::uint32_t error{};
  };

  inline constexpr double LeptonTwoPi =
      6.283185307179586476925286766559005768;
  inline constexpr double LeptonPi =
      3.141592653589793238462643383279502884;
  inline constexpr double ProposalHalfPrecision = 1.e-5;

  C8_ACCELERATOR_INLINE_FUNCTION inline bool leptonFinite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonSqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(value);
#else
    return std::sqrt(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonLog(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::log(value);
#else
    return std::log(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonCos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::cos(value);
#else
    return std::cos(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonSin(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sin(value);
#else
    return std::sin(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonExp(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::exp(value);
#else
    return std::exp(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonFabs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonFmod(
      double left, double right) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fmod(left, right);
#else
    return std::fmod(left, right);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonPow(
      double base, double exponent) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::pow(base, exponent);
#else
    return std::pow(base, exponent);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonMaximum(
      double left, double right) {
    return left > right ? left : right;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double leptonMinimum(
      double left, double right) {
    return left < right ? left : right;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline void deflectLeptonChild(
      double const input[3], double cosine, double azimuth,
      double output[3]) {
    auto const transverse = leptonSqrt(leptonMaximum(
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
    auto const bounded = leptonMaximum(-1., leptonMinimum(1., cosine));
    auto const sine = leptonSqrt(
        leptonMaximum(0., (1. - bounded) * (1. + bounded)));
    auto const local_x = sine * leptonCos(azimuth);
    auto const local_y = sine * leptonSin(azimuth);
    auto local_z = leptonSqrt(leptonMaximum(
        0., 1. - local_x * local_x - local_y * local_y));
    if (bounded < 0.) local_z = -local_z;
    for (int axis = 0; axis < 3; ++axis) {
      output[axis] = local_z * input[axis] + local_x * rotation_x[axis] +
                     local_y * rotation_y[axis];
    }
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool normalizeLeptonDirection(
      double vector[3]) {
    auto const norm_squared = vector[0] * vector[0] + vector[1] * vector[1] +
                              vector[2] * vector[2];
    if (!leptonFinite(norm_squared) || !(norm_squared > 0.)) return false;
    auto const inverse_norm = 1. / leptonSqrt(norm_squared);
    for (int axis = 0; axis < 3; ++axis) vector[axis] *= inverse_norm;
    return leptonFinite(vector[0]) && leptonFinite(vector[1]) &&
           leptonFinite(vector[2]);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double effectiveLossFraction(
      double energy_GeV, double sampled_fraction) {
    auto const energy_MeV = energy_GeV * 1000.;
    auto const loss_MeV = sampled_fraction * energy_MeV;
    return loss_MeV / energy_MeV;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double annihilationFunctionTerm(
      double a1, double a2, double rho) {
    return a1 * leptonLog(rho) + a2 / rho - rho;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline int leptonSign(double value) {
    return (0. < value) - (value < 0.);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double annihilationEquation(
      double value, double a1, double a2, double lower_term,
      double upper_term, double uniform) {
    return annihilationFunctionTerm(a1, a2, value) - lower_term -
           uniform * (upper_term - lower_term);
  }

  /** Exact port of PROPOSAL 7.6.2 HeitlerAnnihilation::CalculateRho. */
  C8_ACCELERATOR_INLINE_FUNCTION inline bool sampleAnnihilationRho(
      double energy_GeV, double mass_GeV, double uniform, double& rho) {
    auto const gamma = energy_GeV / mass_GeV;
    if (!leptonFinite(gamma) || !(gamma > 1.) || !leptonFinite(uniform) ||
        !(uniform > 0.) || !(uniform < 1.))
      return false;
    auto const auxiliary = leptonSqrt((gamma - 1.) / (gamma + 1.));
    auto lower = 0.5 * (1. - auxiliary);
    auto upper = 0.5 * (1. + auxiliary);
    auto const a2 = 1. / ((gamma + 1.) * (gamma + 1.));
    auto const a1 = 1. + 2. * gamma * a2;
    auto const lower_term = annihilationFunctionTerm(a1, a2, lower);
    auto const upper_term = annihilationFunctionTerm(a1, a2, upper);
    auto lower_value = annihilationEquation(
        lower, a1, a2, lower_term, upper_term, uniform);
    auto const upper_value = annihilationEquation(
        upper, a1, a2, lower_term, upper_term, uniform);
    if (!leptonFinite(lower_value) || !leptonFinite(upper_value) ||
        lower_value * upper_value > 0.)
      return false;
    auto const precision =
        lower * (energy_GeV * 1000.) * ProposalHalfPrecision;
    if (!leptonFinite(precision) || !(precision > 0.)) return false;
    for (int iteration = 0; iteration <= 100; ++iteration) {
      auto const center = (lower + upper) / 2.;
      auto const center_value = annihilationEquation(
          center, a1, a2, lower_term, upper_term, uniform);
      if (leptonSign(center_value) == leptonSign(lower_value)) {
        lower = center;
        lower_value = center_value;
      } else {
        upper = center;
      }
      if (leptonFabs(upper - lower) < precision) {
        rho = lower;
        return leptonFinite(rho) && rho > 0. && rho < 1.;
      }
    }
    rho = lower;
    return leptonFinite(rho) && rho > 0. && rho < 1.;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::ProposalFallbackEvent
  invalidLeptonFinalState(
      gpu::em::EmInteractionRecord const& interaction,
      LeptonFinalStateParameters const& parameters = {}) {
    auto event = gpu::em::makeProcessFallbackEvent(
        interaction, gpu::em::ProposalFallbackReason::InvalidFinalState);
    if (parameters.final_state_uniform > 0.) {
      event.final_state_uniform = parameters.final_state_uniform;
      event.final_state_draw_id =
          parameters.process_id == gpu::em::AnnihilationProcessId
              ? AnnihilationRhoDrawId
              : parameters.process_id == gpu::em::ElectronPairProcessId
                    ? EpairRhoDrawId
                    : BremsAzimuthDrawId;
    } else if (parameters.azimuth_uniform > 0.) {
      event.final_state_uniform = parameters.azimuth_uniform;
      event.final_state_draw_id =
          parameters.process_id == gpu::em::AnnihilationProcessId
              ? AnnihilationAzimuthDrawId
              : parameters.process_id == gpu::em::ElectronPairProcessId
                    ? EpairSignDrawId
                    : BremsAzimuthDrawId;
    }
    return event;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool applyLeptonTwoChildThinning(
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmParticleState const& parent, std::int32_t process_id,
      double first_energy_GeV, double second_energy_GeV,
      std::uint64_t random_seed, std::uint64_t shower_id,
      LeptonFinalStateParameters& sample, std::uint32_t& child_count) {
    gpu::em::RandomNumberKey first_key{
        random_seed, shower_id, parent.history_id, parent.step_id,
        static_cast<std::uint32_t>(process_id),
        gpu::em::EmThinningFirstDrawId};
    auto second_key = first_key;
    second_key.draw_id = gpu::em::EmThinningSecondDrawId;
    sample.thinning_first_uniform = gpu::em::uniformOpen01(first_key);
    sample.thinning_second_uniform = gpu::em::uniformOpen01(second_key);
    auto effective_thinning = thinning;
    if (!gpu::em::isElectronOrPositronPid(parent.pid))
      effective_thinning.enabled = 0;
    auto const result = gpu::em::applyEmThinning(
        effective_thinning, parent.energy_GeV, parent.weight,
        first_energy_GeV, second_energy_GeV,
        sample.thinning_first_uniform, sample.thinning_second_uniform);
    if (result.status == gpu::em::EmThinningStatus::InvalidInput) return false;
    sample.thinning_status = static_cast<std::uint32_t>(result.status);
    sample.thinning_keep_mask = result.keep_mask;
    sample.thinning_first_weight = result.first_weight;
    sample.thinning_second_weight = result.second_weight;
    child_count = (result.keep_mask & 0x1U ? 1U : 0U) +
                  (result.keep_mask & 0x2U ? 1U : 0U);
    return true;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline gpu::em::BremsLpmResult
  preparedBremsLpmSuppressionFactor(
      gpu::em::BremsLpmSnapshot const& snapshot,
      gpu::em::BremsLpmPreparedSnapshot const& prepared,
      std::uint64_t component_hash, double energy_MeV, double v,
      double local_mass_density_g_per_cm3) {
    using gpu::em::BremsLpmResult;
    using gpu::em::BremsLpmStatus;
    if (prepared.valid == 0)
      return {BremsLpmStatus::InvalidSnapshot, 0, 0.};
    if (!leptonFinite(energy_MeV) ||
        !(energy_MeV > snapshot.lepton_mass_MeV) || !leptonFinite(v) ||
        !(v > 0.) || !(v < 1.) ||
        !leptonFinite(local_mass_density_g_per_cm3) ||
        !(local_mass_density_g_per_cm3 > 0.))
      return {BremsLpmStatus::InvalidInput, 0, 0.};
    gpu::em::BremsLpmPreparedComponent component{};
    bool found = false;
    for (std::uint32_t index = 0; index < prepared.component_count; ++index) {
      if (prepared.components[index].component_hash == component_hash) {
        component = prepared.components[index];
        found = true;
        break;
      }
    }
    if (!found) return {BremsLpmStatus::ComponentNotFound, 0, 0.};
    constexpr double Fi1 = 1.54954;
    constexpr double G1 = 0.710390;
    constexpr double G2 = 0.904912;
    auto const density_correction = local_mass_density_g_per_cm3 /
                                    snapshot.baseline_mass_density_g_per_cm3;
    auto const sp = 0.125 * leptonSqrt(
                                snapshot.e_lpm_MeV * v /
                                (density_correction * energy_MeV * (1. - v)));
    auto const h = leptonLog(sp) / component.logarithm_s1;
    double xi = 1.;
    if (sp < component.s1) {
      xi = 2.;
    } else if (sp < 1.) {
      xi = 1. + h - 0.08 * (1. - h) *
                          (1. - (1. - h) * (1. - h)) /
                          component.logarithm_s1;
    }
    auto gamma = snapshot.classical_electron_radius_cm *
                 snapshot.electron_mass_MeV /
                 (snapshot.fine_structure_constant *
                  snapshot.lepton_mass_MeV * v);
    gamma = 1. + 4. * LeptonPi * snapshot.sum_charge *
                     snapshot.classical_electron_radius_cm * gamma * gamma *
                     snapshot.molecular_density_per_cm3 * density_correction;
    auto const s = sp / leptonSqrt(xi) * gamma;
    auto const s2 = s * s;
    double fi = 0.;
    if (s < Fi1) {
      fi = 1. - leptonExp(-6. * s * (1. + (3. - LeptonPi) * s) +
                          s2 * s /
                              (0.623 + 0.796 * s + 0.658 * s2));
    } else {
      fi = 1. - 0.012 / (s2 * s2);
    }
    double g = 0.;
    if (s < G1) {
      auto const psi =
          1. - leptonExp(-4. * s -
                         8. * s2 /
                             (1. + 3.936 * s + 4.97 * s2 -
                              0.05 * s2 * s + 7.50 * s2 * s2));
      g = 3. * psi - 2. * fi;
    } else if (s < G2) {
      g = 36. * s2 / (36. * s2 + 1.);
    } else {
      g = 1. - 0.022 / (s2 * s2);
    }
    auto const probability =
        ((xi / 3.) *
         (v * v * g / (gamma * gamma) +
          2. * (1. + (1. - v) * (1. - v)) * fi / gamma)) /
        ((4. / 3.) * (1. - v) + v * v);
    if (!leptonFinite(probability) || probability < 0.)
      return {BremsLpmStatus::NonFiniteResult, 0, 0.};
    return {BremsLpmStatus::Success, 0, probability};
  }

  template <bool UsePreparedLpm>
  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonFinalStateClassification
  classifyLeptonFinalStateImpl(
      gpu::em::BremsLpmSnapshot const& lpm_snapshot,
      gpu::em::BremsLpmPreparedSnapshot const& prepared_lpm,
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmInteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id) {
    using namespace gpu::em;
    LeptonFinalStateClassification result{};
    if (interaction.status == EmInteractionStatus::NoDiscreteInteraction) {
      result.continuation_flag = 1;
      return result;
    }
    if (interaction.status != EmInteractionStatus::Selected) {
      result.fallback = invalidLeptonFinalState(interaction);
      result.fallback_flag = 1;
      return result;
    }
    auto const capability =
        gpuProcessCapability(interaction.particle.pid, interaction.process_id);
    if (capability != GpuProcessCapability::Bremsstrahlung &&
        capability != GpuProcessCapability::Annihilation &&
        capability != GpuProcessCapability::Ionization &&
        capability != GpuProcessCapability::ElectronPair) {
      result.fallback = makeProcessFallbackEvent(
          interaction, processFallbackReason(capability));
      result.fallback_flag = 1;
      return result;
    }
    auto const& parent = interaction.particle;
    auto const v = interaction.energy_fraction;
    // Preserve the native CUDA/PROPOSAL adapter convention exactly.  The
    // snapshot passed to this final-state stage is the active lepton
    // snapshot; its lepton mass is also the legacy electron-mass fallback.
    auto const electron_mass_GeV = lpm_snapshot.lepton_mass_MeV / 1000.;
    auto const lepton_mass_GeV = interaction.particle_mass_GeV > 0.
                                     ? interaction.particle_mass_GeV
                                     : lpm_snapshot.lepton_mass_MeV / 1000.;
    if (!leptonFinite(parent.energy_GeV) ||
        !(parent.energy_GeV > lepton_mass_GeV) ||
        parent.generation == 0xffffffffU ||
        parent.step_id == 0xffffffffffffffffULL) {
      result.fallback = invalidLeptonFinalState(interaction);
      result.fallback_flag = 1;
      return result;
    }
    auto& sample = result.parameters;
    sample.process_id = interaction.process_id;
    if (capability == GpuProcessCapability::Annihilation) {
      if (!leptonFinite(v) || leptonFabs(v - 1.) > 1.e-12) {
        result.fallback = invalidLeptonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      RandomNumberKey const rho_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(AnnihilationProcessId),
          AnnihilationRhoDrawId};
      RandomNumberKey const azimuth_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(AnnihilationProcessId),
          AnnihilationAzimuthDrawId};
      sample.final_state_uniform = uniformOpen01(rho_key);
      sample.azimuth_uniform = uniformOpen01(azimuth_key);
      if (!sampleAnnihilationRho(parent.energy_GeV, lepton_mass_GeV,
                                 sample.final_state_uniform,
                                 sample.energy_split_fraction)) {
        result.fallback = invalidLeptonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      auto const total = parent.energy_GeV + lepton_mass_GeV;
      if (!applyLeptonTwoChildThinning(
              thinning, parent, AnnihilationProcessId,
              total * (1. - sample.energy_split_fraction),
              total * sample.energy_split_fraction, random_seed, shower_id,
              sample, result.child_count)) {
        result.fallback = invalidLeptonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      result.record_flag = 1;
      result.annihilation_flag = 1;
      return result;
    }
    if (capability == GpuProcessCapability::Ionization) {
      auto const effective_v = effectiveLossFraction(parent.energy_GeV, v);
      auto const outgoing = parent.energy_GeV * (1. - effective_v);
      auto const delta = parent.energy_GeV * effective_v + electron_mass_GeV;
      if (!leptonFinite(effective_v) || !(effective_v > 0.) ||
          !(effective_v < 1.) || !leptonFinite(outgoing) ||
          outgoing < lepton_mass_GeV || !leptonFinite(delta) ||
          !(delta > electron_mass_GeV)) {
        result.fallback = invalidLeptonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      RandomNumberKey const azimuth_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(IonizationProcessId),
          IonizationAzimuthDrawId};
      sample.azimuth_uniform = uniformOpen01(azimuth_key);
      if (!applyLeptonTwoChildThinning(
              thinning, parent, IonizationProcessId,
              proposalSecondaryTransportEnergyGeV(parent.pid, outgoing, lepton_mass_GeV),
              proposalSecondaryTransportEnergyGeV(11, delta, electron_mass_GeV),
              random_seed, shower_id, sample, result.child_count)) {
        result.fallback = invalidLeptonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      result.record_flag = 1;
      result.ionization_flag = 1;
      return result;
    }
    if (capability == GpuProcessCapability::ElectronPair) {
      auto const effective_v = effectiveLossFraction(parent.energy_GeV, v);
      auto const loss = parent.energy_GeV * effective_v;
      auto const surviving = parent.energy_GeV - loss;
      if (!leptonFinite(effective_v) || !(effective_v > 0.) ||
          !(effective_v < 1.) || !leptonFinite(loss) ||
          !(loss >= 4. * lepton_mass_GeV) || !leptonFinite(surviving) ||
          surviving < lepton_mass_GeV) {
        result.fallback = invalidLeptonFinalState(interaction, sample);
        result.fallback_flag = 1;
        return result;
      }
      RandomNumberKey const rho_key{
          random_seed, shower_id, parent.history_id, parent.step_id,
          static_cast<std::uint32_t>(ElectronPairProcessId), EpairRhoDrawId};
      auto sign_key = rho_key;
      sign_key.draw_id = EpairSignDrawId;
      auto direction_key = rho_key;
      direction_key.draw_id = EpairDirectionDrawId;
      sample.final_state_uniform = uniformOpen01(rho_key);
      sample.azimuth_uniform = uniformOpen01(sign_key);
      sample.auxiliary_uniform = uniformOpen01(direction_key);
      auto const rejection = sampleEpairRhoRejection(
          lpm_snapshot, interaction.component_hash,
          parent.energy_GeV * 1000., effective_v, sample.azimuth_uniform,
          rho_key);
      auto const split = rejection.sample;
      result.electron_pair_rejection_trials = rejection.trial_count;
      if (rejection.trial_count == 0 &&
          split.status == EpairFinalStateStatus::Success && split.rho == 0.)
        result.electron_pair_zero_weight_flag = 1;
      if (split.status != EpairFinalStateStatus::Success) {
        result.electron_pair_rejection_fallback_flag = 1;
        if (split.status == EpairFinalStateStatus::RejectionEnvelopeExceeded)
          result.electron_pair_envelope_violation_flag = 1;
        auto reason = ProposalFallbackReason::InvalidFinalState;
        if (split.status == EpairFinalStateStatus::ComponentNotFound ||
            split.status == EpairFinalStateStatus::InvalidSnapshot)
          reason = ProposalFallbackReason::LpmParametersUnavailable;
        else if (split.status ==
                 EpairFinalStateStatus::RejectionEnvelopeExceeded)
          reason = ProposalFallbackReason::EpairRejectionEnvelopeExceeded;
        result.fallback = makeProcessFallbackEvent(interaction, reason);
        result.fallback.diagnostic_status =
            static_cast<std::int32_t>(split.status);
        result.fallback.diagnostic_value0 = split.rho_max;
        result.fallback.diagnostic_value1 =
            static_cast<double>(rejection.trial_count);
        result.fallback.final_state_uniform = sample.final_state_uniform;
        result.fallback.final_state_draw_id = EpairRhoDrawId;
        result.fallback_flag = 1;
        return result;
      }
      sample.energy_split_fraction = split.rho;
      auto const lpm = epairLpmSuppressionFactor(
          lpm_snapshot, parent.energy_GeV * 1000., effective_v,
          split.rho * split.rho, interaction.mass_density_g_per_cm3);
      if (lpm.status != EpairLpmStatus::Success) {
        auto const reason = lpm.status == EpairLpmStatus::InvalidInput
                                ? ProposalFallbackReason::InvalidMassDensity
                                : ProposalFallbackReason::LpmParametersUnavailable;
        result.fallback = makeProcessFallbackEvent(interaction, reason);
        result.fallback.final_state_uniform = sample.final_state_uniform;
        result.fallback.final_state_draw_id = EpairRhoDrawId;
        result.fallback_flag = 1;
        return result;
      }
      sample.lpm_survival_probability = lpm.survival_probability;
      auto lpm_key = rho_key;
      lpm_key.draw_id = EpairLpmDrawId;
      sample.lpm_uniform = uniformOpen01(lpm_key);
      if (sample.lpm_uniform > sample.lpm_survival_probability) {
        result.suppression_flag = 1;
        result.electron_pair_suppression_flag = 1;
        return result;
      }
      result.child_count = 3;
      result.record_flag = 1;
      result.electron_pair_flag = 1;
      return result;
    }
    auto const photon_energy = parent.energy_GeV * v;
    auto const lepton_energy = parent.energy_GeV - photon_energy;
    if (!leptonFinite(v) || !(v > 0.) || !(v < 1.) ||
        !leptonFinite(photon_energy) || !(photon_energy > 0.) ||
        !leptonFinite(lepton_energy) || lepton_energy < lepton_mass_GeV) {
      result.fallback = invalidLeptonFinalState(interaction, sample);
      result.fallback.diagnostic_value0 = v;
      result.fallback.diagnostic_value1 = interaction.loss_quantile;
      result.fallback.diagnostic_value2 = lepton_energy;
      result.fallback_flag = 1;
      return result;
    }
    RandomNumberKey const azimuth_key{
        random_seed, shower_id, parent.history_id, parent.step_id,
        static_cast<std::uint32_t>(BremsProcessId), BremsAzimuthDrawId};
    sample.azimuth_uniform = uniformOpen01(azimuth_key);
    auto const lpm = UsePreparedLpm
                         ? preparedBremsLpmSuppressionFactor(
                               lpm_snapshot, prepared_lpm,
                               interaction.component_hash,
                               parent.energy_GeV * 1000., v,
                               interaction.mass_density_g_per_cm3)
                         : bremsLpmSuppressionFactor(
                               lpm_snapshot, interaction.component_hash,
                               parent.energy_GeV * 1000., v,
                               interaction.mass_density_g_per_cm3);
    if (lpm.status != BremsLpmStatus::Success) {
      auto const reason = lpm.status == BremsLpmStatus::InvalidInput
                              ? ProposalFallbackReason::InvalidMassDensity
                              : ProposalFallbackReason::LpmParametersUnavailable;
      result.fallback = makeProcessFallbackEvent(interaction, reason);
      result.fallback.final_state_uniform = sample.azimuth_uniform;
      result.fallback.final_state_draw_id = BremsAzimuthDrawId;
      result.fallback_flag = 1;
      return result;
    }
    sample.lpm_survival_probability = lpm.survival_probability;
    auto lpm_key = azimuth_key;
    lpm_key.draw_id = BremsLpmDrawId;
    sample.lpm_uniform = uniformOpen01(lpm_key);
    if (sample.lpm_uniform > sample.lpm_survival_probability) {
      result.suppression_flag = 1;
      result.brems_suppression_flag = 1;
      return result;
    }
    if (!applyLeptonTwoChildThinning(
            thinning, parent, BremsProcessId,
            proposalSecondaryTransportEnergyGeV(parent.pid, lepton_energy, lepton_mass_GeV),
            photon_energy,
            random_seed, shower_id, sample, result.child_count)) {
      result.fallback = invalidLeptonFinalState(interaction, sample);
      result.fallback_flag = 1;
      return result;
    }
    result.record_flag = 1;
    result.brems_flag = 1;
    return result;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonFinalStateClassification
  classifyLeptonFinalState(
      gpu::em::BremsLpmSnapshot const& lpm_snapshot,
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmInteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id) {
    return classifyLeptonFinalStateImpl<false>(
        lpm_snapshot, {}, thinning, interaction, random_seed, shower_id);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonFinalStateClassification
  classifyLeptonFinalState(
      gpu::em::BremsLpmSnapshot const& lpm_snapshot,
      gpu::em::BremsLpmPreparedSnapshot const& prepared_lpm,
      gpu::em::EmThinningConfig const& thinning,
      gpu::em::EmInteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id) {
    return classifyLeptonFinalStateImpl<true>(
        lpm_snapshot, prepared_lpm, thinning, interaction, random_seed,
        shower_id);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline void captureLeptonFirstInteraction(
      LeptonFinalStateMaterialization& output,
      gpu::em::EmParticleState const& parent, std::int32_t process_id,
      gpu::em::EmParticleState const& first,
      gpu::em::EmParticleState const* second = nullptr,
      gpu::em::EmParticleState const* third = nullptr) {
    if (parent.generation != 0) return;
    auto& snapshot = output.first_interaction;
    snapshot.parent_at_vertex = parent;
    snapshot.process_id = process_id;
    snapshot.secondary_count = third ? 3U : (second ? 2U : 1U);
    snapshot.secondaries[0] = first;
    snapshot.secondaries[0].weight = parent.weight;
    if (second) {
      snapshot.secondaries[1] = *second;
      snapshot.secondaries[1].weight = parent.weight;
    }
    if (third) {
      snapshot.secondaries[2] = *third;
      snapshot.secondaries[2].weight = parent.weight;
    }
    output.has_first_interaction = 1;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline void populateLeptonThinningRecord(
      gpu::em::BremsFinalStateRecord& record,
      LeptonFinalStateParameters const& sample) {
    record.thinning_status = sample.thinning_status;
    record.thinning_keep_mask = sample.thinning_keep_mask;
    record.thinning_first_uniform = sample.thinning_first_uniform;
    record.thinning_second_uniform = sample.thinning_second_uniform;
    record.thinning_first_draw_id = gpu::em::EmThinningFirstDrawId;
    record.thinning_second_draw_id = gpu::em::EmThinningSecondDrawId;
  }

  /** Compatibility sink for callers that still need the aggregate result. */
  struct LeptonFinalStateMaterializationWriter {
    LeptonFinalStateMaterialization& output;

    C8_ACCELERATOR_INLINE_FUNCTION void fallback(
        gpu::em::ProposalFallbackEvent const& value) {
      output.fallback = value;
      output.has_fallback = 1;
    }

    C8_ACCELERATOR_INLINE_FUNCTION void continuation(
        gpu::em::EmInteractionRecord const& value) {
      output.continuation = value;
      output.has_continuation = 1;
    }

    C8_ACCELERATOR_INLINE_FUNCTION void suppression(
        gpu::em::EmParticleState const& parent,
        gpu::em::EmInteractionRecord const& interaction,
        LeptonFinalStateParameters const& sample) {
      output.suppression = {
          parent, interaction.input_index, interaction.component_hash,
          sample.lpm_survival_probability, sample.lpm_uniform,
          sample.process_id == gpu::em::ElectronPairProcessId
              ? EpairLpmDrawId
              : BremsLpmDrawId};
      output.has_suppression = 1;
    }

    C8_ACCELERATOR_INLINE_FUNCTION void firstInteraction(
        gpu::em::EmParticleState const& parent, std::int32_t process_id,
        gpu::em::EmParticleState const& first,
        gpu::em::EmParticleState const* second = nullptr,
        gpu::em::EmParticleState const* third = nullptr) {
      captureLeptonFirstInteraction(
          output, parent, process_id, first, second, third);
    }

    C8_ACCELERATOR_INLINE_FUNCTION void secondary(
        std::uint32_t index, gpu::em::EmParticleState const& particle) {
      output.secondaries[index] = particle;
    }

    C8_ACCELERATOR_INLINE_FUNCTION gpu::em::BremsFinalStateRecord& record() {
      return output.record;
    }

    C8_ACCELERATOR_INLINE_FUNCTION void finishRecord(
        std::uint32_t secondary_count) {
      output.secondary_count = secondary_count;
      output.has_record = 1;
    }

    C8_ACCELERATOR_INLINE_FUNCTION void fail(std::uint32_t error) {
      output.error = error;
    }
  };

  /**
   * Materialize directly into a caller-provided sink.
   *
   * Resident kernels use this overload to avoid constructing the 1.7 KiB
   * aggregate result in each CUDA thread.  The aggregate-return overload below
   * remains the compatibility and validation interface.
   */
  template <class Writer>
  C8_ACCELERATOR_INLINE_FUNCTION inline void
  materializeLeptonFinalStateInPlace(
      gpu::em::EmInteractionRecord const& interaction,
      LeptonFinalStateClassification const& classification,
      std::uint64_t child_offset, std::uint64_t first_history_id,
      double electron_mass_GeV, Writer& output) {
    using namespace gpu::em;
    if (classification.fallback_flag) {
      output.fallback(classification.fallback);
      return;
    }
    if (classification.continuation_flag) {
      output.continuation(interaction);
      return;
    }
    auto const sample = classification.parameters;
    if (classification.suppression_flag) {
      auto parent = interaction.particle;
      ++parent.step_id;
      output.suppression(parent, interaction, sample);
      return;
    }
    if (!classification.record_flag) return;
    auto const parent = interaction.particle;
    auto const lepton_mass_GeV = interaction.particle_mass_GeV > 0.
                                     ? interaction.particle_mass_GeV
                                     : electron_mass_GeV;
    std::uint32_t secondary_count = 0;
    if (sample.process_id == AnnihilationProcessId) {
      auto const rho = sample.energy_split_fraction;
      auto const total = parent.energy_GeV + lepton_mass_GeV;
      auto const first_energy = total * (1. - rho);
      auto const second_energy = total * rho;
      auto const momentum = leptonSqrt(
          (parent.energy_GeV + lepton_mass_GeV) *
          (parent.energy_GeV - lepton_mass_GeV));
      auto const first_cosine =
          (total * (1. - rho) - lepton_mass_GeV) /
          ((1. - rho) * momentum);
      auto const second_cosine =
          (total * rho - lepton_mass_GeV) / (rho * momentum);
      if (!leptonFinite(first_energy) || !leptonFinite(second_energy) ||
          !(first_energy > 0.) || !(second_energy > 0.) ||
          !leptonFinite(first_cosine) || !leptonFinite(second_cosine)) {
        output.fail(1);
        return;
      }
      auto const azimuth = sample.azimuth_uniform * LeptonTwoPi;
      double first_direction[3]{};
      double second_direction[3]{};
      deflectLeptonChild(parent.direction, first_cosine, azimuth,
                         first_direction);
      deflectLeptonChild(parent.direction, second_cosine,
                         leptonFmod(azimuth + LeptonPi, LeptonTwoPi),
                         second_direction);
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
      output.firstInteraction(parent, AnnihilationProcessId, first, &second);
      if (sample.thinning_keep_mask & 0x1U) {
        first.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, first);
      }
      if (sample.thinning_keep_mask & 0x2U) {
        second.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, second);
      }
      output.record() = {interaction.input_index,
                         parent.history_id,
                         child_offset,
                         secondary_count,
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
      populateLeptonThinningRecord(output.record(), sample);
      output.finishRecord(secondary_count);
      return;
    }
    if (sample.process_id == IonizationProcessId) {
      auto const v = effectiveLossFraction(parent.energy_GeV,
                                           interaction.energy_fraction);
      auto const outgoing_energy = parent.energy_GeV * (1. - v);
      auto const delta_energy =
          parent.energy_GeV * v + electron_mass_GeV;
      auto const incoming_momentum = leptonSqrt(
          (parent.energy_GeV + lepton_mass_GeV) *
          (parent.energy_GeV - lepton_mass_GeV));
      auto const outgoing_momentum = leptonSqrt(
          (outgoing_energy + lepton_mass_GeV) *
          (outgoing_energy - lepton_mass_GeV));
      auto const delta_momentum = leptonSqrt(
          (delta_energy + electron_mass_GeV) *
          (delta_energy - electron_mass_GeV));
      auto const outgoing_cosine =
          ((parent.energy_GeV + electron_mass_GeV) * outgoing_energy -
           parent.energy_GeV * electron_mass_GeV -
           lepton_mass_GeV * lepton_mass_GeV) /
          (incoming_momentum * outgoing_momentum);
      auto const delta_cosine =
          ((parent.energy_GeV + electron_mass_GeV) * delta_energy -
           parent.energy_GeV * electron_mass_GeV -
           electron_mass_GeV * electron_mass_GeV) /
          (incoming_momentum * delta_momentum);
      if (!leptonFinite(outgoing_cosine) || !leptonFinite(delta_cosine)) {
        output.fail(1);
        return;
      }
      auto const azimuth = sample.azimuth_uniform * LeptonTwoPi;
      double outgoing_direction[3]{};
      double delta_direction[3]{};
      deflectLeptonChild(parent.direction, outgoing_cosine, azimuth,
                         outgoing_direction);
      deflectLeptonChild(parent.direction, delta_cosine,
                         leptonFmod(azimuth + LeptonPi, LeptonTwoPi),
                         delta_direction);
      auto outgoing = parent;
      outgoing.energy_GeV = proposalSecondaryTransportEnergyGeV(
          outgoing.pid, outgoing_energy, lepton_mass_GeV);
      outgoing.parent_history_id = parent.history_id;
      outgoing.generation = parent.generation + 1;
      outgoing.step_id = 0;
      outgoing.reserved = 0;
      outgoing.weight = sample.thinning_first_weight;
      auto delta = outgoing;
      delta.pid = static_cast<std::int32_t>(EmPid::Electron);
      delta.energy_GeV = proposalSecondaryTransportEnergyGeV(
          delta.pid, delta_energy, electron_mass_GeV);
      delta.weight = sample.thinning_second_weight;
      for (int axis = 0; axis < 3; ++axis) {
        outgoing.direction[axis] = outgoing_direction[axis];
        delta.direction[axis] = delta_direction[axis];
      }
      output.firstInteraction(parent, IonizationProcessId, outgoing, &delta);
      if (sample.thinning_keep_mask & 0x1U) {
        outgoing.history_id =
            first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, outgoing);
      }
      if (sample.thinning_keep_mask & 0x2U) {
        delta.history_id = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, delta);
      }
      output.record() = {interaction.input_index,
                         parent.history_id,
                         child_offset,
                         secondary_count,
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
      populateLeptonThinningRecord(output.record(), sample);
      output.record().weighted_mass_convention_correction_GeV =
          weightedProposalMassCorrectionGeV(parent.pid, lepton_mass_GeV,
              (sample.thinning_keep_mask & 0x1U) ? sample.thinning_first_weight : 0.) +
          weightedProposalMassCorrectionGeV(11, electron_mass_GeV,
              (sample.thinning_keep_mask & 0x2U) ? sample.thinning_second_weight : 0.);
      output.finishRecord(secondary_count);
      return;
    }
    if (sample.process_id == ElectronPairProcessId) {
      auto const v = effectiveLossFraction(parent.energy_GeV,
                                           interaction.energy_fraction);
      auto const loss_energy = parent.energy_GeV * v;
      auto const rho = sample.energy_split_fraction;
      auto const surviving_energy = parent.energy_GeV - loss_energy;
      auto const electron_energy = 0.5 * loss_energy * (1. + rho);
      auto const positron_energy = 0.5 * loss_energy * (1. - rho);
      if (!leptonFinite(surviving_energy) ||
          surviving_energy < lepton_mass_GeV ||
          !leptonFinite(electron_energy) ||
          electron_energy < lepton_mass_GeV ||
          !leptonFinite(positron_energy) ||
          positron_energy < lepton_mass_GeV) {
        output.fail(1);
        return;
      }
      auto surviving = parent;
      surviving.energy_GeV = proposalSecondaryTransportEnergyGeV(
          surviving.pid, surviving_energy, lepton_mass_GeV);
      surviving.parent_history_id = parent.history_id;
      surviving.history_id = first_history_id + child_offset;
      surviving.generation = parent.generation + 1;
      surviving.step_id = 0;
      surviving.reserved = 0;
      auto electron = surviving;
      electron.pid = static_cast<std::int32_t>(EmPid::Electron);
      electron.energy_GeV = proposalSecondaryTransportEnergyGeV(
          electron.pid, electron_energy, electron_mass_GeV);
      electron.history_id = first_history_id + child_offset + 1;
      auto positron = surviving;
      positron.pid = static_cast<std::int32_t>(EmPid::Positron);
      positron.energy_GeV = proposalSecondaryTransportEnergyGeV(
          positron.pid, positron_energy, electron_mass_GeV);
      positron.history_id = first_history_id + child_offset + 2;
      output.firstInteraction(parent, ElectronPairProcessId, surviving,
                              &electron, &positron);
      output.secondary(0, surviving);
      output.secondary(1, electron);
      output.secondary(2, positron);
      secondary_count = 3;
      output.record() = {interaction.input_index,
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
      output.record().weighted_mass_convention_correction_GeV =
          weightedProposalMassCorrectionGeV(parent.pid, lepton_mass_GeV, parent.weight) +
          weightedProposalMassCorrectionGeV(11, electron_mass_GeV, 2. * parent.weight);
      output.finishRecord(secondary_count);
      return;
    }
    auto const photon_energy =
        parent.energy_GeV * interaction.energy_fraction;
    auto const lepton_energy = parent.energy_GeV - photon_energy;
    auto const azimuth = sample.azimuth_uniform * LeptonTwoPi;
    auto const photon_cosine = leptonCos(lepton_mass_GeV / parent.energy_GeV);
    double photon_direction[3]{};
    deflectLeptonChild(parent.direction, photon_cosine, azimuth,
                       photon_direction);
    auto lepton = parent;
    lepton.energy_GeV = proposalSecondaryTransportEnergyGeV(
        lepton.pid, lepton_energy, lepton_mass_GeV);
    lepton.parent_history_id = parent.history_id;
    lepton.generation = parent.generation + 1;
    lepton.step_id = 0;
    lepton.reserved = 0;
    lepton.weight = sample.thinning_first_weight;
    auto const parent_momentum = leptonSqrt(leptonMaximum(
        0., (parent.energy_GeV + lepton_mass_GeV) *
                (parent.energy_GeV - lepton_mass_GeV)));
    for (int axis = 0; axis < 3; ++axis) {
      lepton.direction[axis] = parent.direction[axis] * parent_momentum -
                               photon_direction[axis] * photon_energy;
    }
    if (!normalizeLeptonDirection(lepton.direction)) {
      output.fail(1);
      return;
    }
    auto photon = parent;
    photon.pid = static_cast<std::int32_t>(EmPid::Photon);
    photon.energy_GeV = photon_energy;
    photon.parent_history_id = parent.history_id;
    photon.generation = parent.generation + 1;
    photon.step_id = 0;
    photon.reserved = 0;
    photon.weight = sample.thinning_second_weight;
    for (int axis = 0; axis < 3; ++axis)
      photon.direction[axis] = photon_direction[axis];
    output.firstInteraction(parent, BremsProcessId, lepton, &photon);
    if (sample.thinning_keep_mask & 0x1U) {
      lepton.history_id = first_history_id + child_offset + secondary_count;
      output.secondary(secondary_count++, lepton);
    }
    if (sample.thinning_keep_mask & 0x2U) {
      photon.history_id = first_history_id + child_offset + secondary_count;
      output.secondary(secondary_count++, photon);
    }
    output.record() = {interaction.input_index,
                       parent.history_id,
                       child_offset,
                       secondary_count,
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
    populateLeptonThinningRecord(output.record(), sample);
    output.record().weighted_mass_convention_correction_GeV =
        weightedProposalMassCorrectionGeV(parent.pid, lepton_mass_GeV,
            (sample.thinning_keep_mask & 0x1U) ? sample.thinning_first_weight : 0.);
    output.finishRecord(secondary_count);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonFinalStateMaterialization
  materializeLeptonFinalState(
      gpu::em::EmInteractionRecord const& interaction,
      LeptonFinalStateClassification const& classification,
      std::uint64_t child_offset, std::uint64_t first_history_id,
      double electron_mass_GeV) {
    LeptonFinalStateMaterialization output{};
    LeptonFinalStateMaterializationWriter writer{output};
    materializeLeptonFinalStateInPlace(
        interaction, classification, child_offset, first_history_id,
        electron_mass_GeV, writer);
    return output;
  }

} // namespace corsika::accelerator::em::detail
