/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace lepton_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool lepton_finite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_sqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_log(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::log(value);
#else
      return std::log(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_cos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::cos(value);
#else
      return std::cos(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_sin(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::sin(value);
#else
      return std::sin(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_exp(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::exp(value);
#else
      return std::exp(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_fabs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::fabs(value);
#else
      return std::fabs(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_fmod(double left, double right) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::fmod(left, right);
#else
      return std::fmod(left, right);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_pow(double base, double exponent) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::pow(base, exponent);
#else
      return std::pow(base, exponent);
#endif
    }
    CORSIKA_HOST_DEVICE inline double lepton_maximum(double left, double right) {
      return left > right ? left : right;
    }
    CORSIKA_HOST_DEVICE inline double lepton_minimum(double left, double right) {
      return left < right ? left : right;
    }
    CORSIKA_HOST_DEVICE inline void deflect_lepton_child(double const input[3],
                                                         double cosine, double azimuth,
                                                         double output[3]) {
      auto const transverse = lepton_final_state_detail::lepton_sqrt(
          lepton_final_state_detail::lepton_maximum(
              0., input[0] * input[0] + input[1] * input[1]));
      double cosine_phi = 1.;
      double sine_phi = 0.;
      if (transverse > 0.) {
        cosine_phi = input[0] / transverse;
        sine_phi = input[1] / transverse;
      }
      auto const cosine_theta = input[2];
      auto const sine_theta = transverse;
      double const rotation_x[3]{cosine_theta * cosine_phi, cosine_theta * sine_phi,
                                 -sine_theta};
      double const rotation_y[3]{-sine_phi, cosine_phi, 0.};
      auto const bounded = lepton_final_state_detail::lepton_maximum(
          -1., lepton_final_state_detail::lepton_minimum(1., cosine));
      auto const sine = lepton_final_state_detail::lepton_sqrt(
          lepton_final_state_detail::lepton_maximum(0., (1. - bounded) * (1. + bounded)));
      auto const local_x = sine * lepton_final_state_detail::lepton_cos(azimuth);
      auto const local_y = sine * lepton_final_state_detail::lepton_sin(azimuth);
      auto local_z = lepton_final_state_detail::lepton_sqrt(
          lepton_final_state_detail::lepton_maximum(
              0., 1. - local_x * local_x - local_y * local_y));
      if (bounded < 0.) local_z = -local_z;
      for (int axis = 0; axis < 3; ++axis) {
        output[axis] = local_z * input[axis] + local_x * rotation_x[axis] +
                       local_y * rotation_y[axis];
      }
    }
    CORSIKA_HOST_DEVICE inline bool normalize_lepton_direction(double vector[3]) {
      auto const norm_squared =
          vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
      if (!lepton_final_state_detail::lepton_finite(norm_squared) || !(norm_squared > 0.))
        return false;
      auto const inverse_norm = 1. / lepton_final_state_detail::lepton_sqrt(norm_squared);
      for (int axis = 0; axis < 3; ++axis) vector[axis] *= inverse_norm;
      return lepton_final_state_detail::lepton_finite(vector[0]) &&
             lepton_final_state_detail::lepton_finite(vector[1]) &&
             lepton_final_state_detail::lepton_finite(vector[2]);
    }
    CORSIKA_HOST_DEVICE inline double effective_loss_fraction(double energy_GeV,
                                                              double sampled_fraction) {
      auto const energy_MeV = energy_GeV * 1000.;
      auto const loss_MeV = sampled_fraction * energy_MeV;
      return loss_MeV / energy_MeV;
    }
    CORSIKA_HOST_DEVICE inline double annihilation_function_term(double a1, double a2,
                                                                 double rho) {
      return a1 * lepton_final_state_detail::lepton_log(rho) + a2 / rho - rho;
    }
    CORSIKA_HOST_DEVICE inline int lepton_sign(double value) {
      return (0. < value) - (value < 0.);
    }
    CORSIKA_HOST_DEVICE inline double annihilation_equation(double value, double a1,
                                                            double a2, double lower_term,
                                                            double upper_term,
                                                            double uniform) {
      return lepton_final_state_detail::annihilation_function_term(a1, a2, value) -
             lower_term - uniform * (upper_term - lower_term);
    }
    CORSIKA_HOST_DEVICE inline bool sample_annihilation_rho(double energy_GeV,
                                                            double mass_GeV,
                                                            double uniform, double& rho) {
      auto const gamma = energy_GeV / mass_GeV;
      if (!lepton_final_state_detail::lepton_finite(gamma) || !(gamma > 1.) ||
          !lepton_final_state_detail::lepton_finite(uniform) || !(uniform > 0.) ||
          !(uniform < 1.))
        return false;
      auto const auxiliary =
          lepton_final_state_detail::lepton_sqrt((gamma - 1.) / (gamma + 1.));
      auto lower = 0.5 * (1. - auxiliary);
      auto upper = 0.5 * (1. + auxiliary);
      auto const a2 = 1. / ((gamma + 1.) * (gamma + 1.));
      auto const a1 = 1. + 2. * gamma * a2;
      auto const lower_term =
          lepton_final_state_detail::annihilation_function_term(a1, a2, lower);
      auto const upper_term =
          lepton_final_state_detail::annihilation_function_term(a1, a2, upper);
      auto lower_value = lepton_final_state_detail::annihilation_equation(
          lower, a1, a2, lower_term, upper_term, uniform);
      auto const upper_value = lepton_final_state_detail::annihilation_equation(
          upper, a1, a2, lower_term, upper_term, uniform);
      if (!lepton_final_state_detail::lepton_finite(lower_value) ||
          !lepton_final_state_detail::lepton_finite(upper_value) ||
          lower_value * upper_value > 0.)
        return false;
      auto const precision =
          lower * (energy_GeV * 1000.) * lepton_final_state_detail::ProposalHalfPrecision;
      if (!lepton_final_state_detail::lepton_finite(precision) || !(precision > 0.))
        return false;
      for (int iteration = 0; iteration <= 100; ++iteration) {
        auto const center = (lower + upper) / 2.;
        auto const center_value = lepton_final_state_detail::annihilation_equation(
            center, a1, a2, lower_term, upper_term, uniform);
        if (lepton_final_state_detail::lepton_sign(center_value) ==
            lepton_final_state_detail::lepton_sign(lower_value)) {
          lower = center;
          lower_value = center_value;
        } else {
          upper = center;
        }
        if (lepton_final_state_detail::lepton_fabs(upper - lower) < precision) {
          rho = lower;
          return lepton_final_state_detail::lepton_finite(rho) && rho > 0. && rho < 1.;
        }
      }
      rho = lower;
      return lepton_final_state_detail::lepton_finite(rho) && rho > 0. && rho < 1.;
    }
    CORSIKA_HOST_DEVICE inline FallbackRequest invalid_lepton_final_state(
        InteractionRecord const& interaction,
        LeptonFinalStateParameters const& parameters) {
      auto event =
          make_process_fallback_event(interaction, FallbackReason::InvalidFinalState);
      if (parameters.finalStateUniform_ > 0.) {
        event.finalStateUniform_ = parameters.finalStateUniform_;
        event.finalStateDrawId_ = parameters.processId_ == AnnihilationProcessId
                                      ? AnnihilationRhoDrawId
                                      : parameters.processId_ == ElectronPairProcessId
                                            ? EpairRhoDrawId
                                            : BremsAzimuthDrawId;
      } else if (parameters.azimuthUniform_ > 0.) {
        event.finalStateUniform_ = parameters.azimuthUniform_;
        event.finalStateDrawId_ = parameters.processId_ == AnnihilationProcessId
                                      ? AnnihilationAzimuthDrawId
                                      : parameters.processId_ == ElectronPairProcessId
                                            ? EpairSignDrawId
                                            : BremsAzimuthDrawId;
      }
      return event;
    }
    CORSIKA_HOST_DEVICE inline bool apply_lepton_two_child_thinning(
        EmThinningConfig const& thinning, ParticleState const& parent,
        std::int32_t process_id, double first_energy_GeV, double second_energy_GeV,
        std::uint64_t random_seed, std::uint64_t shower_id,
        LeptonFinalStateParameters& sample, std::uint32_t& child_count) {
      RandomNumberKey first_key{random_seed,
                                shower_id,
                                parent.historyId_,
                                parent.stepId_,
                                static_cast<std::uint32_t>(process_id),
                                EmThinningFirstDrawId};
      auto second_key = first_key;
      second_key.drawId_ = EmThinningSecondDrawId;
      sample.thinningFirstUniform_ = uniform_open01(first_key);
      sample.thinningSecondUniform_ = uniform_open01(second_key);
      auto effective_thinning = thinning;
      if (!is_electron_or_positron_pid(parent.pid_)) effective_thinning.enabled_ = 0;
      auto const result = apply_em_thinning(
          effective_thinning, parent.energyGeV_, parent.weight_, first_energy_GeV,
          second_energy_GeV, sample.thinningFirstUniform_, sample.thinningSecondUniform_);
      if (result.status_ == EmThinningStatus::InvalidInput) return false;
      sample.thinningStatus_ = static_cast<std::uint32_t>(result.status_);
      sample.thinningKeepMask_ = result.keepMask_;
      sample.thinningFirstWeight_ = result.firstWeight_;
      sample.thinningSecondWeight_ = result.secondWeight_;
      child_count =
          (result.keepMask_ & 0x1U ? 1U : 0U) + (result.keepMask_ & 0x2U ? 1U : 0U);
      return true;
    }
    CORSIKA_HOST_DEVICE inline BremsLpmResult prepared_brems_lpm_suppression_factor(
        BremsLpmSnapshot const& snapshot, BremsLpmPreparedSnapshot const& prepared,
        std::uint64_t component_hash, double energy_MeV, double v,
        double local_mass_density_g_per_cm3) {
      using ::corsika::kokkos::BremsLpmResult;
      using ::corsika::kokkos::BremsLpmStatus;
      if (prepared.valid_ == 0) return {BremsLpmStatus::InvalidSnapshot, 0, 0.};
      if (!lepton_final_state_detail::lepton_finite(energy_MeV) ||
          !(energy_MeV > snapshot.leptonMassMeV_) ||
          !lepton_final_state_detail::lepton_finite(v) || !(v > 0.) || !(v < 1.) ||
          !lepton_final_state_detail::lepton_finite(local_mass_density_g_per_cm3) ||
          !(local_mass_density_g_per_cm3 > 0.))
        return {BremsLpmStatus::InvalidInput, 0, 0.};
      BremsLpmPreparedComponent component{};
      bool found = false;
      for (std::uint32_t index = 0; index < prepared.componentCount_; ++index) {
        if (prepared.components_[index].componentHash_ == component_hash) {
          component = prepared.components_[index];
          found = true;
          break;
        }
      }
      if (!found) return {BremsLpmStatus::ComponentNotFound, 0, 0.};
      constexpr double Fi1 = 1.54954;
      constexpr double G1 = 0.710390;
      constexpr double G2 = 0.904912;
      auto const density_correction =
          local_mass_density_g_per_cm3 / snapshot.baselineMassDensityGPerCm3_;
      auto const sp = 0.125 * lepton_final_state_detail::lepton_sqrt(
                                  snapshot.eLpmMeV_ * v /
                                  (density_correction * energy_MeV * (1. - v)));
      auto const h = lepton_final_state_detail::lepton_log(sp) / component.logarithmS1_;
      double xi = 1.;
      if (sp < component.s1_) {
        xi = 2.;
      } else if (sp < 1.) {
        xi = 1. + h -
             0.08 * (1. - h) * (1. - (1. - h) * (1. - h)) / component.logarithmS1_;
      }
      auto gamma = snapshot.classicalElectronRadiusCm_ * snapshot.electronMassMeV_ /
                   (snapshot.fineStructureConstant_ * snapshot.leptonMassMeV_ * v);
      gamma = 1. + 4. * lepton_final_state_detail::LeptonPi * snapshot.sumCharge_ *
                       snapshot.classicalElectronRadiusCm_ * gamma * gamma *
                       snapshot.molecularDensityPerCm3_ * density_correction;
      auto const s = sp / lepton_final_state_detail::lepton_sqrt(xi) * gamma;
      auto const s2 = s * s;
      double fi = 0.;
      if (s < Fi1) {
        fi = 1. - lepton_final_state_detail::lepton_exp(
                      -6. * s * (1. + (3. - lepton_final_state_detail::LeptonPi) * s) +
                      s2 * s / (0.623 + 0.796 * s + 0.658 * s2));
      } else {
        fi = 1. - 0.012 / (s2 * s2);
      }
      double g = 0.;
      if (s < G1) {
        auto const psi =
            1. -
            lepton_final_state_detail::lepton_exp(
                -4. * s -
                8. * s2 / (1. + 3.936 * s + 4.97 * s2 - 0.05 * s2 * s + 7.50 * s2 * s2));
        g = 3. * psi - 2. * fi;
      } else if (s < G2) {
        g = 36. * s2 / (36. * s2 + 1.);
      } else {
        g = 1. - 0.022 / (s2 * s2);
      }
      auto const probability =
          ((xi / 3.) *
           (v * v * g / (gamma * gamma) + 2. * (1. + (1. - v) * (1. - v)) * fi / gamma)) /
          ((4. / 3.) * (1. - v) + v * v);
      if (!lepton_final_state_detail::lepton_finite(probability) || probability < 0.)
        return {BremsLpmStatus::NonFiniteResult, 0, 0.};
      return {BremsLpmStatus::Success, 0, probability};
    }
    template <bool UsePreparedLpm, int EpairMaximumTrials>
    CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification
    classify_lepton_final_state_impl(BremsLpmSnapshot const& lpm_snapshot,
                                     BremsLpmPreparedSnapshot const& prepared_lpm,
                                     EmThinningConfig const& thinning,
                                     InteractionRecord const& interaction,
                                     std::uint64_t random_seed, std::uint64_t shower_id) {
      using namespace corsika::kokkos;
      LeptonFinalStateClassification result{};
      if (interaction.status_ == InteractionStatus::AtRestAnnihilation) {
        auto const& parent = interaction.particle_;
        if (parent.pid_ != static_cast<std::int32_t>(EmPid::Positron) ||
            parent.energyGeV_ != transport_mass_gev(parent.pid_) ||
            !(parent.weight_ > 0.) ||
            !lepton_final_state_detail::lepton_finite(parent.weight_) ||
            parent.generation_ == 0xffffffffU ||
            parent.stepId_ == 0xffffffffffffffffULL) {
          result.fallback_ =
              lepton_final_state_detail::invalid_lepton_final_state(interaction);
          result.fallbackFlag_ = 1;
          return result;
        }
        auto& sample = result.parameters_;
        sample.processId_ = AnnihilationProcessId;
        sample.energySplitFraction_ = 0.5;
        sample.finalStateUniform_ =
            uniform_open01({random_seed, shower_id, parent.historyId_, parent.stepId_,
                            static_cast<std::uint32_t>(AnnihilationProcessId),
                            AtRestAnnihilationPolarDrawId});
        sample.azimuthUniform_ =
            uniform_open01({random_seed, shower_id, parent.historyId_, parent.stepId_,
                            static_cast<std::uint32_t>(AnnihilationProcessId),
                            AtRestAnnihilationAzimuthDrawId});
        sample.thinningFirstWeight_ = parent.weight_;
        sample.thinningSecondWeight_ = parent.weight_;
        result.childCount_ = 2;
        result.recordFlag_ = 1;
        result.annihilationFlag_ = 1;
        return result;
      }
      if (interaction.status_ == InteractionStatus::NoDiscreteInteraction) {
        result.continuationFlag_ = 1;
        return result;
      }
      if (interaction.status_ != InteractionStatus::Selected) {
        result.fallback_ =
            lepton_final_state_detail::invalid_lepton_final_state(interaction);
        result.fallbackFlag_ = 1;
        return result;
      }
      auto const capability =
          process_capability(interaction.particle_.pid_, interaction.processId_);
      if (capability != ProcessCapability::Bremsstrahlung &&
          capability != ProcessCapability::Annihilation &&
          capability != ProcessCapability::Ionization) {
        result.fallback_ =
            make_process_fallback_event(interaction, process_fallback_reason(capability));
        result.fallbackFlag_ = 1;
        return result;
      }
      auto const& parent = interaction.particle_;
      auto const v = interaction.energyFraction_;
      // The parent may be a muon; the produced pair is always e- / e+.
      auto const electron_mass_GeV = lpm_snapshot.electronMassMeV_ / 1000.;
      auto const lepton_mass_GeV = interaction.particleMassGeV_ > 0.
                                       ? interaction.particleMassGeV_
                                       : lpm_snapshot.leptonMassMeV_ / 1000.;
      if (!lepton_final_state_detail::lepton_finite(parent.energyGeV_) ||
          !(parent.energyGeV_ > lepton_mass_GeV) || parent.generation_ == 0xffffffffU ||
          parent.stepId_ == 0xffffffffffffffffULL) {
        result.fallback_ =
            lepton_final_state_detail::invalid_lepton_final_state(interaction);
        result.fallbackFlag_ = 1;
        return result;
      }
      auto& sample = result.parameters_;
      sample.processId_ = interaction.processId_;
      if (capability == ProcessCapability::Annihilation) {
        if (!lepton_final_state_detail::lepton_finite(v) ||
            lepton_final_state_detail::lepton_fabs(v - 1.) > 1.e-12) {
          result.fallback_ =
              lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
          result.fallbackFlag_ = 1;
          return result;
        }
        RandomNumberKey const rho_key{random_seed,
                                      shower_id,
                                      parent.historyId_,
                                      parent.stepId_,
                                      static_cast<std::uint32_t>(AnnihilationProcessId),
                                      AnnihilationRhoDrawId};
        RandomNumberKey const azimuth_key{
            random_seed,
            shower_id,
            parent.historyId_,
            parent.stepId_,
            static_cast<std::uint32_t>(AnnihilationProcessId),
            AnnihilationAzimuthDrawId};
        sample.finalStateUniform_ = uniform_open01(rho_key);
        sample.azimuthUniform_ = uniform_open01(azimuth_key);
        if (!lepton_final_state_detail::sample_annihilation_rho(
                parent.energyGeV_, lepton_mass_GeV, sample.finalStateUniform_,
                sample.energySplitFraction_)) {
          result.fallback_ =
              lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
          result.fallbackFlag_ = 1;
          return result;
        }
        auto const total = parent.energyGeV_ + lepton_mass_GeV;
        if (!lepton_final_state_detail::apply_lepton_two_child_thinning(
                thinning, parent, AnnihilationProcessId,
                total * (1. - sample.energySplitFraction_),
                total * sample.energySplitFraction_, random_seed, shower_id, sample,
                result.childCount_)) {
          result.fallback_ =
              lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
          result.fallbackFlag_ = 1;
          return result;
        }
        result.recordFlag_ = 1;
        result.annihilationFlag_ = 1;
        return result;
      }
      if (capability == ProcessCapability::Ionization) {
        auto const effective_v =
            lepton_final_state_detail::effective_loss_fraction(parent.energyGeV_, v);
        auto const outgoing = parent.energyGeV_ * (1. - effective_v);
        auto const delta = parent.energyGeV_ * effective_v + electron_mass_GeV;
        if (!lepton_final_state_detail::lepton_finite(effective_v) ||
            !(effective_v > 0.) || !(effective_v < 1.) ||
            !lepton_final_state_detail::lepton_finite(outgoing) ||
            outgoing < lepton_mass_GeV ||
            !lepton_final_state_detail::lepton_finite(delta) ||
            !(delta > electron_mass_GeV)) {
          result.fallback_ =
              lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
          result.fallbackFlag_ = 1;
          return result;
        }
        RandomNumberKey const azimuth_key{random_seed,
                                          shower_id,
                                          parent.historyId_,
                                          parent.stepId_,
                                          static_cast<std::uint32_t>(IonizationProcessId),
                                          IonizationAzimuthDrawId};
        sample.azimuthUniform_ = uniform_open01(azimuth_key);
        if (!lepton_final_state_detail::apply_lepton_two_child_thinning(
                thinning, parent, IonizationProcessId,
                proposal_secondary_transport_energy_gev(parent.pid_, outgoing,
                                                        lepton_mass_GeV),
                proposal_secondary_transport_energy_gev(11, delta, electron_mass_GeV),
                random_seed, shower_id, sample, result.childCount_)) {
          result.fallback_ =
              lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
          result.fallbackFlag_ = 1;
          return result;
        }
        result.recordFlag_ = 1;
        result.ionizationFlag_ = 1;
        return result;
      }
      auto const photon_energy = parent.energyGeV_ * v;
      auto const lepton_energy = parent.energyGeV_ - photon_energy;
      if (!lepton_final_state_detail::lepton_finite(v) || !(v > 0.) || !(v < 1.) ||
          !lepton_final_state_detail::lepton_finite(photon_energy) ||
          !(photon_energy > 0.) ||
          !lepton_final_state_detail::lepton_finite(lepton_energy) ||
          lepton_energy < lepton_mass_GeV) {
        result.fallback_ =
            lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
        result.fallback_.diagnosticValue0_ = v;
        result.fallback_.diagnosticValue1_ = interaction.lossQuantile_;
        result.fallback_.diagnosticValue2_ = lepton_energy;
        result.fallbackFlag_ = 1;
        return result;
      }
      RandomNumberKey const azimuth_key{random_seed,
                                        shower_id,
                                        parent.historyId_,
                                        parent.stepId_,
                                        static_cast<std::uint32_t>(BremsProcessId),
                                        BremsAzimuthDrawId};
      sample.azimuthUniform_ = uniform_open01(azimuth_key);
      auto const lpm =
          UsePreparedLpm
              ? lepton_final_state_detail::prepared_brems_lpm_suppression_factor(
                    lpm_snapshot, prepared_lpm, interaction.componentHash_,
                    parent.energyGeV_ * 1000., v, interaction.massDensityGPerCm3_)
              : brems_lpm_suppression_factor(lpm_snapshot, interaction.componentHash_,
                                             parent.energyGeV_ * 1000., v,
                                             interaction.massDensityGPerCm3_);
      if (lpm.status_ != BremsLpmStatus::Success) {
        auto const reason = lpm.status_ == BremsLpmStatus::InvalidInput
                                ? FallbackReason::InvalidMassDensity
                                : FallbackReason::LpmParametersUnavailable;
        result.fallback_ = make_process_fallback_event(interaction, reason);
        result.fallback_.finalStateUniform_ = sample.azimuthUniform_;
        result.fallback_.finalStateDrawId_ = BremsAzimuthDrawId;
        result.fallbackFlag_ = 1;
        return result;
      }
      sample.lpmSurvivalProbability_ = lpm.survivalProbability_;
      auto lpm_key = azimuth_key;
      lpm_key.drawId_ = BremsLpmDrawId;
      sample.lpmUniform_ = uniform_open01(lpm_key);
      if (sample.lpmUniform_ > sample.lpmSurvivalProbability_) {
        result.suppressionFlag_ = 1;
        result.bremsSuppressionFlag_ = 1;
        return result;
      }
      if (!lepton_final_state_detail::apply_lepton_two_child_thinning(
              thinning, parent, BremsProcessId,
              proposal_secondary_transport_energy_gev(parent.pid_, lepton_energy,
                                                      lepton_mass_GeV),
              photon_energy, random_seed, shower_id, sample, result.childCount_)) {
        result.fallback_ =
            lepton_final_state_detail::invalid_lepton_final_state(interaction, sample);
        result.fallbackFlag_ = 1;
        return result;
      }
      result.recordFlag_ = 1;
      result.bremsFlag_ = 1;
      return result;
    }
    CORSIKA_HOST_DEVICE inline bool needs_muon_radiative_final_state(
        InteractionRecord const& interaction) {
      return is_muon_pid(interaction.particle_.pid_) &&
             interaction.processId_ == BremsProcessId &&
             interaction.status_ == InteractionStatus::Selected;
    }
    CORSIKA_HOST_DEVICE inline void capture_lepton_first_interaction(
        LeptonFinalStateMaterialization& output, ParticleState const& parent,
        std::int32_t process_id, ParticleState const& first, ParticleState const* second,
        ParticleState const* third) {
      if (parent.generation_ != 0) return;
      auto& snapshot = output.firstInteraction_;
      snapshot.parentAtVertex_ = parent;
      snapshot.processId_ = process_id;
      snapshot.secondaryCount_ = third ? 3U : (second ? 2U : 1U);
      snapshot.secondaries_[0] = first;
      snapshot.secondaries_[0].weight_ = parent.weight_;
      if (second) {
        snapshot.secondaries_[1] = *second;
        snapshot.secondaries_[1].weight_ = parent.weight_;
      }
      if (third) {
        snapshot.secondaries_[2] = *third;
        snapshot.secondaries_[2].weight_ = parent.weight_;
      }
      output.hasFirstInteraction_ = 1;
    }
    CORSIKA_HOST_DEVICE inline void populate_lepton_thinning_record(
        LeptonFinalStateRecord& record, LeptonFinalStateParameters const& sample) {
      record.thinningStatus_ = sample.thinningStatus_;
      record.thinningKeepMask_ = sample.thinningKeepMask_;
      record.thinningFirstUniform_ = sample.thinningFirstUniform_;
      record.thinningSecondUniform_ = sample.thinningSecondUniform_;
      record.thinningFirstDrawId_ = EmThinningFirstDrawId;
      record.thinningSecondDrawId_ = EmThinningSecondDrawId;
    }
    class LeptonFinalStateMaterializationWriter {
    public:
      CORSIKA_HOST_DEVICE explicit LeptonFinalStateMaterializationWriter(
          LeptonFinalStateMaterialization& output)
          : output_(output) {}

      CORSIKA_HOST_DEVICE void fallback(FallbackRequest const& value) {
        output_.fallback_ = value;
        output_.hasFallback_ = 1;
      }

      CORSIKA_HOST_DEVICE void continuation(InteractionRecord const& value) {
        output_.continuation_ = value;
        output_.hasContinuation_ = 1;
      }

      CORSIKA_HOST_DEVICE void suppression(ParticleState const& parent,
                                           InteractionRecord const& interaction,
                                           LeptonFinalStateParameters const& sample) {
        output_.suppression_ = {
            parent,
            interaction.inputIndex_,
            interaction.componentHash_,
            sample.lpmSurvivalProbability_,
            sample.lpmUniform_,
            sample.processId_ == ElectronPairProcessId ? EpairLpmDrawId : BremsLpmDrawId};
        output_.hasSuppression_ = 1;
      }

      CORSIKA_HOST_DEVICE void firstInteraction(ParticleState const& parent,
                                                std::int32_t process_id,
                                                ParticleState const& first,
                                                ParticleState const* second = nullptr,
                                                ParticleState const* third = nullptr) {
        lepton_final_state_detail::capture_lepton_first_interaction(
            output_, parent, process_id, first, second, third);
      }

      CORSIKA_HOST_DEVICE void secondary(std::uint32_t index,
                                         ParticleState const& particle) {
        output_.secondaries_[index] = particle;
      }

      CORSIKA_HOST_DEVICE LeptonFinalStateRecord& record() { return output_.record_; }

      CORSIKA_HOST_DEVICE void finishRecord(std::uint32_t secondary_count) {
        output_.secondaryCount_ = secondary_count;
        output_.hasRecord_ = 1;
      }

      CORSIKA_HOST_DEVICE void fail(std::uint32_t error) { output_.error_ = error; }

    private:
      LeptonFinalStateMaterialization& output_;
    };
  } // namespace lepton_final_state_detail
  template <int EpairMaximumTrials>
  CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification
  classify_muon_radiative_final_state(BremsLpmSnapshot const& source,
                                      EmThinningConfig const& thinning,
                                      InteractionRecord const& interaction,
                                      std::uint64_t random_seed,
                                      std::uint64_t shower_id) {
    if (interaction.processId_ == ElectronPairProcessId) {
      // CPU-only channel: do not require any device muon LPM data.
      return lepton_final_state_detail::classify_lepton_final_state_impl<false>(
          source, {}, thinning, interaction, random_seed, shower_id);
    }
    // Missing auxiliary data must never silently use the electron model.
    if (!lepton_final_state_detail::lepton_finite(source.muonELpmMeV_) ||
        !(source.muonELpmMeV_ > 0.) ||
        !lepton_final_state_detail::lepton_finite(source.muonMassMeV_) ||
        !(source.muonMassMeV_ > 0.) ||
        !lepton_final_state_detail::lepton_finite(interaction.particleMassGeV_) ||
        lepton_final_state_detail::lepton_fabs(interaction.particleMassGeV_ * 1000. -
                                               source.muonMassMeV_) >
            1.e-10 * source.muonMassMeV_) {
      LeptonFinalStateClassification result{};
      result.fallback_ = make_process_fallback_event(
          interaction, FallbackReason::LpmParametersUnavailable);
      result.fallbackFlag_ = 1;
      return result;
    }
    auto snapshot = source;
    snapshot.leptonMassMeV_ = source.muonMassMeV_;
    snapshot.baselineMassDensityGPerCm3_ = source.muonBaselineMassDensityGPerCm3_;
    snapshot.molecularDensityPerCm3_ = source.muonMolecularDensityPerCm3_;
    snapshot.sumCharge_ = source.muonSumCharge_;
    snapshot.eLpmMeV_ = source.muonELpmMeV_;
    return lepton_final_state_detail::classify_lepton_final_state_impl<
        false, EpairMaximumTrials>(snapshot, {}, thinning, interaction, random_seed,
                                   shower_id);
  }
  CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification classify_lepton_final_state(
      BremsLpmSnapshot const& lpm_snapshot, EmThinningConfig const& thinning,
      InteractionRecord const& interaction, std::uint64_t random_seed,
      std::uint64_t shower_id) {
    if (lepton_final_state_detail::needs_muon_radiative_final_state(interaction))
      return classify_muon_radiative_final_state(lpm_snapshot, thinning, interaction,
                                                 random_seed, shower_id);
    return lepton_final_state_detail::classify_lepton_final_state_impl<false>(
        lpm_snapshot, {}, thinning, interaction, random_seed, shower_id);
  }
  CORSIKA_HOST_DEVICE inline LeptonFinalStateClassification classify_lepton_final_state(
      BremsLpmSnapshot const& lpm_snapshot, BremsLpmPreparedSnapshot const& prepared_lpm,
      EmThinningConfig const& thinning, InteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id) {
    if (lepton_final_state_detail::needs_muon_radiative_final_state(interaction))
      return classify_muon_radiative_final_state(lpm_snapshot, thinning, interaction,
                                                 random_seed, shower_id);
    return lepton_final_state_detail::classify_lepton_final_state_impl<true>(
        lpm_snapshot, prepared_lpm, thinning, interaction, random_seed, shower_id);
  }
  template <class Writer>
  CORSIKA_HOST_DEVICE inline void materialize_lepton_final_state_in_place(
      InteractionRecord const& interaction,
      LeptonFinalStateClassification const& classification, std::uint64_t child_offset,
      std::uint64_t first_history_id, double electron_mass_GeV, Writer& output) {
    using namespace corsika::kokkos;
    if (classification.fallbackFlag_) {
      output.fallback(classification.fallback_);
      return;
    }
    if (classification.continuationFlag_) {
      output.continuation(interaction);
      return;
    }
    auto const sample = classification.parameters_;
    if (classification.suppressionFlag_) {
      auto parent = interaction.particle_;
      ++parent.stepId_;
      output.suppression(parent, interaction, sample);
      return;
    }
    if (!classification.recordFlag_) return;
    auto const parent = interaction.particle_;
    auto const lepton_mass_GeV = interaction.particleMassGeV_ > 0.
                                     ? interaction.particleMassGeV_
                                     : electron_mass_GeV;
    std::uint32_t secondary_count = 0;
    if (sample.processId_ == AnnihilationProcessId) {
      auto const at_rest = interaction.status_ == InteractionStatus::AtRestAnnihilation;
      auto const rho = sample.energySplitFraction_;
      auto const total = parent.energyGeV_ + lepton_mass_GeV;
      auto const first_energy = total * (1. - rho);
      auto const second_energy = total * rho;
      auto const momentum = lepton_final_state_detail::lepton_sqrt(
          (parent.energyGeV_ + lepton_mass_GeV) * (parent.energyGeV_ - lepton_mass_GeV));
      auto const first_cosine =
          at_rest ? 0. : (total * (1. - rho) - lepton_mass_GeV) / ((1. - rho) * momentum);
      auto const second_cosine =
          at_rest ? 0. : (total * rho - lepton_mass_GeV) / (rho * momentum);
      if (!lepton_final_state_detail::lepton_finite(first_energy) ||
          !lepton_final_state_detail::lepton_finite(second_energy) ||
          !(first_energy > 0.) || !(second_energy > 0.) ||
          !lepton_final_state_detail::lepton_finite(first_cosine) ||
          !lepton_final_state_detail::lepton_finite(second_cosine)) {
        output.fail(1);
        return;
      }
      auto const azimuth =
          sample.azimuthUniform_ * lepton_final_state_detail::LeptonTwoPi;
      double first_direction[3]{};
      double second_direction[3]{};
      if (at_rest) {
        positron_at_rest_direction(sample.finalStateUniform_, sample.azimuthUniform_,
                                   first_direction);
        for (int axis = 0; axis < 3; ++axis)
          second_direction[axis] = -first_direction[axis];
      } else {
        lepton_final_state_detail::deflect_lepton_child(parent.direction_, first_cosine,
                                                        azimuth, first_direction);
        lepton_final_state_detail::deflect_lepton_child(
            parent.direction_, second_cosine,
            lepton_final_state_detail::lepton_fmod(
                azimuth + lepton_final_state_detail::LeptonPi,
                lepton_final_state_detail::LeptonTwoPi),
            second_direction);
      }
      auto first = parent;
      first.pid_ = static_cast<std::int32_t>(EmPid::Photon);
      first.energyGeV_ = first_energy;
      first.parentHistoryId_ = parent.historyId_;
      first.generation_ = parent.generation_ + 1;
      first.stepId_ = 0;
      first.reserved_ = 0;
      first.weight_ = sample.thinningFirstWeight_;
      auto second = first;
      second.energyGeV_ = second_energy;
      second.weight_ = sample.thinningSecondWeight_;
      for (int axis = 0; axis < 3; ++axis) {
        first.direction_[axis] = first_direction[axis];
        second.direction_[axis] = second_direction[axis];
      }
      output.firstInteraction(parent, AnnihilationProcessId, first, &second);
      if (sample.thinningKeepMask_ & 0x1U) {
        first.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, first);
      }
      if (sample.thinningKeepMask_ & 0x2U) {
        second.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, second);
      }
      output.record() = {
          interaction.inputIndex_,
          parent.historyId_,
          child_offset,
          secondary_count,
          AnnihilationProcessId,
          rho,
          sample.finalStateUniform_,
          sample.azimuthUniform_,
          0.,
          1.,
          0.,
          at_rest ? AtRestAnnihilationPolarDrawId : AnnihilationRhoDrawId,
          at_rest ? AtRestAnnihilationAzimuthDrawId : AnnihilationAzimuthDrawId,
          0,
          0};
      lepton_final_state_detail::populate_lepton_thinning_record(output.record(), sample);
      output.finishRecord(secondary_count);
      return;
    }
    if (sample.processId_ == IonizationProcessId) {
      auto const v = lepton_final_state_detail::effective_loss_fraction(
          parent.energyGeV_, interaction.energyFraction_);
      auto const outgoing_energy = parent.energyGeV_ * (1. - v);
      auto const delta_energy = parent.energyGeV_ * v + electron_mass_GeV;
      auto const incoming_momentum = lepton_final_state_detail::lepton_sqrt(
          (parent.energyGeV_ + lepton_mass_GeV) * (parent.energyGeV_ - lepton_mass_GeV));
      auto const outgoing_momentum = lepton_final_state_detail::lepton_sqrt(
          (outgoing_energy + lepton_mass_GeV) * (outgoing_energy - lepton_mass_GeV));
      auto const delta_momentum = lepton_final_state_detail::lepton_sqrt(
          (delta_energy + electron_mass_GeV) * (delta_energy - electron_mass_GeV));
      auto const outgoing_cosine =
          ((parent.energyGeV_ + electron_mass_GeV) * outgoing_energy -
           parent.energyGeV_ * electron_mass_GeV - lepton_mass_GeV * lepton_mass_GeV) /
          (incoming_momentum * outgoing_momentum);
      auto const delta_cosine = ((parent.energyGeV_ + electron_mass_GeV) * delta_energy -
                                 parent.energyGeV_ * electron_mass_GeV -
                                 electron_mass_GeV * electron_mass_GeV) /
                                (incoming_momentum * delta_momentum);
      if (!lepton_final_state_detail::lepton_finite(outgoing_cosine) ||
          !lepton_final_state_detail::lepton_finite(delta_cosine)) {
        output.fail(1);
        return;
      }
      auto const azimuth =
          sample.azimuthUniform_ * lepton_final_state_detail::LeptonTwoPi;
      double outgoing_direction[3]{};
      double delta_direction[3]{};
      lepton_final_state_detail::deflect_lepton_child(parent.direction_, outgoing_cosine,
                                                      azimuth, outgoing_direction);
      lepton_final_state_detail::deflect_lepton_child(
          parent.direction_, delta_cosine,
          lepton_final_state_detail::lepton_fmod(
              azimuth + lepton_final_state_detail::LeptonPi,
              lepton_final_state_detail::LeptonTwoPi),
          delta_direction);
      auto outgoing = parent;
      outgoing.energyGeV_ = proposal_secondary_transport_energy_gev(
          outgoing.pid_, outgoing_energy, lepton_mass_GeV);
      outgoing.parentHistoryId_ = parent.historyId_;
      outgoing.generation_ = parent.generation_ + 1;
      outgoing.stepId_ = 0;
      outgoing.reserved_ = 0;
      outgoing.weight_ = sample.thinningFirstWeight_;
      auto delta = outgoing;
      delta.pid_ = static_cast<std::int32_t>(EmPid::Electron);
      delta.energyGeV_ = proposal_secondary_transport_energy_gev(delta.pid_, delta_energy,
                                                                 electron_mass_GeV);
      delta.weight_ = sample.thinningSecondWeight_;
      for (int axis = 0; axis < 3; ++axis) {
        outgoing.direction_[axis] = outgoing_direction[axis];
        delta.direction_[axis] = delta_direction[axis];
      }
      output.firstInteraction(parent, IonizationProcessId, outgoing, &delta);
      if (sample.thinningKeepMask_ & 0x1U) {
        outgoing.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, outgoing);
      }
      if (sample.thinningKeepMask_ & 0x2U) {
        delta.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, delta);
      }
      output.record() = {interaction.inputIndex_,
                         parent.historyId_,
                         child_offset,
                         secondary_count,
                         IonizationProcessId,
                         interaction.energyFraction_,
                         0.,
                         sample.azimuthUniform_,
                         0.,
                         1.,
                         0.,
                         0,
                         IonizationAzimuthDrawId,
                         0,
                         0};
      lepton_final_state_detail::populate_lepton_thinning_record(output.record(), sample);
      output.record().weightedMassConventionCorrectionGeV_ =
          weighted_proposal_mass_correction_gev(
              parent.pid_, lepton_mass_GeV,
              (sample.thinningKeepMask_ & 0x1U) ? sample.thinningFirstWeight_ : 0.) +
          weighted_proposal_mass_correction_gev(
              11, electron_mass_GeV,
              (sample.thinningKeepMask_ & 0x2U) ? sample.thinningSecondWeight_ : 0.);
      output.finishRecord(secondary_count);
      return;
    }
    if (sample.processId_ == ElectronPairProcessId) {
      auto const v = lepton_final_state_detail::effective_loss_fraction(
          parent.energyGeV_, interaction.energyFraction_);
      auto const loss_energy = parent.energyGeV_ * v;
      auto const rho = sample.energySplitFraction_;
      auto const surviving_energy = parent.energyGeV_ - loss_energy;
      auto const electron_energy = 0.5 * loss_energy * (1. + rho);
      auto const positron_energy = 0.5 * loss_energy * (1. - rho);
      if (!lepton_final_state_detail::lepton_finite(surviving_energy) ||
          surviving_energy < lepton_mass_GeV ||
          !lepton_final_state_detail::lepton_finite(electron_energy) ||
          electron_energy < electron_mass_GeV ||
          !lepton_final_state_detail::lepton_finite(positron_energy) ||
          positron_energy < electron_mass_GeV) {
        output.fail(1);
        return;
      }
      auto surviving = parent;
      surviving.energyGeV_ = proposal_secondary_transport_energy_gev(
          surviving.pid_, surviving_energy, lepton_mass_GeV);
      surviving.parentHistoryId_ = parent.historyId_;
      surviving.historyId_ = first_history_id + child_offset;
      surviving.generation_ = parent.generation_ + 1;
      surviving.stepId_ = 0;
      surviving.reserved_ = 0;
      auto electron = surviving;
      electron.pid_ = static_cast<std::int32_t>(EmPid::Electron);
      electron.energyGeV_ = proposal_secondary_transport_energy_gev(
          electron.pid_, electron_energy, electron_mass_GeV);
      electron.historyId_ = first_history_id + child_offset + 1;
      auto positron = surviving;
      positron.pid_ = static_cast<std::int32_t>(EmPid::Positron);
      positron.energyGeV_ = proposal_secondary_transport_energy_gev(
          positron.pid_, positron_energy, electron_mass_GeV);
      positron.historyId_ = first_history_id + child_offset + 2;
      output.firstInteraction(parent, ElectronPairProcessId, surviving, &electron,
                              &positron);
      output.secondary(0, surviving);
      output.secondary(1, electron);
      output.secondary(2, positron);
      secondary_count = 3;
      output.record() = {interaction.inputIndex_,
                         parent.historyId_,
                         child_offset,
                         3,
                         ElectronPairProcessId,
                         interaction.energyFraction_,
                         sample.finalStateUniform_,
                         sample.azimuthUniform_,
                         sample.auxiliaryUniform_,
                         sample.lpmSurvivalProbability_,
                         sample.lpmUniform_,
                         EpairRhoDrawId,
                         EpairSignDrawId,
                         EpairDirectionDrawId,
                         EpairLpmDrawId};
      output.record().weightedMassConventionCorrectionGeV_ =
          weighted_proposal_mass_correction_gev(parent.pid_, lepton_mass_GeV,
                                                parent.weight_) +
          weighted_proposal_mass_correction_gev(11, electron_mass_GeV,
                                                2. * parent.weight_);
      output.finishRecord(secondary_count);
      return;
    }
    auto const photon_energy = parent.energyGeV_ * interaction.energyFraction_;
    auto const lepton_energy = parent.energyGeV_ - photon_energy;
    auto const azimuth = sample.azimuthUniform_ * lepton_final_state_detail::LeptonTwoPi;
    auto const photon_cosine =
        lepton_final_state_detail::lepton_cos(lepton_mass_GeV / parent.energyGeV_);
    double photon_direction[3]{};
    lepton_final_state_detail::deflect_lepton_child(parent.direction_, photon_cosine,
                                                    azimuth, photon_direction);
    auto lepton = parent;
    lepton.energyGeV_ = proposal_secondary_transport_energy_gev(
        lepton.pid_, lepton_energy, lepton_mass_GeV);
    lepton.parentHistoryId_ = parent.historyId_;
    lepton.generation_ = parent.generation_ + 1;
    lepton.stepId_ = 0;
    lepton.reserved_ = 0;
    lepton.weight_ = sample.thinningFirstWeight_;
    auto const parent_momentum =
        lepton_final_state_detail::lepton_sqrt(lepton_final_state_detail::lepton_maximum(
            0., (parent.energyGeV_ + lepton_mass_GeV) *
                    (parent.energyGeV_ - lepton_mass_GeV)));
    for (int axis = 0; axis < 3; ++axis) {
      lepton.direction_[axis] = parent.direction_[axis] * parent_momentum -
                                photon_direction[axis] * photon_energy;
    }
    if (!lepton_final_state_detail::normalize_lepton_direction(lepton.direction_)) {
      output.fail(1);
      return;
    }
    auto photon = parent;
    photon.pid_ = static_cast<std::int32_t>(EmPid::Photon);
    photon.energyGeV_ = photon_energy;
    photon.parentHistoryId_ = parent.historyId_;
    photon.generation_ = parent.generation_ + 1;
    photon.stepId_ = 0;
    photon.reserved_ = 0;
    photon.weight_ = sample.thinningSecondWeight_;
    for (int axis = 0; axis < 3; ++axis) photon.direction_[axis] = photon_direction[axis];
    output.firstInteraction(parent, BremsProcessId, lepton, &photon);
    if (sample.thinningKeepMask_ & 0x1U) {
      lepton.historyId_ = first_history_id + child_offset + secondary_count;
      output.secondary(secondary_count++, lepton);
    }
    if (sample.thinningKeepMask_ & 0x2U) {
      photon.historyId_ = first_history_id + child_offset + secondary_count;
      output.secondary(secondary_count++, photon);
    }
    output.record() = {interaction.inputIndex_,
                       parent.historyId_,
                       child_offset,
                       secondary_count,
                       BremsProcessId,
                       interaction.energyFraction_,
                       0.,
                       sample.azimuthUniform_,
                       0.,
                       sample.lpmSurvivalProbability_,
                       sample.lpmUniform_,
                       0,
                       BremsAzimuthDrawId,
                       0,
                       BremsLpmDrawId};
    lepton_final_state_detail::populate_lepton_thinning_record(output.record(), sample);
    output.record().weightedMassConventionCorrectionGeV_ =
        weighted_proposal_mass_correction_gev(
            parent.pid_, lepton_mass_GeV,
            (sample.thinningKeepMask_ & 0x1U) ? sample.thinningFirstWeight_ : 0.);
    output.finishRecord(secondary_count);
  }
  CORSIKA_HOST_DEVICE inline LeptonFinalStateMaterialization
  materialize_lepton_final_state(InteractionRecord const& interaction,
                                 LeptonFinalStateClassification const& classification,
                                 std::uint64_t child_offset,
                                 std::uint64_t first_history_id,
                                 double electron_mass_GeV) {
    LeptonFinalStateMaterialization output{};
    lepton_final_state_detail::LeptonFinalStateMaterializationWriter writer{output};
    materialize_lepton_final_state_in_place(interaction, classification, child_offset,
                                            first_history_id, electron_mass_GeV, writer);
    return output;
  }

} // namespace corsika::kokkos
