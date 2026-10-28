/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace photon_final_state_detail {
    CORSIKA_HOST_DEVICE inline bool photon_finite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double photon_sqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::sqrt(value);
#else
      return std::sqrt(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double photon_cos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::cos(value);
#else
      return std::cos(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double photon_sin(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::sin(value);
#else
      return std::sin(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double photon_fabs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::fabs(value);
#else
      return std::fabs(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double photon_fmod(double left, double right) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return ::fmod(left, right);
#else
      return std::fmod(left, right);
#endif
    }
    CORSIKA_HOST_DEVICE inline double photon_maximum(double left, double right) {
      return left > right ? left : right;
    }
    CORSIKA_HOST_DEVICE inline double bounded_cosine(double value) {
      return photon_final_state_detail::photon_maximum(-1., value < 1. ? value : 1.);
    }
    CORSIKA_HOST_DEVICE inline double sauter_cosine(double energy_GeV, double uniform) {
      auto const mass_squared = ElectronMassGeV * ElectronMassGeV;
      auto const momentum = photon_final_state_detail::photon_sqrt(
          photon_final_state_detail::photon_maximum(
              0., energy_GeV * energy_GeV - mass_squared));
      auto const coordinate = 2. * uniform - 1.;
      auto const energy_minus_momentum = mass_squared / (energy_GeV + momentum);
      auto const one_minus_cosine = energy_minus_momentum * (1. - coordinate) /
                                    (momentum * coordinate + energy_GeV);
      return photon_final_state_detail::bounded_cosine(1. - one_minus_cosine);
    }
    CORSIKA_HOST_DEVICE inline bool compton_cosines(double energy_GeV, double v,
                                                    double& photon_cosine,
                                                    double& electron_cosine) {
      auto const scattered_energy = energy_GeV * (1. - v);
      auto const electron_momentum_squared =
          2. * v * energy_GeV * ElectronMassGeV + v * v * energy_GeV * energy_GeV;
      if (!photon_final_state_detail::photon_finite(energy_GeV) || !(energy_GeV > 0.) ||
          !photon_final_state_detail::photon_finite(v) || !(v > 0.) || !(v < 1.) ||
          !(scattered_energy > 0.) || !(electron_momentum_squared > 0.)) {
        return false;
      }
      photon_cosine = 1. - (v * ElectronMassGeV) / scattered_energy;
      electron_cosine = v * (energy_GeV + ElectronMassGeV) /
                        photon_final_state_detail::photon_sqrt(electron_momentum_squared);
      auto constexpr tolerance = 1.e-12;
      if (!photon_final_state_detail::photon_finite(photon_cosine) ||
          !photon_final_state_detail::photon_finite(electron_cosine) ||
          photon_cosine < -1. - tolerance || photon_cosine > 1. + tolerance ||
          electron_cosine < -1. - tolerance || electron_cosine > 1. + tolerance) {
        return false;
      }
      photon_cosine = photon_final_state_detail::bounded_cosine(photon_cosine);
      electron_cosine = photon_final_state_detail::bounded_cosine(electron_cosine);
      return true;
    }
    CORSIKA_HOST_DEVICE inline bool photoelectric_energy(
        PhotonPairLpmSnapshot const& snapshot, std::uint64_t component_hash,
        double parent_energy_GeV, double& electron_total_energy_GeV,
        double& kinetic_fraction) {
      if (!photon_final_state_detail::photon_finite(parent_energy_GeV) ||
          !(parent_energy_GeV > 0.) || snapshot.componentCount_ == 0 ||
          snapshot.componentCount_ > MaxPhotonPairLpmComponents ||
          !photon_final_state_detail::photon_finite(snapshot.fineStructureConstant_) ||
          !(snapshot.fineStructureConstant_ > 0.)) {
        return false;
      }
      double nuclear_charge = 0.;
      for (std::uint32_t index = 0; index < snapshot.componentCount_; ++index) {
        if (snapshot.components_[index].componentHash_ == component_hash) {
          nuclear_charge = snapshot.components_[index].nuclearCharge_;
          break;
        }
      }
      if (!photon_final_state_detail::photon_finite(nuclear_charge) ||
          !(nuclear_charge > 0.))
        return false;
      auto const z_alpha = nuclear_charge * snapshot.fineStructureConstant_;
      auto const binding_energy_GeV = z_alpha * z_alpha * ElectronMassGeV / 2.;
      auto const kinetic_energy_GeV = parent_energy_GeV - binding_energy_GeV;
      if (!photon_final_state_detail::photon_finite(binding_energy_GeV) ||
          !photon_final_state_detail::photon_finite(kinetic_energy_GeV) ||
          !(kinetic_energy_GeV >= 0.)) {
        return false;
      }
      electron_total_energy_GeV = ElectronMassGeV + kinetic_energy_GeV;
      kinetic_fraction = kinetic_energy_GeV / parent_energy_GeV;
      return photon_final_state_detail::photon_finite(electron_total_energy_GeV) &&
             electron_total_energy_GeV >= ElectronMassGeV &&
             photon_final_state_detail::photon_finite(kinetic_fraction) &&
             kinetic_fraction >= 0. && kinetic_fraction <= 1.;
    }
    CORSIKA_HOST_DEVICE inline void deflect_photon_child(double const input[3],
                                                         double cosine, double azimuth,
                                                         double output[3]) {
      auto const transverse = photon_final_state_detail::photon_sqrt(
          photon_final_state_detail::photon_maximum(
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
      auto const sine = photon_final_state_detail::photon_sqrt(
          photon_final_state_detail::photon_maximum(0., (1. - cosine) * (1. + cosine)));
      auto const local_x = sine * photon_final_state_detail::photon_cos(azimuth);
      auto const local_y = sine * photon_final_state_detail::photon_sin(azimuth);
      auto local_z = photon_final_state_detail::photon_sqrt(
          photon_final_state_detail::photon_maximum(
              0., 1. - local_x * local_x - local_y * local_y));
      if (cosine < 0.) local_z = -local_z;
      for (int axis = 0; axis < 3; ++axis) {
        output[axis] = local_z * input[axis] + local_x * rotation_x[axis] +
                       local_y * rotation_y[axis];
      }
    }
    CORSIKA_HOST_DEVICE inline FallbackRequest invalid_photon_final_state(
        InteractionRecord const& interaction,
        PhotonFinalStateParameters const& parameters) {
      auto event =
          make_process_fallback_event(interaction, FallbackReason::InvalidFinalState);
      if (parameters.splitUniform_ > 0.) {
        event.finalStateUniform_ = parameters.splitUniform_;
        event.finalStateDrawId_ = parameters.splitDrawId_;
      }
      return event;
    }
    CORSIKA_HOST_DEVICE inline bool apply_photon_two_child_thinning(
        EmThinningConfig const& thinning, ParticleState const& parent,
        std::int32_t process_id, double first_energy_GeV, double second_energy_GeV,
        std::uint64_t random_seed, std::uint64_t shower_id,
        PhotonFinalStateParameters& sample, std::uint32_t& child_count) {
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
      auto const result = apply_em_thinning(
          thinning, parent.energyGeV_, parent.weight_, first_energy_GeV,
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
    CORSIKA_HOST_DEVICE inline void fill_first_interaction(
        ParticleState const& parent, std::int32_t process_id, ParticleState const& first,
        ParticleState const* second, PhotonFinalStateMaterialization& output) {
      if (parent.generation_ != 0) return;
      output.hasFirstInteraction_ = 1;
      output.firstInteraction_.parentAtVertex_ = parent;
      output.firstInteraction_.processId_ = process_id;
      output.firstInteraction_.secondaryCount_ = second == nullptr ? 1U : 2U;
      output.firstInteraction_.secondaries_[0] = first;
      output.firstInteraction_.secondaries_[0].weight_ = parent.weight_;
      if (second != nullptr) {
        output.firstInteraction_.secondaries_[1] = *second;
        output.firstInteraction_.secondaries_[1].weight_ = parent.weight_;
      }
    }
  } // namespace photon_final_state_detail

  namespace photon_final_state_detail {
    class PhotonFinalStateMaterializationWriter {
    public:
      CORSIKA_HOST_DEVICE explicit PhotonFinalStateMaterializationWriter(
          PhotonFinalStateMaterialization& output)
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
                                           PhotonFinalStateParameters const& sample) {
        output_.suppression_ = {parent,
                                interaction.inputIndex_,
                                interaction.componentHash_,
                                sample.lpmSurvivalProbability_,
                                sample.lpmUniform_,
                                PhotonPairLpmDrawId};
        output_.hasSuppression_ = 1;
      }

      CORSIKA_HOST_DEVICE void firstInteraction(ParticleState const& parent,
                                                std::int32_t process_id,
                                                ParticleState const& first,
                                                ParticleState const* second) {
        photon_final_state_detail::fill_first_interaction(parent, process_id, first,
                                                          second, output_);
      }

      CORSIKA_HOST_DEVICE void secondary(std::uint32_t index,
                                         ParticleState const& particle) {
        output_.secondaries_[index] = particle;
      }

      CORSIKA_HOST_DEVICE PhotonFinalStateRecord& record() { return output_.record_; }

      CORSIKA_HOST_DEVICE void finishRecord(std::uint32_t secondary_count) {
        output_.secondaryCount_ = secondary_count;
        output_.hasRecord_ = 1;
      }

      CORSIKA_HOST_DEVICE void fail(std::uint32_t error) { output_.error_ = error; }

    private:
      PhotonFinalStateMaterialization& output_;
    };
  } // namespace photon_final_state_detail
  CORSIKA_HOST_DEVICE inline PhotonFinalStateClassification classify_photon_final_state(
      ContinuousPhysicsView const& table, PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning, InteractionRecord const& interaction,
      std::uint64_t random_seed, std::uint64_t shower_id) {
    static_assert(PhotonPairSplitDrawId < PhotonPairAnalyticCandidateDrawIdBase);
    static_assert(PhotonPairAnalyticCandidateDrawIdBase +
                      PhotonPairAnalyticMaximumAttempts <=
                  PhotonPairAnalyticAcceptanceDrawIdBase);
    static_assert(PhotonPairAnalyticAcceptanceDrawIdBase +
                      PhotonPairAnalyticMaximumAttempts <=
                  EmThinningFirstDrawId);
    PhotonFinalStateClassification result{};
    if (interaction.status_ == InteractionStatus::NoDiscreteInteraction) {
      result.continuationFlag_ = 1;
      return result;
    }
    if (interaction.status_ != InteractionStatus::Selected) {
      result.fallback_ =
          photon_final_state_detail::invalid_photon_final_state(interaction);
      result.fallbackFlag_ = 1;
      return result;
    }
    if (interaction.deferredPhotonFallbackReason_ >= 0) {
      auto const reason =
          static_cast<FallbackReason>(interaction.deferredPhotonFallbackReason_);
      if (interaction.interactionVertexReached_ != 1 ||
          !proposal_fallback_requires_selected_loss(reason)) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction);
      } else {
        result.fallback_ = make_process_fallback_event(interaction, reason);
        if (interaction.deferredPhotonTableStatus_ != 0) {
          result.fallback_.diagnosticStatus_ = interaction.deferredPhotonTableStatus_;
          result.fallback_.diagnosticValue0_ = interaction.particle_.energyGeV_ * 1000.;
          result.fallback_.diagnosticValue1_ = interaction.lossQuantile_;
          result.fallback_.diagnosticValue2_ = interaction.energyFraction_;
        }
      }
      result.fallbackFlag_ = 1;
      return result;
    }
    auto const capability =
        process_capability(interaction.particle_.pid_, interaction.processId_);
    if (capability != ProcessCapability::PhotonPair &&
        capability != ProcessCapability::PhotonMuonPair &&
        capability != ProcessCapability::Compton &&
        capability != ProcessCapability::Photoelectric) {
      result.fallback_ =
          make_process_fallback_event(interaction, process_fallback_reason(capability));
      result.fallbackFlag_ = 1;
      return result;
    }
    auto const& parent = interaction.particle_;
    if (capability == ProcessCapability::PhotonMuonPair) {
      auto& sample = result.parameters_;
      sample.processId_ = PhotonMuonPairProcessId;
      sample.secondaryMassGeV_ =
          table.getNativeView().getConstants().muonMassMeV_ / 1000.;
      if (!photon_final_state_detail::photon_finite(parent.energyGeV_) ||
          !photon_final_state_detail::photon_finite(parent.weight_) ||
          parent.weight_ < 0. || !(sample.secondaryMassGeV_ > 0.) ||
          parent.generation_ == 0xffffffffU || parent.stepId_ == 0xffffffffffffffffULL) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction);
        result.fallbackFlag_ = 1;
        return result;
      }
      RandomNumberKey key{random_seed,
                          shower_id,
                          parent.historyId_,
                          parent.stepId_,
                          static_cast<std::uint32_t>(PhotonMuonPairProcessId),
                          PhotonMuonPairSplitDrawId};
      sample.splitUniform_ = uniform_open01(key);
      sample.splitDrawId_ = PhotonMuonPairSplitDrawId;
      auto const split = execute_table_query(
          table,
          {TableQueryKind::LossFraction, parent.pid_, PhotonMuonPairFinalStateProcessId,
           interaction.componentHash_, parent.energyGeV_ * 1000., sample.splitUniform_});
      if (split.status_ != TableLookupStatus::Success) {
        // A corrupt/missing/out-of-range native table is not a normal CPU
        // physics path. Fail closed with the original vertex for diagnosis.
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction, sample);
        result.fallback_.diagnosticStatus_ = static_cast<std::uint32_t>(split.status_);
        result.fallbackFlag_ = 1;
        return result;
      }
      sample.splitFraction_ = split.value_; // x = E_mu+ / E_gamma.
      auto const minus = (1. - split.value_) * parent.energyGeV_;
      auto const plus = split.value_ * parent.energyGeV_;
      if (!photon_final_state_detail::photon_finite(split.value_) || !(minus > 0.) ||
          !(plus > 0.)) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction, sample);
        result.fallbackFlag_ = 1;
        return result;
      }
      key.drawId_ = PhotonMuonPairAzimuthDrawId;
      sample.azimuthUniform_ = uniform_open01(key);
      // EMThinning::doSecondaries returns when either child is non-EM.
      // Muons therefore inherit the parent weight even when EM thinning is on.
      // CheckForLPM likewise applies no electron-pair LPM veto to PhotoMuPair.
      sample.thinningStatus_ = static_cast<std::uint32_t>(EmThinningStatus::NotApplied);
      sample.thinningKeepMask_ = 3U;
      sample.thinningFirstWeight_ = sample.thinningSecondWeight_ = parent.weight_;
      result.childCount_ = 2;
      result.recordFlag_ = 1;
      result.photonMuonPairFlag_ = 1;
      return result;
    }
    if (capability == ProcessCapability::Compton) {
      double photon_cosine = 0.;
      double electron_cosine = 0.;
      if (parent.generation_ == 0xffffffffU || parent.stepId_ == 0xffffffffffffffffULL ||
          !photon_final_state_detail::compton_cosines(parent.energyGeV_,
                                                      interaction.energyFraction_,
                                                      photon_cosine, electron_cosine)) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction);
        result.fallbackFlag_ = 1;
        return result;
      }
      auto& sample = result.parameters_;
      sample.processId_ = ComptonProcessId;
      sample.splitFraction_ = interaction.energyFraction_;
      sample.splitUniform_ = interaction.lossQuantile_;
      RandomNumberKey const azimuth_key{random_seed,
                                        shower_id,
                                        parent.historyId_,
                                        parent.stepId_,
                                        static_cast<std::uint32_t>(ComptonProcessId),
                                        ComptonAzimuthDrawId};
      sample.azimuthUniform_ = uniform_open01(azimuth_key);
      if (!photon_final_state_detail::apply_photon_two_child_thinning(
              thinning, parent, ComptonProcessId,
              parent.energyGeV_ * (1. - sample.splitFraction_),
              proposal_secondary_transport_energy_gev(
                  11, ElectronMassGeV + parent.energyGeV_ * sample.splitFraction_,
                  ElectronMassGeV),
              random_seed, shower_id, sample, result.childCount_)) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction, sample);
        result.fallbackFlag_ = 1;
        return result;
      }
      result.recordFlag_ = 1;
      result.comptonFlag_ = 1;
      return result;
    }
    if (capability == ProcessCapability::Photoelectric) {
      double electron_energy_GeV = 0.;
      double kinetic_fraction = 0.;
      if (parent.generation_ == 0xffffffffU ||
          !photon_final_state_detail::photon_finite(interaction.energyFraction_) ||
          photon_final_state_detail::photon_fabs(interaction.energyFraction_ - 1.) >
              1.e-12 ||
          !photon_final_state_detail::photoelectric_energy(
              lpm_snapshot, interaction.componentHash_, parent.energyGeV_,
              electron_energy_GeV, kinetic_fraction)) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction);
        result.fallbackFlag_ = 1;
        return result;
      }
      auto& sample = result.parameters_;
      sample.processId_ = PhotoelectricProcessId;
      sample.splitFraction_ = kinetic_fraction;
      sample.splitUniform_ = interaction.lossQuantile_;
      result.childCount_ = 1;
      result.recordFlag_ = 1;
      result.photoelectricFlag_ = 1;
      return result;
    }
    if (!photon_final_state_detail::photon_finite(parent.energyGeV_) ||
        parent.energyGeV_ < 2. * ElectronMassGeV || parent.generation_ == 0xffffffffU ||
        parent.stepId_ == 0xffffffffffffffffULL) {
      result.fallback_ =
          photon_final_state_detail::invalid_photon_final_state(interaction);
      result.fallbackFlag_ = 1;
      return result;
    }

    auto& sample = result.parameters_;
    sample.processId_ = PhotonPairProcessId;
    // Independent CDF quantile; no candidate/acceptance rejection loop.
    RandomNumberKey const split_key{random_seed,
                                    shower_id,
                                    parent.historyId_,
                                    parent.stepId_,
                                    static_cast<std::uint32_t>(PhotonPairProcessId),
                                    PhotonPairSplitDrawId};
    sample.splitUniform_ = uniform_open01(split_key);
    sample.splitDrawId_ = PhotonPairSplitDrawId;
    TableQuery const split_query{
        TableQueryKind::LossFraction,  parent.pid_,
        PhotonPairFinalStateProcessId, interaction.componentHash_,
        parent.energyGeV_ * 1000.,     sample.splitUniform_};
    auto const split = execute_table_query(table, split_query);
    if (split.status_ == TableLookupStatus::Success) {
      auto const split_ready = decode_photon_pair_normalized_split(
          parent.energyGeV_, split.value_, sample.splitFraction_);
      if (!split_ready) {
        result.fallback_ =
            photon_final_state_detail::invalid_photon_final_state(interaction, sample);
        result.fallbackFlag_ = 1;
        return result;
      }
    } else {
      auto event = make_table_fallback_event(
          parent, split_query, split, PhotonPairSplitDrawId, interaction.inputIndex_);
      event.interactionVertexReached_ = interaction.interactionVertexReached_;
      event.energyFraction_ = interaction.energyFraction_;
      event.processId_ = interaction.processId_;
      auto const native_selection = interaction.proposalSelectionRandomProcessId_ != 0u;
      event.selectionUniform_ = native_selection ? interaction.proposalSelectionUniform_
                                                 : interaction.processUniform_;
      event.lossQuantile_ = interaction.lossQuantile_;
      event.outerAcceptanceUniform_ = interaction.processUniform_;
      event.outerAcceptanceRandomProcessId_ = interaction.processRandomProcessId_;
      event.outerAcceptanceDrawId_ = interaction.processDrawId_;
      event.randomProcessId_ =
          native_selection ? interaction.proposalSelectionRandomProcessId_ : 0u;
      event.randomDrawId_ = native_selection ? interaction.proposalSelectionDrawId_
                                             : interaction.lossDrawId_;
      event.finalStateUniform_ = sample.splitUniform_;
      event.finalStateDrawId_ = PhotonPairSplitDrawId;
      result.fallback_ = event;
      result.fallbackFlag_ = 1;
      return result;
    }
    auto const electron_energy = parent.energyGeV_ * sample.splitFraction_;
    auto const positron_energy = parent.energyGeV_ - electron_energy;
    if (!photon_final_state_detail::photon_finite(sample.splitFraction_) ||
        !(sample.splitFraction_ > 0.) || !(sample.splitFraction_ < 1.) ||
        electron_energy < ElectronMassGeV || positron_energy < ElectronMassGeV) {
      result.fallback_ =
          photon_final_state_detail::invalid_photon_final_state(interaction, sample);
      result.fallbackFlag_ = 1;
      return result;
    }
    RandomNumberKey angle_key{random_seed,
                              shower_id,
                              parent.historyId_,
                              parent.stepId_,
                              static_cast<std::uint32_t>(PhotonPairProcessId),
                              PhotonPairAzimuthDrawId};
    sample.azimuthUniform_ = uniform_open01(angle_key);
    angle_key.drawId_ = PhotonPairElectronPolarDrawId;
    sample.electronPolarUniform_ = uniform_open01(angle_key);
    angle_key.drawId_ = PhotonPairPositronPolarDrawId;
    sample.positronPolarUniform_ = uniform_open01(angle_key);
    auto const lpm = photon_pair_lpm_suppression_factor(
        lpm_snapshot, interaction.componentHash_, parent.energyGeV_ * 1000.,
        sample.splitFraction_, interaction.massDensityGPerCm3_);
    if (lpm.status_ != PhotonPairLpmStatus::Success) {
      auto const reason = lpm.status_ == PhotonPairLpmStatus::InvalidInput
                              ? FallbackReason::InvalidMassDensity
                              : FallbackReason::LpmParametersUnavailable;
      result.fallback_ = make_process_fallback_event(interaction, reason);
      result.fallbackFlag_ = 1;
      return result;
    }
    sample.lpmSurvivalProbability_ = lpm.survivalProbability_;
    RandomNumberKey const lpm_key{random_seed,
                                  shower_id,
                                  parent.historyId_,
                                  parent.stepId_,
                                  static_cast<std::uint32_t>(PhotonPairProcessId),
                                  PhotonPairLpmDrawId};
    sample.lpmUniform_ = uniform_open01(lpm_key);
    if (sample.lpmUniform_ > sample.lpmSurvivalProbability_) {
      result.suppressionFlag_ = 1;
      return result;
    }
    if (!photon_final_state_detail::apply_photon_two_child_thinning(
            thinning, parent, PhotonPairProcessId,
            proposal_secondary_transport_energy_gev(11, electron_energy, ElectronMassGeV),
            proposal_secondary_transport_energy_gev(-11, positron_energy,
                                                    ElectronMassGeV),
            random_seed, shower_id, sample, result.childCount_)) {
      result.fallback_ =
          photon_final_state_detail::invalid_photon_final_state(interaction, sample);
      result.fallbackFlag_ = 1;
      return result;
    }
    result.recordFlag_ = 1;
    result.photonPairFlag_ = 1;
    return result;
  }
  template <class Writer>
  CORSIKA_HOST_DEVICE inline void materialize_photon_final_state_in_place(
      InteractionRecord const& interaction,
      PhotonFinalStateClassification const& classification, std::uint64_t child_offset,
      std::uint64_t first_history_id, Writer& output) {
    if (classification.fallbackFlag_ != 0) {
      output.fallback(classification.fallback_);
      return;
    }
    if (classification.continuationFlag_ != 0) {
      output.continuation(interaction);
      return;
    }
    if (classification.suppressionFlag_ != 0) {
      auto parent = interaction.particle_;
      ++parent.stepId_;
      auto const& sample = classification.parameters_;
      output.suppression(parent, interaction, sample);
      return;
    }
    if (classification.recordFlag_ == 0) return;

    auto const& parent = interaction.particle_;
    auto const& sample = classification.parameters_;
    std::uint32_t secondary_count = 0;
    auto const azimuth = sample.azimuthUniform_ * 6.283185307179586476925286766559;
    if (sample.processId_ == ComptonProcessId) {
      double photon_cosine = 0.;
      double electron_cosine = 0.;
      if (!photon_final_state_detail::compton_cosines(
              parent.energyGeV_, sample.splitFraction_, photon_cosine, electron_cosine)) {
        output.fail(1);
        return;
      }
      auto photon = parent;
      photon.energyGeV_ = parent.energyGeV_ * (1. - sample.splitFraction_);
      photon.parentHistoryId_ = parent.historyId_;
      photon.generation_ = parent.generation_ + 1;
      photon.stepId_ = 0;
      photon.reserved_ = 0;
      photon.weight_ = sample.thinningFirstWeight_;
      photon_final_state_detail::deflect_photon_child(parent.direction_, photon_cosine,
                                                      azimuth, photon.direction_);
      auto electron = parent;
      electron.pid_ = static_cast<std::int32_t>(EmPid::Electron);
      electron.energyGeV_ = proposal_secondary_transport_energy_gev(
          electron.pid_, ElectronMassGeV + parent.energyGeV_ * sample.splitFraction_,
          ElectronMassGeV);
      electron.parentHistoryId_ = parent.historyId_;
      electron.generation_ = parent.generation_ + 1;
      electron.stepId_ = 0;
      electron.reserved_ = 0;
      electron.weight_ = sample.thinningSecondWeight_;
      photon_final_state_detail::deflect_photon_child(
          parent.direction_, electron_cosine,
          photon_final_state_detail::photon_fmod(
              azimuth + 3.1415926535897932384626433832795,
              6.283185307179586476925286766559),
          electron.direction_);
      output.firstInteraction(parent, ComptonProcessId, photon, &electron);
      if ((sample.thinningKeepMask_ & 0x1U) != 0) {
        photon.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, photon);
      }
      if ((sample.thinningKeepMask_ & 0x2U) != 0) {
        electron.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, electron);
      }
      output.record() = {interaction.inputIndex_,
                         parent.historyId_,
                         child_offset,
                         secondary_count,
                         ComptonProcessId,
                         sample.splitFraction_,
                         interaction.lossQuantile_,
                         sample.azimuthUniform_,
                         0.,
                         0.,
                         0.,
                         0.,
                         interaction.lossDrawId_,
                         ComptonAzimuthDrawId,
                         0,
                         0,
                         0};
    } else if (sample.processId_ == PhotoelectricProcessId) {
      auto electron = parent;
      electron.pid_ = static_cast<std::int32_t>(EmPid::Electron);
      electron.energyGeV_ = proposal_secondary_transport_energy_gev(
          electron.pid_, ElectronMassGeV + parent.energyGeV_ * sample.splitFraction_,
          ElectronMassGeV);
      electron.parentHistoryId_ = parent.historyId_;
      electron.historyId_ = first_history_id + child_offset;
      electron.generation_ = parent.generation_ + 1;
      electron.stepId_ = 0;
      electron.reserved_ = 0;
      output.firstInteraction(parent, PhotoelectricProcessId, electron, nullptr);
      output.secondary(0, electron);
      secondary_count = 1;
      output.record() = {interaction.inputIndex_,
                         parent.historyId_,
                         child_offset,
                         1,
                         PhotoelectricProcessId,
                         sample.splitFraction_,
                         interaction.lossQuantile_,
                         0.,
                         0.,
                         0.,
                         0.,
                         0.,
                         interaction.lossDrawId_,
                         0,
                         0,
                         0,
                         0};
    } else if (sample.processId_ == PhotonMuonPairProcessId) {
      // PROPOSAL BKK uses equal polar angles m_mu/E_gamma and opposite
      // azimuths. This is its existing direction approximation, not an exact
      // nuclear recoil solver. Keep its (E in MeV)/MMU operation order.
      auto const k = (parent.energyGeV_ * 1000.) / (sample.secondaryMassGeV_ * 1000.);
      auto cosine = photon_final_state_detail::photon_cos(1. / k);
      if (cosine == -1.) cosine *= -1.;
      ParticleState minus = parent, plus = parent;
      minus.pid_ = static_cast<std::int32_t>(EmPid::MuonMinus);
      plus.pid_ = static_cast<std::int32_t>(EmPid::MuonPlus);
      minus.energyGeV_ = proposal_secondary_transport_energy_gev(
          minus.pid_, (1. - sample.splitFraction_) * parent.energyGeV_,
          sample.secondaryMassGeV_);
      plus.energyGeV_ = proposal_secondary_transport_energy_gev(
          plus.pid_, sample.splitFraction_ * parent.energyGeV_, sample.secondaryMassGeV_);
      minus.parentHistoryId_ = plus.parentHistoryId_ = parent.historyId_;
      minus.generation_ = plus.generation_ = parent.generation_ + 1;
      minus.stepId_ = plus.stepId_ = 0;
      minus.reserved_ = plus.reserved_ = 0;
      if (minus.energyGeV_ <= transport_mass_gev(minus.pid_))
        minus.reserved_ = NativeMuonPairCutFlag;
      if (plus.energyGeV_ <= transport_mass_gev(plus.pid_))
        plus.reserved_ = NativeMuonPairCutFlag;
      minus.weight_ = sample.thinningFirstWeight_;
      plus.weight_ = sample.thinningSecondWeight_;
      photon_final_state_detail::deflect_photon_child(parent.direction_, cosine, azimuth,
                                                      minus.direction_);
      photon_final_state_detail::deflect_photon_child(
          parent.direction_, cosine,
          photon_final_state_detail::photon_fmod(
              azimuth + 3.1415926535897932384626433832795,
              6.283185307179586476925286766559),
          plus.direction_);
      output.firstInteraction(parent, sample.processId_, minus, &plus);
      if (sample.thinningKeepMask_ & 0x1U) {
        minus.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, minus);
      }
      if (sample.thinningKeepMask_ & 0x2U) {
        plus.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, plus);
      }
      output.record() = {interaction.inputIndex_,
                         parent.historyId_,
                         child_offset,
                         secondary_count,
                         sample.processId_,
                         sample.splitFraction_,
                         sample.splitUniform_,
                         sample.azimuthUniform_,
                         0.,
                         0.,
                         0.,
                         0.,
                         sample.splitDrawId_,
                         PhotonMuonPairAzimuthDrawId,
                         0,
                         0,
                         0};
    } else {
      auto const electron_energy = parent.energyGeV_ * sample.splitFraction_;
      auto const positron_energy = parent.energyGeV_ - electron_energy;
      auto const electron_cosine = photon_final_state_detail::sauter_cosine(
          electron_energy, sample.electronPolarUniform_);
      auto const positron_cosine = photon_final_state_detail::sauter_cosine(
          positron_energy, sample.positronPolarUniform_);
      auto electron = parent;
      electron.pid_ = static_cast<std::int32_t>(EmPid::Electron);
      electron.energyGeV_ = proposal_secondary_transport_energy_gev(
          electron.pid_, electron_energy, ElectronMassGeV);
      electron.parentHistoryId_ = parent.historyId_;
      electron.generation_ = parent.generation_ + 1;
      electron.stepId_ = 0;
      electron.reserved_ = 0;
      electron.weight_ = sample.thinningFirstWeight_;
      photon_final_state_detail::deflect_photon_child(parent.direction_, electron_cosine,
                                                      azimuth, electron.direction_);
      auto positron = parent;
      positron.pid_ = static_cast<std::int32_t>(EmPid::Positron);
      positron.energyGeV_ = proposal_secondary_transport_energy_gev(
          positron.pid_, positron_energy, ElectronMassGeV);
      positron.parentHistoryId_ = parent.historyId_;
      positron.generation_ = parent.generation_ + 1;
      positron.stepId_ = 0;
      positron.reserved_ = 0;
      positron.weight_ = sample.thinningSecondWeight_;
      photon_final_state_detail::deflect_photon_child(
          parent.direction_, positron_cosine,
          photon_final_state_detail::photon_fmod(
              azimuth + 3.1415926535897932384626433832795,
              6.283185307179586476925286766559),
          positron.direction_);
      output.firstInteraction(parent, PhotonPairProcessId, electron, &positron);
      if ((sample.thinningKeepMask_ & 0x1U) != 0) {
        electron.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, electron);
      }
      if ((sample.thinningKeepMask_ & 0x2U) != 0) {
        positron.historyId_ = first_history_id + child_offset + secondary_count;
        output.secondary(secondary_count++, positron);
      }
      output.record() = {interaction.inputIndex_,
                         parent.historyId_,
                         child_offset,
                         secondary_count,
                         PhotonPairProcessId,
                         sample.splitFraction_,
                         sample.splitUniform_,
                         sample.azimuthUniform_,
                         sample.electronPolarUniform_,
                         sample.positronPolarUniform_,
                         sample.lpmSurvivalProbability_,
                         sample.lpmUniform_,
                         sample.splitDrawId_,
                         PhotonPairAzimuthDrawId,
                         PhotonPairElectronPolarDrawId,
                         PhotonPairPositronPolarDrawId,
                         PhotonPairLpmDrawId};
    }
    auto& record = output.record();
    double kept_lepton_weight = 0.;
    if (sample.processId_ == PhotoelectricProcessId) {
      kept_lepton_weight = parent.weight_;
    } else if (sample.processId_ == ComptonProcessId) {
      if (sample.thinningKeepMask_ & 0x2U)
        kept_lepton_weight = sample.thinningSecondWeight_;
    } else {
      if (sample.thinningKeepMask_ & 0x1U)
        kept_lepton_weight += sample.thinningFirstWeight_;
      if (sample.thinningKeepMask_ & 0x2U)
        kept_lepton_weight += sample.thinningSecondWeight_;
    }
    record.weightedMassConventionCorrectionGeV_ = weighted_proposal_mass_correction_gev(
        sample.processId_ == PhotonMuonPairProcessId ? 13 : 11,
        sample.processId_ == PhotonMuonPairProcessId ? sample.secondaryMassGeV_
                                                     : ElectronMassGeV,
        kept_lepton_weight);
    record.thinningStatus_ = sample.thinningStatus_;
    record.thinningKeepMask_ = sample.thinningKeepMask_;
    record.thinningFirstUniform_ = sample.thinningFirstUniform_;
    record.thinningSecondUniform_ = sample.thinningSecondUniform_;
    record.thinningFirstDrawId_ = EmThinningFirstDrawId;
    record.thinningSecondDrawId_ = EmThinningSecondDrawId;
    output.finishRecord(secondary_count);
  }
  CORSIKA_HOST_DEVICE inline PhotonFinalStateMaterialization
  materialize_photon_final_state(InteractionRecord const& interaction,
                                 PhotonFinalStateClassification const& classification,
                                 std::uint64_t child_offset,
                                 std::uint64_t first_history_id) {
    PhotonFinalStateMaterialization output{};
    photon_final_state_detail::PhotonFinalStateMaterializationWriter writer{output};
    materialize_photon_final_state_in_place(interaction, classification, child_offset,
                                            first_history_id, writer);
    return output;
  }

} // namespace corsika::kokkos
