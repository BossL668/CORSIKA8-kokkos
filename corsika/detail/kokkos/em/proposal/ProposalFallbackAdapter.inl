/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  namespace proposal_fallback_detail {

    inline proposal::ProposalInteractionRecord make_record_from_identity(
        FallbackRequest const& event, std::uint64_t randomSeed, std::uint64_t showerId) {
      if (!has_specified_proposal_interaction_identity(event)) {
        throw std::invalid_argument(
            "GPU fallback does not contain a specified PROPOSAL interaction identity");
      }
      if (event.mediumHash_ > std::numeric_limits<std::size_t>::max() ||
          event.interactionHash_ > std::numeric_limits<std::size_t>::max() ||
          event.componentHash_ > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("GPU fallback PROPOSAL hash does not fit size_t");
      }

      auto const projectileId =
          convert_from_PDG(static_cast<PDGCode>(event.particle_.pid_));
      if (projectileId == Code::Unknown) {
        throw std::invalid_argument(
            "GPU fallback contains an unknown projectile PDG code");
      }

      proposal::ProposalInteractionContext context{};
      context.projectileId_ = projectileId;
      context.mediumHash_ = static_cast<std::size_t>(event.mediumHash_);
      context.projectileEnergyMeV_ = event.particle_.energyGeV_ * 1000.;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        context.positionCm_[axis] = event.particle_.positionM_[axis] * 100.;
        context.direction_[axis] = event.particle_.direction_[axis];
      }
      context.timeS_ = event.particle_.timeS_;

      proposal::ProposalRandomKey const randomKey{
          randomSeed,
          showerId,
          event.particle_.historyId_,
          event.particle_.stepId_,
          event.randomProcessId_ != 0u ? event.randomProcessId_
                                       : static_cast<std::uint32_t>(event.processId_),
          event.randomDrawId_};
      return proposal::ProposalInteractionRecord{
          context,
          static_cast<std::size_t>(event.interactionHash_),
          static_cast<PROPOSAL::InteractionType>(event.processId_),
          static_cast<std::size_t>(event.componentHash_),
          event.energyFraction_,
          event.selectionUniform_,
          randomKey};
    }

  } // namespace proposal_fallback_detail

  inline bool is_known_proposal_interaction_type(std::int32_t processId) noexcept {
    using PROPOSAL::InteractionType;
    switch (static_cast<InteractionType>(processId)) {
      case InteractionType::Particle:
      case InteractionType::Brems:
      case InteractionType::Ioniz:
      case InteractionType::Epair:
      case InteractionType::Photonuclear:
      case InteractionType::MuPair:
      case InteractionType::Hadrons:
      case InteractionType::ContinuousEnergyLoss:
      case InteractionType::WeakInt:
      case InteractionType::Compton:
      case InteractionType::Decay:
      case InteractionType::Annihilation:
      case InteractionType::Photopair:
      case InteractionType::Photoproduction:
      case InteractionType::PhotoMuPair:
      case InteractionType::Photoeffect:
        return true;
      case InteractionType::Undefined:
        return false;
    }
    return false;
  }

  inline bool proposal_fallback_requires_selected_loss(
      FallbackRequest const& event) noexcept {
    return proposal_fallback_requires_selected_loss(event.reason_);
  }

  inline bool has_specified_proposal_interaction_identity(
      FallbackRequest const& event) noexcept {
    if (event.interactionVertexReached_ != 1 || event.mediumHash_ == 0 ||
        event.interactionHash_ == 0 ||
        !is_known_proposal_interaction_type(event.processId_) ||
        !std::isfinite(event.particle_.energyGeV_) ||
        !(event.particle_.energyGeV_ > 0.) || !std::isfinite(event.particle_.timeS_) ||
        !std::isfinite(event.selectionUniform_) || event.selectionUniform_ < 0. ||
        event.selectionUniform_ >= 1.) {
      return false;
    }
    double directionNormSquared = 0.;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(event.particle_.positionM_[axis]) ||
          !std::isfinite(event.particle_.direction_[axis])) {
        return false;
      }
      directionNormSquared +=
          event.particle_.direction_[axis] * event.particle_.direction_[axis];
    }
    return std::isfinite(directionNormSquared) && directionNormSquared > 0.;
  }

  inline bool has_resolvable_proposal_selected_loss(
      FallbackRequest const& event) noexcept {
    if (!has_specified_proposal_interaction_identity(event) ||
        !proposal_fallback_requires_selected_loss(event)) {
      return false;
    }
    // NativeSelectionReplay invokes live PROPOSAL SampleLoss with the
    // retained selection_uniform.  Its residual conditional quantile is
    // diagnostic only and can legitimately round to exactly one at a
    // cumulative-rate boundary.  Other selected-loss completions do consume
    // loss_quantile and retain their strict half-open [0,1) contract.
    if (event.reason_ == FallbackReason::NativeSelectionReplay) { return true; }
    return std::isfinite(event.lossQuantile_) && event.lossQuantile_ >= 0. &&
           event.lossQuantile_ < 1.;
  }

  inline bool has_specified_proposal_final_state(FallbackRequest const& event) noexcept {
    if (!has_specified_proposal_interaction_identity(event) ||
        proposal_fallback_requires_selected_loss(event) ||
        !std::isfinite(event.energyFraction_) || event.energyFraction_ < 0. ||
        event.energyFraction_ > 1.) {
      return false;
    }
    return true;
  }

  inline bool permits_specified_proposal_cpu_final_state(
      FallbackRequest const& event) noexcept {
    if (!has_specified_proposal_final_state(event)) { return false; }
    if (event.reason_ == FallbackReason::CpuOnlyProcess) {
      return process_capability(event.particle_.pid_, event.processId_) ==
             ProcessCapability::CpuOnlyFinalState;
    }
    if (event.reason_ == FallbackReason::EpairRejectionEnvelopeExceeded ||
        event.reason_ == FallbackReason::EpairRejectionTrialsExhausted) {
      return event.processId_ == ElectronPairProcessId &&
             (is_electron_or_positron_pid(event.particle_.pid_) ||
              is_muon_pid(event.particle_.pid_));
    }
    return false;
  }

  inline proposal::ProposalInteractionRecord make_proposal_interaction_record(
      FallbackRequest const& event, std::uint64_t randomSeed, std::uint64_t showerId) {
    if (!has_specified_proposal_final_state(event)) {
      throw std::invalid_argument(
          "GPU fallback does not contain a fully specified PROPOSAL final state");
    }
    return proposal_fallback_detail::make_record_from_identity(event, randomSeed,
                                                               showerId);
  }

  inline proposal::ProposalInteractionRecord
  make_proposal_interaction_record_for_selected_loss(FallbackRequest const& event,
                                                     std::uint64_t randomSeed,
                                                     std::uint64_t showerId) {
    if (!has_resolvable_proposal_selected_loss(event)) {
      throw std::invalid_argument(
          "GPU fallback does not contain a resolvable selected PROPOSAL loss");
    }
    return proposal_fallback_detail::make_record_from_identity(event, randomSeed,
                                                               showerId);
  }

  inline std::vector<double> make_proposal_final_state_random_numbers(
      FallbackRequest const& event, std::size_t count, std::uint64_t randomSeed,
      std::uint64_t showerId) {
    if (!has_specified_proposal_final_state(event) &&
        !has_resolvable_proposal_selected_loss(event)) {
      throw std::invalid_argument(
          "cannot draw a PROPOSAL final state for an unresolved GPU fallback");
    }
    if (count >
        std::numeric_limits<std::uint64_t>::max() - CpuProposalFinalStateDrawIdBase) {
      throw std::overflow_error("PROPOSAL final-state random-number count overflow");
    }
    std::vector<double> randomNumbers(count);
    for (std::size_t index = 0; index < count; ++index) {
      RandomNumberKey const key{
          randomSeed,
          showerId,
          event.particle_.historyId_,
          event.particle_.stepId_,
          static_cast<std::uint32_t>(event.processId_),
          CpuProposalFinalStateDrawIdBase + static_cast<std::uint64_t>(index)};
      randomNumbers[index] = uniform_open01(key);
    }
    return randomNumbers;
  }

  template <typename TProposalInteractionModel>
  auto generate_specified_proposal_final_state(TProposalInteractionModel& model,
                                               FallbackRequest const& event,
                                               std::uint64_t randomSeed,
                                               std::uint64_t showerId) {
    auto const record = make_proposal_interaction_record(event, randomSeed, showerId);
    auto const count = model.requiredFinalStateRandomNumbers(record);
    auto randomNumbers =
        make_proposal_final_state_random_numbers(event, count, randomSeed, showerId);
    return model.generateFinalState(record, std::move(randomNumbers));
  }
} // namespace corsika::kokkos
