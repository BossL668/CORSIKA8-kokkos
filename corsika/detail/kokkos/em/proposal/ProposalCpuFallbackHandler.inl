/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  template <typename TStack, typename TProposalInteractionModel, typename TProcessList,
            typename TEnvironment>
  ProposalCpuFallbackHandler<
      TStack, TProposalInteractionModel, TProcessList,
      TEnvironment>::ProposalCpuFallbackHandler(TProposalInteractionModel& proposalModel,
                                                TProcessList& processList,
                                                TEnvironment const& environment,
                                                topology_type const& topology,
                                                CoordinateSystemPtr coordinateSystem,
                                                std::uint64_t randomSeed,
                                                std::uint64_t showerId,
                                                HEPEnergyType specifiedEnergyCut)
      : proposalModel_(proposalModel)
      , processList_(processList)
      , environment_(environment)
      , topology_(topology)
      , coordinateSystem_(std::move(coordinateSystem))
      , randomSeed_(randomSeed)
      , showerId_(showerId)
      , specifiedEnergyCut_(specifiedEnergyCut) {
    if (!coordinateSystem_) {
      throw std::invalid_argument(
          "PROPOSAL CPU fallback handler requires a coordinate system");
    }
  }

  template <typename TStack, typename TProposalInteractionModel, typename TProcessList,
            typename TEnvironment>
  bool ProposalCpuFallbackHandler<TStack, TProposalInteractionModel, TProcessList,
                                  TEnvironment>::canHandle(FallbackRequest const& event)
      const noexcept {
    return permits_specified_proposal_cpu_final_state(event) ||
           has_resolvable_proposal_selected_loss(event);
  }

  template <typename TStack, typename TProposalInteractionModel, typename TProcessList,
            typename TEnvironment>
  void ProposalCpuFallbackHandler<TStack, TProposalInteractionModel, TProcessList,
                                  TEnvironment>::handle(TStack& stack,
                                                        FallbackRequest const& event) {
    if (!canHandle(event)) {
      throw std::invalid_argument(
          "PROPOSAL CPU fallback handler received an incomplete event");
    }
    auto parent = particle_transfer::import_particle(stack, event.particle_,
                                                     coordinateSystem_, topology_);
    auto const* node = parent.getNode();
    if (node == nullptr || (node == environment_.getUniverse().get() &&
                            !environment_.getUniverse()->hasModelProperties())) {
      parent.erase();
      throw std::runtime_error(
          "specified PROPOSAL fallback vertex is outside the physics environment");
    }
    parent.setNode(node);

    stack_view_type secondaries(parent);
    auto const completeSelectedLoss = has_resolvable_proposal_selected_loss(event);
    auto record = completeSelectedLoss
                      ? make_proposal_interaction_record_for_selected_loss(
                            event, randomSeed_, showerId_)
                      : make_proposal_interaction_record(event, randomSeed_, showerId_);
    // The table metadata stores PROPOSAL::Medium::GetHash(), while
    // InteractionModel indexes calculators by the CORSIKA
    // NuclearComposition hash. They describe the same material but are
    // intentionally different hash domains. Resolve the CPU key from the
    // actual environment node at the already selected GPU vertex; the
    // interaction and target-component hashes remain those selected from
    // the versioned PROPOSAL table and are validated by the CPU model.
    record.context_.mediumHash_ =
        parent.getNode()->getModelProperties().getNuclearComposition().getHash();
    if (specifiedEnergyCut_ > HEPEnergyType::zero()) {
      proposalModel_.prepareSpecifiedInteractionCalculator(record, specifiedEnergyCut_);
    }
    auto const replayNativeSelection =
        event.reason_ == FallbackReason::NativeSelectionReplay;
    if (replayNativeSelection) {
      proposalModel_.completeNativeSelectionReplay(record);
    } else if (completeSelectedLoss) {
      proposalModel_.completeSelectedLoss(record, event.lossQuantile_);
    }
    auto const randomCount = proposalModel_.requiredFinalStateRandomNumbers(record);
    auto randomNumbers = make_proposal_final_state_random_numbers(event, randomCount,
                                                                  randomSeed_, showerId_);
    auto const returnCode = proposalModel_.doSpecifiedInteraction(
        secondaries, record, std::move(randomNumbers));
    if (returnCode != ProcessReturn::Ok && returnCode != ProcessReturn::Interacted) {
      parent.erase();
      throw std::runtime_error("specified CPU PROPOSAL fallback did not complete");
    }

    if constexpr (stack_view_type::has_event) {
      for (auto&& secondary : secondaries) {
        secondary.getEvent()->setEventType(history::EventType::Interaction);
      }
    }
    auto const generated = static_cast<std::uint64_t>(secondaries.getSize());
    processList_.doSecondaries(secondaries);
    parent.erase();
    ++statistics_.specifiedInteractions_;
    if (completeSelectedLoss) { ++statistics_.completedSelectedLosses_; }
    if (replayNativeSelection) { ++statistics_.completedNativeSelectionReplays_; }
    statistics_.generatedSecondaries_ += generated;
  }

  template <typename TStack, typename TProposalInteractionModel, typename TProcessList,
            typename TEnvironment>
  ProposalCpuFallbackStatistics const& ProposalCpuFallbackHandler<
      TStack, TProposalInteractionModel, TProcessList, TEnvironment>::statistics() const
      noexcept {
    return statistics_;
  }
} // namespace corsika::kokkos
