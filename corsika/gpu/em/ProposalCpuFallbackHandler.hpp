/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/gpu/em/ProposalFallbackAdapter.hpp>
#include <corsika/gpu/em/RouterParticleConversion.hpp>
#include <corsika/stack/history/HistoryStackExtension.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace corsika::gpu::em {

  struct ProposalCpuFallbackStatistics {
    std::uint64_t specified_interactions{};
    std::uint64_t completed_selected_losses{};
    std::uint64_t completed_native_selection_replays{};
    std::uint64_t generated_secondaries{};
  };

  /**
   * Execute a GPU-selected interaction with CPU PROPOSAL without resampling
   * type, target component or fractional loss.
   *
   * The handler imports the interaction-vertex parent only as the projectile
   * backing a SecondaryView, assigns its CORSIKA environment node, calls
   * InteractionModel::doSpecifiedInteraction(), runs the ordinary
   * doSecondaries chain (cut/thinning/output), then erases the consumed parent.
   */
  template <
      typename TStack, typename TProposalInteractionModel,
      typename TProcessList, typename TEnvironment>
  class ProposalCpuFallbackHandler {
  public:
    using stack_view_type =
        typename TStack::stack_view_type;

    ProposalCpuFallbackHandler(
        TProposalInteractionModel& proposal_model,
        TProcessList& process_list,
        TEnvironment const& environment,
        CoordinateSystemPtr coordinate_system,
        std::uint64_t random_seed,
        std::uint64_t shower_id,
        HEPEnergyType specified_energy_cut =
            HEPEnergyType::zero())
        : proposal_model_(proposal_model)
        , process_list_(process_list)
        , environment_(environment)
        , coordinate_system_(std::move(coordinate_system))
        , random_seed_(random_seed)
        , shower_id_(shower_id)
        , specified_energy_cut_(
              specified_energy_cut) {
      if (!coordinate_system_) {
        throw std::invalid_argument(
            "PROPOSAL CPU fallback handler requires a coordinate system");
      }
    }

    bool canHandle(
        ProposalFallbackEvent const& event) const noexcept {
      return permitsSpecifiedProposalCpuFinalState(event) ||
             hasResolvableProposalSelectedLoss(event);
    }

    void handle(
        TStack& stack,
        ProposalFallbackEvent const& event) {
      if (!canHandle(event)) {
        throw std::invalid_argument(
            "PROPOSAL CPU fallback handler received an incomplete event");
      }
      auto parent = router_detail::importParticle(
          stack, event.particle, coordinate_system_);
      auto const* node =
          environment_.getUniverse()->getContainingNode(
              parent.getPosition());
      if (node == nullptr ||
          (node == environment_.getUniverse().get() &&
           !environment_.getUniverse()
                ->hasModelProperties())) {
        parent.erase();
        throw std::runtime_error(
            "specified PROPOSAL fallback vertex is outside the physics environment");
      }
      parent.setNode(node);

      stack_view_type secondaries(parent);
      auto const complete_selected_loss =
          hasResolvableProposalSelectedLoss(event);
      auto record = complete_selected_loss
                        ? makeProposalInteractionRecordForSelectedLoss(
                              event, random_seed_, shower_id_)
                        : makeProposalInteractionRecord(
                              event, random_seed_, shower_id_);
      // The table metadata stores PROPOSAL::Medium::GetHash(), while
      // InteractionModel indexes calculators by the CORSIKA
      // NuclearComposition hash. They describe the same material but are
      // intentionally different hash domains. Resolve the CPU key from the
      // actual environment node at the already selected GPU vertex; the
      // interaction and target-component hashes remain those selected from
      // the versioned PROPOSAL table and are validated by the CPU model.
      record.context.medium_hash =
          parent.getNode()
              ->getModelProperties()
              .getNuclearComposition()
              .getHash();
      if (specified_energy_cut_ >
          HEPEnergyType::zero()) {
        proposal_model_
            .prepareSpecifiedInteractionCalculator(
                record, specified_energy_cut_);
      }
      auto const replay_native_selection =
          event.reason ==
          ProposalFallbackReason::NativeSelectionReplay;
      if (replay_native_selection) {
        proposal_model_.completeNativeSelectionReplay(record);
      } else if (complete_selected_loss) {
        proposal_model_.completeSelectedLoss(
            record, event.loss_quantile);
      }
      auto const random_count =
          proposal_model_.requiredFinalStateRandomNumbers(
              record);
      auto random_numbers =
          makeProposalFinalStateRandomNumbers(
              event, random_count, random_seed_,
              shower_id_);
      auto const return_code =
          proposal_model_.doSpecifiedInteraction(
              secondaries, record,
              std::move(random_numbers));
      if (return_code != ProcessReturn::Ok &&
          return_code != ProcessReturn::Interacted) {
        parent.erase();
        throw std::runtime_error(
            "specified CPU PROPOSAL fallback did not complete");
      }

      if constexpr (stack_view_type::has_event) {
        for (auto&& secondary : secondaries) {
          secondary.getEvent()->setEventType(
              history::EventType::Interaction);
        }
      }
      auto const generated =
          static_cast<std::uint64_t>(
              secondaries.getSize());
      process_list_.doSecondaries(secondaries);
      parent.erase();
      ++statistics_.specified_interactions;
      if (complete_selected_loss) {
        ++statistics_.completed_selected_losses;
      }
      if (replay_native_selection) {
        ++statistics_.completed_native_selection_replays;
      }
      statistics_.generated_secondaries += generated;
    }

    ProposalCpuFallbackStatistics const&
    statistics() const noexcept {
      return statistics_;
    }

  private:
    TProposalInteractionModel& proposal_model_;
    TProcessList& process_list_;
    TEnvironment const& environment_;
    CoordinateSystemPtr coordinate_system_;
    std::uint64_t random_seed_{};
    std::uint64_t shower_id_{};
    HEPEnergyType specified_energy_cut_{
        HEPEnergyType::zero()};
    ProposalCpuFallbackStatistics statistics_{};
  };

} // namespace corsika::gpu::em
