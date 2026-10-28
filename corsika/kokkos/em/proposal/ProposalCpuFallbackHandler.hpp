/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/process/ProcessReturn.hpp>
#include <corsika/kokkos/em/proposal/ProposalFallbackAdapter.hpp>
#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <corsika/stack/history/HistoryStackExtension.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace corsika::kokkos {

  struct ProposalCpuFallbackStatistics {
    std::uint64_t specifiedInteractions_{};
    std::uint64_t completedSelectedLosses_{};
    std::uint64_t completedNativeSelectionReplays_{};
    std::uint64_t generatedSecondaries_{};
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
  template <typename TStack, typename TProposalInteractionModel, typename TProcessList,
            typename TEnvironment>
  class ProposalCpuFallbackHandler {
  public:
    using topology_type =
        EnvironmentTopology<typename TEnvironment::BaseNodeType::IModelProperties>;
    using stack_view_type = typename TStack::stack_view_type;

    ProposalCpuFallbackHandler(TProposalInteractionModel& proposalModel,
                               TProcessList& processList, TEnvironment const& environment,
                               topology_type const& topology,
                               CoordinateSystemPtr coordinateSystem,
                               std::uint64_t randomSeed, std::uint64_t showerId,
                               HEPEnergyType specifiedEnergyCut = HEPEnergyType::zero());

    bool canHandle(FallbackRequest const& event) const noexcept;

    void handle(TStack& stack, FallbackRequest const& event);

    ProposalCpuFallbackStatistics const& statistics() const noexcept;

  private:
    TProposalInteractionModel& proposalModel_;
    TProcessList& processList_;
    TEnvironment const& environment_;
    topology_type const& topology_;
    CoordinateSystemPtr coordinateSystem_;
    std::uint64_t randomSeed_{};
    std::uint64_t showerId_{};
    HEPEnergyType specifiedEnergyCut_{HEPEnergyType::zero()};
    ProposalCpuFallbackStatistics statistics_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/proposal/ProposalCpuFallbackHandler.inl>
