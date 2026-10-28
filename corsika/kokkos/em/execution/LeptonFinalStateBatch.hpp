/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <corsika/kokkos/em/execution/BatchResults.hpp>
#include <corsika/kokkos/em/algorithms/LeptonFinalStateStep.hpp>

namespace corsika::kokkos {

  struct LeptonFinalStateExecutionResult {
    LeptonFinalStateBatchResult batch_{};
    std::optional<FirstInteractionRecord> firstInteraction_{};
  };

  /**
   * Kokkos implementation of bremsstrahlung, annihilation, ionization and
   * electron-pair final states.  Every output offset is assigned by a stable
   * exclusive scan; no completion-order atomic append participates in the
   * shower history tree.
   */

  template <class TExecutionSpace>
  LeptonFinalStateExecutionResult generate_lepton_final_states_batch(
      BremsLpmSnapshot const lpmSnapshot, EmThinningConfig const thinning,
      std::vector<InteractionRecord> const& input, std::uint64_t const randomSeed,
      std::uint64_t const showerId, std::uint64_t const firstSecondaryHistoryId,
      TExecutionSpace const& execution = {});
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/LeptonFinalStateBatch.inl>
