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
#include <corsika/kokkos/em/algorithms/PhotonFinalStateStep.hpp>
#include <corsika/kokkos/em/algorithms/InteractionPhysicsView.hpp>

namespace corsika::kokkos {

  struct PhotonFinalStateExecutionResult {
    PhotonFinalStateBatchResult batch_{};
    std::optional<FirstInteractionRecord> firstInteraction_{};
  };

  /**
   * Kokkos implementation of photon pair, Compton and photoelectric final
   * states.  Child identities are allocated by exclusive scans, so changing
   * team size or execution space cannot change the shower history tree.
   */

  template <class TExecutionSpace>
  PhotonFinalStateExecutionResult generate_photon_final_states_batch(
      InteractionPhysicsView const physics, PhotonPairLpmSnapshot const lpmSnapshot,
      EmThinningConfig const thinning, std::vector<InteractionRecord> const& input,
      std::uint64_t const randomSeed, std::uint64_t const showerId,
      std::uint64_t const firstSecondaryHistoryId, TExecutionSpace const& execution = {});
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/PhotonFinalStateBatch.inl>
