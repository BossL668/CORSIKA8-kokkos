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
#include <corsika/kokkos/em/algorithms/InteractionSelection.hpp>

namespace corsika::kokkos {

  struct InteractionSelectionBatchResult {
    InteractionBatchResult batch_{};
    std::uint64_t nativeNewtonIterations_{};
    std::uint64_t nativeBisectionIterations_{};
    std::uint64_t nativeInverseFailures_{};
  };

  /**
   * Selection plus stable flag/scan/scatter compaction shared by every Kokkos
   * execution space.  Each particle uses the same history-keyed Philox draws
   * and the same device function as native CUDA; only the launch mechanism
   * and memory space differ.
   */

  template <class TExecutionSpace>
  InteractionSelectionBatchResult select_interactions_batch(
      InteractionPhysicsView const physics, std::vector<ParticleState> const& particles,
      std::uint64_t const randomSeed, std::uint64_t const showerId,
      TExecutionSpace const& execution = {});
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/InteractionSelectionBatch.inl>
