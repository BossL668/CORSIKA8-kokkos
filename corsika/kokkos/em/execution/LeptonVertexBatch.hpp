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
#include <corsika/kokkos/em/algorithms/LeptonVertexSelection.hpp>

namespace corsika::kokkos {

  struct LeptonVertexBatchResult {
    LeptonVertexSelectionBatchResult batch_{};
    std::uint64_t nativeNewtonIterations_{};
    std::uint64_t nativeBisectionIterations_{};
    std::uint64_t nativeInverseFailures_{};
  };

  template <class TExecutionSpace>
  LeptonVertexBatchResult select_lepton_vertices_batch(
      InteractionPhysicsView const physics, std::vector<InteractionRecord> const& input,
      std::uint64_t const randomSeed, std::uint64_t const showerId,
      TExecutionSpace const& execution = {});
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/LeptonVertexBatch.inl>
