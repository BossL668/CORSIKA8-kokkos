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
#include <corsika/kokkos/em/algorithms/LeptonTransportStep.hpp>
#include <corsika/kokkos/em/algorithms/MoliereStep.hpp>
#include <corsika/kokkos/environment/EnvironmentBorrowedView.hpp>
#include <corsika/kokkos/em/algorithms/InteractionPhysicsView.hpp>

namespace corsika::kokkos {

  /**
   * Complete charged-lepton propagation up to, but not including, the split
   * Moliere stage.  The physics function is shared verbatim with native CUDA;
   * Kokkos owns only queue storage and stable compaction.
   */

  template <class TExecutionSpace>
  LeptonTransportBatchResult transport_leptons_batch(
      InteractionPhysicsView const physics, EnvironmentBorrowedView const environment,
      TransportGeometryConfig const geometry, MoliereSnapshot const electronMoliere,
      MoliereSnapshot const muonMoliere,
      MoliereInterpolationView const moliereInterpolation,
      bool const muonMoliereAvailable, std::uint64_t const randomSeed,
      std::uint64_t const showerId, std::vector<InteractionRecord> const& input,
      TExecutionSpace const& execution = {});
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/LeptonTransportBatch.inl>
