/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

  /**
   * Complete charged-particle process/component/v selection at the transported
   * vertex energy.
   *
   * The process threshold remains uniform on [0, pre-step total rate), exactly
   * matching ScalarCascadeStepper. If continuous loss reduced the vertex total
   * below that threshold, the candidate becomes an explicit no-interaction
   * continuation.
   */
  LeptonVertexSelectionBatchResult
  reselectLeptonInteractionsAtVertexForValidation(
      tables::FlatRateTableView device_table,
      std::vector<EmInteractionRecord> const& candidates,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
