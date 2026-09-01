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

  inline constexpr std::uint32_t
      InteractionDistanceRandomProcessId = 0x454d0001U;
  inline constexpr std::uint32_t
      InteractionColumnRandomProcessId = 0x454d0002U;
  inline constexpr std::uint32_t
      ProposalSelectionRandomProcessId = 0x454d0003U;
  inline constexpr std::uint64_t InteractionDistanceDrawId = 0;
  inline constexpr std::uint64_t InteractionColumnDrawId = 0;
  inline constexpr std::uint64_t ProposalSelectionDrawId = 0;
  inline constexpr std::uint64_t InteractionLossDrawId = 0;

  /**
   * Validation bridge for the first physical selection kernel.
   *
   * It uploads an AoS input batch, samples one discrete interaction per
   * particle and returns stable compacted success/fallback arrays. Production
   * transport will invoke the same device logic on the resident particle SoA.
   */
  EmInteractionBatchResult selectDiscreteInteractionsForValidation(
      tables::FlatRateTableView device_table,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
