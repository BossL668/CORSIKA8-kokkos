/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <corsika/gpu/em/PhotonPairFinalState.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

  inline constexpr std::uint64_t PhotonPairSplitDrawId = 1;
  inline constexpr std::uint64_t PhotonPairAzimuthDrawId = 2;
  inline constexpr std::uint64_t PhotonPairElectronPolarDrawId = 3;
  inline constexpr std::uint64_t PhotonPairPositronPolarDrawId = 4;
  inline constexpr std::uint64_t PhotonPairLpmDrawId = 5;
  inline constexpr std::uint64_t
      PhotonPairAnalyticCandidateDrawIdBase = 0x20;
  inline constexpr std::uint64_t
      PhotonPairAnalyticAcceptanceDrawIdBase = 0x60;
  inline constexpr std::uint32_t
      PhotonPairAnalyticMaximumAttempts = 32;
  inline constexpr std::uint64_t ComptonAzimuthDrawId = 1;
  /**
   * Validation bridge for the first physical GPU final-state kernel.
   *
   * It dispatches every selected interaction through the capability map,
   * generates photon-pair and Compton final states, preserves no-interaction
   * records and emits stable CPU fallback records for every process not
   * implemented on the GPU.
   */
  EmFinalStateBatchResult generatePhotonPairFinalStatesForValidation(
      tables::FlatRateTableView device_table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      std::vector<EmInteractionRecord> const& interactions,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
