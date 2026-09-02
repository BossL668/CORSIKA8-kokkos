/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <corsika/accelerator/em/PhotonFinalStateRandomDomains.hpp>
#include <corsika/gpu/em/PhotonPairFinalState.hpp>
#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

  // Preserve the historical namespace for source compatibility.  The actual
  // counter domains now live in the execution-model-neutral accelerator
  // layer and are shared by native CUDA and Kokkos.
  using accelerator::em::ComptonAzimuthDrawId;
  using accelerator::em::PhotonPairAnalyticAcceptanceDrawIdBase;
  using accelerator::em::PhotonPairAnalyticCandidateDrawIdBase;
  using accelerator::em::PhotonPairAnalyticMaximumAttempts;
  using accelerator::em::PhotonPairAzimuthDrawId;
  using accelerator::em::PhotonPairElectronPolarDrawId;
  using accelerator::em::PhotonPairLpmDrawId;
  using accelerator::em::PhotonPairPositronPolarDrawId;
  using accelerator::em::PhotonPairSplitDrawId;
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
