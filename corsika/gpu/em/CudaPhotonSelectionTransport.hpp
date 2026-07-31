/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <vector>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

  struct PhotonSelectionTransportBatchResult {
    std::size_t input_particles{};
    std::vector<PhotonTransportRecord> records{};
    std::vector<ProposalFallbackEvent>
        selection_fallback_events{};
    std::vector<ProposalFallbackEvent>
        transport_fallback_events{};
  };

  struct PhotonDevicePipelineBatchResult {
    std::size_t input_particles{};
    std::vector<PhotonTransportRecord> transport_records{};
    std::vector<ProposalFallbackEvent>
        selection_fallback_events{};
    std::vector<ProposalFallbackEvent>
        transport_fallback_events{};
    std::vector<EmParticleState> next_photons{};
    std::vector<ObservationRecord> observations{};
    EmFinalStateBatchResult final_states{};
  };

  /**
   * Device-chained selector and spherical transport.
   *
   * Photon states are uploaded once. Compact selected interactions remain in
   * the reusable device workspace and are consumed directly by transport;
   * only final transport records and explicit fallbacks are downloaded.
   */
  PhotonSelectionTransportBatchResult
  selectAndTransportPhotonsForValidation(
      tables::FlatRateTableView device_table,
      EnvironmentSnapshot const& environment,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      detail::DeviceWorkspace& workspace);

  /**
   * Device-chain selection, spherical transport and photon-pair/LPM final
   * state. Neither compact selected interactions nor transported interaction
   * vertices are round-tripped through the host.
   */
  PhotonDevicePipelineBatchResult
  runPhotonDevicePipelineForValidation(
      tables::FlatRateTableView device_table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      EnvironmentSnapshot const& environment,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id, int device,
      std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
