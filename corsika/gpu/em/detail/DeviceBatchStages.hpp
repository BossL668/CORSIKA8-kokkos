/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/PhotonPairLpm.hpp>
#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em::detail {

  struct PhotonFinalStateSummaryLayout {
    static constexpr std::size_t SecondaryCount = 0;
    static constexpr std::size_t GpuCount = 1;
    static constexpr std::size_t FallbackCount = 2;
    static constexpr std::size_t ContinuationCount = 3;
    static constexpr std::size_t SuppressionCount = 4;
    static constexpr std::size_t PhotonPairCount = 5;
    static constexpr std::size_t ComptonCount = 6;
    static constexpr std::size_t PhotoelectricCount = 7;
    static constexpr std::size_t InputCount = 8;
    static constexpr std::size_t Error = 9;
    static constexpr std::size_t Size = 10;
  };

  struct BremsFinalStateSummaryLayout {
    static constexpr std::size_t SecondaryCount = 0;
    static constexpr std::size_t GpuCount = 1;
    static constexpr std::size_t FallbackCount = 2;
    static constexpr std::size_t ContinuationCount = 3;
    static constexpr std::size_t SuppressionCount = 4;
    static constexpr std::size_t BremsCount = 5;
    static constexpr std::size_t AnnihilationCount = 6;
    static constexpr std::size_t IonizationCount = 7;
    static constexpr std::size_t ElectronPairCount = 8;
    static constexpr std::size_t BremsSuppressionCount = 9;
    static constexpr std::size_t ElectronPairSuppressionCount = 10;
    static constexpr std::size_t ElectronPairRejectionTrials = 11;
    static constexpr std::size_t ElectronPairZeroWeightSamples = 12;
    static constexpr std::size_t ElectronPairRejectionFallbacks = 13;
    static constexpr std::size_t ElectronPairEnvelopeViolations = 14;
    static constexpr std::size_t Error = 15;
    static constexpr std::size_t Size = 16;
  };

  struct LeptonVertexSummaryLayout {
    static constexpr std::size_t InputCount = 0;
    static constexpr std::size_t InteractionCount = 1;
    static constexpr std::size_t ContinuationCount = 2;
    static constexpr std::size_t FallbackCount = 3;
    static constexpr std::size_t Error = 4;
    static constexpr std::size_t Size = 5;
  };

  struct DeviceInteractionSelectionBatch {
    std::size_t input_count{};
    std::size_t interaction_count{};
    std::size_t fallback_count{};
    EmInteractionRecord* interactions{};
    ProposalFallbackEvent* fallbacks{};
    EmInteractionRecord* raw_interactions{};
    std::uint32_t* device_fallback_count{};
    bool counts_deferred{};
  };

  struct DevicePhotonTransportBatch {
    std::size_t input_count{};
    std::size_t record_count{};
    std::size_t fallback_count{};
    PhotonTransportRecord* records{};
    ProposalFallbackEvent* fallbacks{};
  };

  struct DeviceLeptonTransportBatch {
    std::size_t input_count{};
    std::size_t record_count{};
    std::size_t fallback_count{};
    LeptonTransportRecord* records{};
    ProposalFallbackEvent* fallbacks{};
  };

  struct DeviceTransportInteractionBatch {
    std::size_t input_count{};
    std::size_t interaction_count{};
    EmInteractionRecord* interactions{};
    std::uint32_t* device_interaction_count{};
    bool count_deferred{};
  };

  struct DeviceLeptonVertexSelectionBatch {
    std::size_t input_count{};
    std::size_t interaction_count{};
    std::size_t continuation_count{};
    std::size_t fallback_count{};
    EmInteractionRecord* interactions{};
    EmInteractionRecord* continuations{};
    ProposalFallbackEvent* fallbacks{};
    std::uint32_t* device_summary{};
    bool counts_deferred{};
  };

  struct DevicePhotonPairFinalStateBatch {
    std::size_t input_count{};
    std::size_t gpu_interaction_count{};
    std::size_t photon_pair_interaction_count{};
    std::size_t compton_interaction_count{};
    std::size_t photoelectric_interaction_count{};
    std::size_t secondary_count{};
    std::size_t fallback_count{};
    std::size_t continuation_count{};
    std::size_t suppression_count{};
    PhotonPairFinalStateRecord* records{};
    EmParticleState* secondaries{};
    ProposalFallbackEvent* fallbacks{};
    EmInteractionRecord* continuations{};
    PhotonPairLpmSuppressionRecord* suppressions{};
    std::uint32_t* error_flag{};
    std::uint32_t* device_summary{};
    bool counts_deferred{};
  };

  struct DeviceBremsFinalStateBatch {
    std::size_t input_count{};
    std::size_t gpu_interaction_count{};
    std::size_t brems_interaction_count{};
    std::size_t annihilation_interaction_count{};
    std::size_t ionization_interaction_count{};
    std::size_t electron_pair_interaction_count{};
    std::size_t brems_lpm_trial_count{};
    std::size_t brems_lpm_suppression_count{};
    std::size_t electron_pair_lpm_trial_count{};
    std::size_t electron_pair_lpm_suppression_count{};
    std::size_t electron_pair_rejection_trials{};
    std::size_t electron_pair_zero_weight_samples{};
    std::size_t electron_pair_rejection_fallbacks{};
    std::size_t electron_pair_envelope_violations{};
    std::size_t secondary_count{};
    std::size_t fallback_count{};
    std::size_t continuation_count{};
    std::size_t suppression_count{};
    BremsFinalStateRecord* records{};
    EmParticleState* secondaries{};
    ProposalFallbackEvent* fallbacks{};
    EmInteractionRecord* continuations{};
    BremsLpmSuppressionRecord* suppressions{};
    std::uint32_t* error_flag{};
    std::uint32_t* device_summary{};
    bool counts_deferred{};
  };

  struct DevicePhotonEndpointBatch {
    std::size_t source_count{};
    std::size_t next_photon_count{};
    std::size_t observation_count{};
    std::size_t particle_cut_count{};
    std::size_t generated_lepton_count{};
    bool generated_leptons_compacted{};
    EmParticleState* next_photons{};
    ObservationRecord* observations{};
  };

  /**
   * Optional persistent destination for charged photon final-state children.
   *
   * The storage is owned by CudaEmBackend. Photon endpoint compaction writes a
   * stable DeviceSelect result into this sink before downloading its existing
   * endpoint summary, so the host does not need a second synchronization just
   * to learn the selected count.
   */
  struct DeviceChargedSecondarySink {
    EmParticleState* output{};
    std::size_t capacity{};
    void* temporary_storage{};
    std::size_t temporary_storage_bytes{};
    std::size_t* selected_count{};
  };

  struct DeviceLeptonEndpointBatch {
    std::size_t source_count{};
    std::size_t next_lepton_count{};
    std::size_t generated_photon_count{};
    std::size_t observation_count{};
    std::size_t decay_candidate_count{};
    EmParticleState* next_leptons{};
    EmParticleState* generated_photons{};
    ObservationRecord* observations{};
    EmParticleState* decay_candidates{};
  };

  struct DevicePhotonPipelineBatch {
    DeviceInteractionSelectionBatch selection{};
    DevicePhotonTransportBatch transport{};
    DeviceTransportInteractionBatch at_interaction{};
    DevicePhotonPairFinalStateBatch final_state{};
    DevicePhotonEndpointBatch endpoints{};
  };

  struct DeviceLeptonPipelineBatch {
    DeviceInteractionSelectionBatch selection{};
    DeviceLeptonTransportBatch transport{};
    DeviceTransportInteractionBatch at_interaction{};
    DeviceLeptonVertexSelectionBatch vertex{};
    DeviceBremsFinalStateBatch final_state{};
    DeviceLeptonEndpointBatch endpoints{};
  };

  /**
   * Optional CUDA events recorded after each fused lepton pipeline stage.
   *
   * The caller owns the events. A null pointer disables all stage records and
   * therefore adds no event-launch overhead to production transport.
   */
  struct LeptonPipelineStageEvents {
    cudaEvent_t selection_done{};
    cudaEvent_t transport_physics_done{};
    cudaEvent_t moliere_done{};
    cudaEvent_t transport_control_done{};
    cudaEvent_t transport_done{};
    cudaEvent_t interaction_extraction_done{};
    cudaEvent_t vertex_selection_done{};
    cudaEvent_t final_state_classification_done{};
    cudaEvent_t final_state_scan_done{};
    cudaEvent_t final_state_summary_done{};
    cudaEvent_t final_state_write_done{};
    cudaEvent_t final_state_done{};
    cudaEvent_t endpoint_compaction_done{};
  };

  /**
   * Append exact stage storage to a shared arena sizing pass.
   *
   * launch*OnDevice must be called in the same order after prepare(), and
   * never resets or grows the arena. Its compact output remains valid while
   * later stages acquire additional slices.
   */
  void appendInteractionSelectionWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceInteractionSelectionBatch
  launchInteractionSelectionOnDevice(
      tables::FlatRateTableView device_table,
      EmParticleState const* device_particles, std::size_t count,
      std::uint64_t random_seed, std::uint64_t shower_id,
      DeviceWorkspace&, bool defer_count_download = false);

  void appendPhotonTransportWorkspace(
      WorkspaceSize&, std::size_t count);

  DevicePhotonTransportBatch launchPhotonTransportOnDevice(
      EnvironmentSnapshot const& environment,
      EmInteractionRecord const* device_interactions,
      std::size_t count, DeviceWorkspace&,
      DeviceInteractionSelectionBatch* deferred_selection =
          nullptr);

  void appendLeptonTransportWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceLeptonTransportBatch launchLeptonTransportOnDevice(
      tables::FlatRateTableView device_table,
      MoliereSnapshot const& moliere_snapshot,
      MoliereSnapshot const& muon_moliere_snapshot,
      MoliereInterpolationView const& moliere_interpolation,
      bool apply_moliere, bool muon_moliere_available,
      EnvironmentSnapshot const& environment,
      EmInteractionRecord const* device_interactions,
      std::size_t count, std::uint64_t random_seed,
      std::uint64_t shower_id, DeviceWorkspace&,
      DeviceInteractionSelectionBatch* deferred_selection =
          nullptr,
      LeptonPipelineStageEvents const* stage_events = nullptr);

  void appendTransportInteractionWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceTransportInteractionBatch
  extractTransportInteractionsOnDevice(
      PhotonTransportRecord const* device_records,
      std::size_t count, DeviceWorkspace&,
      bool defer_count_download = false);

  void appendLeptonInteractionWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceTransportInteractionBatch
  extractLeptonInteractionsOnDevice(
      LeptonTransportRecord const* device_records,
      std::size_t count, DeviceWorkspace&,
      bool defer_count_download = false);

  void appendLeptonVertexSelectionWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceLeptonVertexSelectionBatch
  launchLeptonVertexSelectionOnDevice(
      tables::FlatRateTableView device_table,
      EmInteractionRecord const* device_candidates,
      std::size_t count, std::uint64_t random_seed,
      std::uint64_t shower_id, DeviceWorkspace&,
      DeviceTransportInteractionBatch*
          deferred_candidates = nullptr,
      bool defer_count_download = false);

  void appendPhotonPairFinalStateWorkspace(
      WorkspaceSize&, std::size_t count);

  DevicePhotonPairFinalStateBatch
  launchPhotonPairFinalStateOnDevice(
      tables::FlatRateTableView device_table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      EmInteractionRecord const* device_interactions,
      std::size_t count, std::uint64_t random_seed,
      std::uint64_t shower_id,
      std::uint64_t first_secondary_history_id,
      DeviceWorkspace&, bool defer_count_download = false,
      DeviceTransportInteractionBatch*
          deferred_interactions = nullptr);

  void appendBremsFinalStateWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceBremsFinalStateBatch launchBremsFinalStateOnDevice(
      BremsLpmSnapshot const& lpm_snapshot,
      BremsLpmPreparedSnapshot const& prepared_lpm,
      EmThinningConfig const& thinning,
      EmInteractionRecord const* device_interactions,
      std::size_t count, std::uint64_t random_seed,
      std::uint64_t shower_id,
      std::uint64_t first_secondary_history_id,
      DeviceWorkspace&, bool defer_count_download = false,
      DeviceLeptonVertexSelectionBatch*
          deferred_vertex = nullptr,
      LeptonPipelineStageEvents const* stage_events = nullptr);

  void appendLeptonEndpointWorkspace(
      WorkspaceSize&, std::size_t source_count);

  DeviceLeptonEndpointBatch compactLeptonEndpointsOnDevice(
      LeptonTransportRecord const* transport_records,
      std::size_t transport_count,
      DeviceLeptonVertexSelectionBatch& vertex,
      DeviceBremsFinalStateBatch& final_state,
      std::size_t source_count,
      DeviceWorkspace&);

  void appendPhotonEndpointWorkspace(
      WorkspaceSize&, std::size_t source_count);

  DevicePhotonEndpointBatch compactPhotonEndpointsOnDevice(
      PhotonTransportRecord const* transport_records,
      std::size_t transport_count,
      DevicePhotonPairFinalStateBatch& final_state,
      std::size_t source_count,
      DeviceChargedSecondarySink const* charged_secondary_sink,
      DeviceWorkspace&);

  void appendPhotonDevicePipelineWorkspace(
      WorkspaceSize&, std::size_t source_count);

  DevicePhotonPipelineBatch launchPhotonDevicePipelineOnDevice(
      tables::FlatRateTableView device_table,
      PhotonPairLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      EnvironmentSnapshot const& environment,
      EmParticleState const* device_particles,
      std::size_t count, std::uint64_t random_seed,
      std::uint64_t shower_id,
      std::uint64_t first_secondary_history_id,
      DeviceWorkspace&,
      DeviceChargedSecondarySink const* charged_secondary_sink =
          nullptr);

  void appendLeptonDevicePipelineWorkspace(
      WorkspaceSize&, std::size_t source_count);

  DeviceLeptonPipelineBatch launchLeptonDevicePipelineOnDevice(
      tables::FlatRateTableView device_table,
      BremsLpmSnapshot const& lpm_snapshot,
      BremsLpmPreparedSnapshot const& prepared_lpm,
      EmThinningConfig const& thinning,
      MoliereSnapshot const& moliere_snapshot,
      MoliereSnapshot const& muon_moliere_snapshot,
      MoliereInterpolationView const& moliere_interpolation,
      bool apply_moliere, bool muon_moliere_available,
      EnvironmentSnapshot const& environment,
      EmParticleState const* device_particles,
      std::size_t count, std::uint64_t random_seed,
      std::uint64_t shower_id,
      std::uint64_t first_secondary_history_id,
      DeviceWorkspace&,
      LeptonPipelineStageEvents const* stage_events = nullptr);

} // namespace corsika::gpu::em::detail
