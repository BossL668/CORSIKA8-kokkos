/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/accelerator/em/KokkosEmBackend.hpp>

namespace corsika::accelerator::em::detail {

// Host-wavefront boundary only. All per-particle calls remain compile-time
// execution-space templates; no virtual dispatch is introduced in kernels.
class KokkosBackendInstance : public IAcceleratedEmBackend {
 public:
  virtual void initialize(
      gpu::em::EnvironmentSnapshot const& environment,
      gpu::em::tables::ProposalNativeTableSet const& table,
      gpu::em::tables::ProposalNativeAuxData const& auxiliary,
      gpu::em::GpuEmConfig const& config) = 0;
  virtual gpu::em::EmInteractionBatchResult
  selectInteractionsForValidation(
      std::vector<gpu::em::EmParticleState> const& particles) = 0;
  virtual gpu::em::PhotonTransportBatchResult
  transportPhotonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) = 0;
  virtual gpu::em::LeptonTransportBatchResult
  transportLeptonsForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions) = 0;
  virtual gpu::em::LeptonVertexSelectionBatchResult
  selectLeptonVerticesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& candidates) = 0;
  virtual gpu::em::EmFinalStateBatchResult
  generatePhotonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) = 0;
  virtual gpu::em::BremsFinalStateBatchResult
  generateLeptonFinalStatesForValidation(
      std::vector<gpu::em::EmInteractionRecord> const& interactions,
      std::uint64_t const first_secondary_history_id) = 0;
  virtual gpu::radio::GpuRadioWaveforms projectRadioForValidation(
      std::vector<gpu::em::LeptonTransportRecord> const& records) = 0;
  virtual KokkosRuntimeInfo const& runtimeInfo() const noexcept = 0;
};

std::unique_ptr<KokkosBackendInstance> makeCudaBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeOpenMPBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeHipBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeSyclBackendInstance(
    KokkosRuntimeConfig const&);

} // namespace corsika::accelerator::em::detail
