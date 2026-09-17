/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/accelerator/em/KokkosEmBackend.hpp>
#include <corsika/accelerator/em/detail/CooperativeProfileMerge.hpp>
#include <corsika/accelerator/radio/detail/CooperativeRadioMerge.hpp>
#include <cstdint>
#include <functional>

namespace corsika::accelerator::em::detail {

// Host-only diagnostics, sampled after the endpoint is drained. This counts
// completed blocking resident waits, not CUDA kernel time or CPU utilization.
struct CooperativeCudaWaitStatistics {
  bool blocking_enabled{};
  std::uint64_t blocking_calls{};
  double blocking_host_seconds{};
};

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
  // Cooperative host boundary only; no callback runs inside a physics kernel.
  virtual void setCooperativeProgress(std::function<bool()> progress) = 0;
  // CPU-primary helper only: opt in while initialized and idle, before the
  // first transport. The selection survives beginShower(); counters do not.
  virtual void setCooperativeBlockingWait(bool enabled) = 0;
  virtual CooperativeCudaWaitStatistics cooperativeWaitStatistics() const = 0;
  virtual void prepareCooperativeWorkspace() = 0;
  // Opt-in after bounded workspace preparation, before the first transport.
  // Returns persistent shard count/bytes; single endpoints never call this.
  virtual std::pair<std::size_t, std::size_t> enableCooperativeHostProfileShards() {
    throw std::logic_error("host profile shards unavailable for this endpoint");
  }
  virtual void setCooperativePendingInputLimit(std::size_t count) = 0;
  virtual std::vector<gpu::em::EmParticleState> takeCooperativePending(
      bool photons, std::size_t count) = 0;
  virtual FixedProfileSnapshot downloadFixedProfile(
      std::string const& identity, CooperativeEndpoint endpoint) = 0;
  virtual radio::detail::FixedRadioSnapshot downloadFixedRadio(
      std::string const& identity, CooperativeEndpoint endpoint) = 0;
};

std::unique_ptr<KokkosBackendInstance> makeCudaBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeOpenMPBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeHipBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeSyclBackendInstance(
    KokkosRuntimeConfig const&);
std::unique_ptr<KokkosBackendInstance> makeCooperativeBackendInstance(
    KokkosRuntimeConfig const&);

} // namespace corsika::accelerator::em::detail
