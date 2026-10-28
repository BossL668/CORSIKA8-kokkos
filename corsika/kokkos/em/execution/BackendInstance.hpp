/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <memory>
#include <string>
#include <functional>
#include <utility>
#include <corsika/kokkos/em/execution/IEmBackend.hpp>
#include <corsika/kokkos/em/execution/BatchResults.hpp>
#include <corsika/kokkos/runtime/RuntimeConfig.hpp>
#include <corsika/kokkos/runtime/RuntimeInfo.hpp>
#include <corsika/kokkos/scheduling/CooperativeEndpoint.hpp>

namespace corsika::kokkos {
  struct EnvironmentDataView;
  struct TransportGeometryConfig;
  struct ProposalNativeTableSet;
  struct ProposalNativeAuxData;
  struct FixedProfileSnapshot;
  namespace radio {
    struct FixedRadioSnapshot;
  }
  struct CooperativeCudaWaitStatistics {
    bool blockingEnabled_{};
    std::uint64_t blockingCalls_{};
    double blockingHostSeconds_{};
  };
  class BackendInstance : public IEmBackend {
  public:
    virtual void initialize(EnvironmentDataView const& environment,
                            TransportGeometryConfig const& geometry,
                            ProposalNativeTableSet const& table,
                            ProposalNativeAuxData const& auxiliary,
                            BackendConfig const& config) = 0;
    virtual InteractionBatchResult selectInteractionsForValidation(
        std::vector<ParticleState> const& particles) = 0;
    virtual PhotonTransportBatchResult transportPhotonsForValidation(
        std::vector<InteractionRecord> const& interactions) = 0;
    virtual LeptonTransportBatchResult transportLeptonsForValidation(
        std::vector<InteractionRecord> const& interactions) = 0;
    virtual LeptonVertexSelectionBatchResult selectLeptonVerticesForValidation(
        std::vector<InteractionRecord> const& candidates) = 0;
    virtual PhotonFinalStateBatchResult generatePhotonFinalStatesForValidation(
        std::vector<InteractionRecord> const& interactions,
        std::uint64_t const firstSecondaryHistoryId) = 0;
    virtual LeptonFinalStateBatchResult generateLeptonFinalStatesForValidation(
        std::vector<InteractionRecord> const& interactions,
        std::uint64_t const firstSecondaryHistoryId) = 0;
    virtual radio::RadioWaveforms projectRadioForValidation(
        std::vector<LeptonTransportRecord> const& records) = 0;
    virtual RuntimeInfo const& runtimeInfo() const noexcept = 0;
    // Cooperative host boundary only; no callback runs inside a physics kernel.
    virtual void setCooperativeProgress(std::function<bool()> progress) = 0;
    // CPU-primary helper only: opt in while initialized and idle, before the
    // first transport. The selection survives beginShower(); counters do not.
    virtual void setCooperativeBlockingWait(bool enabled) = 0;
    virtual CooperativeCudaWaitStatistics cooperativeWaitStatistics() const = 0;
    virtual void prepareCooperativeWorkspace() = 0;
    // Opt-in after bounded workspace preparation, before the first transport.
    // Returns persistent shard count/bytes; single endpoints never call this.
    virtual std::pair<std::size_t, std::size_t> enableCooperativeHostProfileShards();
    virtual void setCooperativePendingInputLimit(std::size_t count) = 0;
    virtual std::vector<ParticleState> takeCooperativePending(bool photons,
                                                              std::size_t count) = 0;
    virtual FixedProfileSnapshot downloadFixedProfile(std::string const& identity,
                                                      CooperativeEndpoint endpoint) = 0;
    virtual radio::FixedRadioSnapshot downloadFixedRadio(
        std::string const& identity, CooperativeEndpoint endpoint) = 0;

  protected:
    /** Log initialization defaults on the host without importing native C8
     * particle formatters into a device compilation unit.
     */
    static void reportUnmatchedTuningCache(std::string const& backend);
  };
  std::unique_ptr<BackendInstance> make_cuda_backend_instance(RuntimeConfig const&);
  std::unique_ptr<BackendInstance> make_openmp_backend_instance(RuntimeConfig const&);
  std::unique_ptr<BackendInstance> make_hip_backend_instance(RuntimeConfig const&);
  std::unique_ptr<BackendInstance> make_sycl_backend_instance(RuntimeConfig const&);
  std::unique_ptr<BackendInstance> make_cooperative_backend_instance(
      RuntimeConfig const&);

} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/BackendInstance.inl>
