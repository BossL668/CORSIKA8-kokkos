/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <memory>
#include <corsika/kokkos/em/execution/IEmBackend.hpp>
#include <corsika/kokkos/em/execution/BatchResults.hpp>
#include <corsika/kokkos/runtime/RuntimeConfig.hpp>
#include <corsika/kokkos/runtime/RuntimeInfo.hpp>

namespace corsika::kokkos {
  struct EnvironmentDataView;
  struct TransportGeometryConfig;
  struct ProposalNativeTableSet;
  struct ProposalNativeAuxData;
  class EmBackend final : public IEmBackend {
  public:
    explicit EmBackend(RuntimeConfig const& = {});
    ~EmBackend() override;

    EmBackend(EmBackend&&) noexcept;
    EmBackend& operator=(EmBackend&&) noexcept;
    EmBackend(EmBackend const&) = delete;
    EmBackend& operator=(EmBackend const&) = delete;

    void initialize(EnvironmentDataView const&, TransportGeometryConfig const&,
                    ProposalNativeTableSet const&, ProposalNativeAuxData const&,
                    BackendConfig const&);

    InteractionBatchResult selectInteractionsForValidation(
        std::vector<ParticleState> const&);
    PhotonTransportBatchResult transportPhotonsForValidation(
        std::vector<InteractionRecord> const&);
    LeptonTransportBatchResult transportLeptonsForValidation(
        std::vector<InteractionRecord> const&);
    LeptonVertexSelectionBatchResult selectLeptonVerticesForValidation(
        std::vector<InteractionRecord> const&);
    LeptonVertexSelectionBatchResult reselectLeptonInteractionsAtVertexForValidation(
        std::vector<InteractionRecord> const& interactions);
    PhotonFinalStateBatchResult generatePhotonFinalStatesForValidation(
        std::vector<InteractionRecord> const&, std::uint64_t firstSecondaryHistoryId);
    LeptonFinalStateBatchResult generateLeptonFinalStatesForValidation(
        std::vector<InteractionRecord> const&, std::uint64_t firstSecondaryHistoryId);
    radio::RadioWaveforms projectRadioForValidation(
        std::vector<LeptonTransportRecord> const&);

    void beginShower(ShowerConfig const&) override;
    bool canTransport(ParticleState const&) const override;
    bool hasProposalTable() const override;
    std::size_t minimumBatchSize() const override;
    std::size_t maximumResidentPhotonBatchSize() const override;
    std::size_t maximumResidentLeptonBatchSize() const override;
    std::size_t maximumResidentInputBatchSize() const override;
    std::size_t pendingPhotonCount() const noexcept override;
    std::size_t pendingLeptonCount() const noexcept override;
    bool independentSubshowersEnabled() const noexcept override;
    bool independentSubshowersReady() const noexcept override;
    bool batchIndependentSpecifiedFallbacks() const noexcept override;
    std::size_t advanceIndependentSubshowers(std::vector<ParticleState> const&,
                                             SubshowerCallbacks const&) override;
    PhotonCascadeResult runPhotonWavefront(std::vector<ParticleState> const&,
                                           std::uint64_t, std::size_t,
                                           std::size_t) override;
    LeptonCascadeResult runLeptonWavefront(std::vector<ParticleState> const&,
                                           std::uint64_t, std::size_t, std::uint64_t,
                                           std::size_t) override;
    BackendCapabilities capabilities() const override;
    BackendStatistics const& statistics() const override;
    bool gpuProfileEnabled() const noexcept override;
    ProfileResult downloadProfile() override;
    bool gpuRadioEnabled() const noexcept override;
    void enqueueRadioTracks(boost::span<LeptonTransportRecord const>) override;
    radio::RadioWaveforms downloadRadioWaveforms() override;
    std::optional<FirstInteractionRecord> downloadFirstInteractionSnapshot() override;
    RuntimeInfo const& runtimeInfo() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/EmBackend.inl>
