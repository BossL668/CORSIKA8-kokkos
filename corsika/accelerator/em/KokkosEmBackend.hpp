/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <memory>
#include <vector>

#include <corsika/accelerator/em/IAcceleratedEmBackend.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>

namespace corsika::accelerator::em {

  /**
   * Execution-space-specific Kokkos electromagnetic backend.
   *
   * Its public surface intentionally contains no Kokkos type.  The selected
   * execution space is fixed at build time and hidden by the implementation,
   * which keeps ordinary C++ application translation units compatible with a
   * Kokkos-CUDA/HIP build.
   */
  class KokkosEmBackend final : public IAcceleratedEmBackend {
  public:
    explicit KokkosEmBackend(KokkosRuntimeConfig const& = {});
    ~KokkosEmBackend() override;

    KokkosEmBackend(KokkosEmBackend&&) noexcept;
    KokkosEmBackend& operator=(KokkosEmBackend&&) noexcept;
    KokkosEmBackend(KokkosEmBackend const&) = delete;
    KokkosEmBackend& operator=(KokkosEmBackend const&) = delete;

    void initialize(
        gpu::em::EnvironmentSnapshot const&,
        gpu::em::tables::ProposalNativeTableSet const&,
        gpu::em::tables::ProposalNativeAuxData const&,
        gpu::em::GpuEmConfig const&);

    gpu::em::EmInteractionBatchResult selectInteractionsForValidation(
        std::vector<gpu::em::EmParticleState> const&);
    gpu::em::PhotonTransportBatchResult transportPhotonsForValidation(
        std::vector<gpu::em::EmInteractionRecord> const&);
    gpu::em::LeptonTransportBatchResult transportLeptonsForValidation(
        std::vector<gpu::em::EmInteractionRecord> const&);
    gpu::em::LeptonVertexSelectionBatchResult
    selectLeptonVerticesForValidation(
        std::vector<gpu::em::EmInteractionRecord> const&);
    gpu::em::LeptonVertexSelectionBatchResult
    reselectLeptonInteractionsAtVertexForValidation(
        std::vector<gpu::em::EmInteractionRecord> const& interactions) {
      return selectLeptonVerticesForValidation(interactions);
    }
    gpu::em::EmFinalStateBatchResult generatePhotonFinalStatesForValidation(
        std::vector<gpu::em::EmInteractionRecord> const&,
        std::uint64_t first_secondary_history_id);
    gpu::em::BremsFinalStateBatchResult
    generateLeptonFinalStatesForValidation(
        std::vector<gpu::em::EmInteractionRecord> const&,
        std::uint64_t first_secondary_history_id);
    gpu::radio::GpuRadioWaveforms projectRadioForValidation(
        std::vector<gpu::em::LeptonTransportRecord> const&);

    void beginShower(AcceleratedEmShowerConfig const&) override;
    bool canTransport(gpu::em::EmParticleState const&) const override;
    bool hasProposalTable() const override;
    std::size_t minimumBatchSize() const override;
    std::size_t maximumResidentPhotonBatchSize() const override;
    std::size_t maximumResidentLeptonBatchSize() const override;
    std::size_t maximumResidentInputBatchSize() const override;
    std::size_t pendingPhotonCount() const noexcept override;
    std::size_t pendingLeptonCount() const noexcept override;
    gpu::em::ResidentPhotonCascadeResult runPhotonWavefront(
        std::vector<gpu::em::EmParticleState> const&, std::uint64_t,
        std::size_t, std::size_t) override;
    gpu::em::ResidentLeptonCascadeResult runLeptonWavefront(
        std::vector<gpu::em::EmParticleState> const&, std::uint64_t,
        std::size_t, std::uint64_t, std::size_t) override;
    BackendCapabilities capabilities() const override;
    AcceleratedEmStatistics const& statistics() const override;
    bool gpuProfileEnabled() const noexcept override;
    gpu::em::GpuProfileResult downloadProfile() override;
    bool gpuRadioEnabled() const noexcept override;
    gpu::radio::GpuRadioWaveforms downloadRadioWaveforms() override;
    std::optional<gpu::em::GpuFirstInteractionSnapshot>
    downloadFirstInteractionSnapshot() override;
    KokkosRuntimeInfo const& runtimeInfo() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::accelerator::em
