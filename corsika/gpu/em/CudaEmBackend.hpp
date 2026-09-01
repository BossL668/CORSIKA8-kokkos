/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/CudaLeptonSelectionTransport.hpp>
#include <corsika/gpu/em/CudaPhotonSelectionTransport.hpp>
#include <corsika/gpu/radio/Types.hpp>
#include <corsika/gpu/em/tables/ProposalNativeAux.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTable.hpp>

namespace corsika::gpu::em {

  /**
   * Resident CUDA electromagnetic wavefront backend.
   *
   * initialize() uploads invariant physics/environment data and starts the
   * first shower. beginShower() reuses those allocations for a subsequent
   * shower after the previous profile/radio outputs have been consumed.
   */
  class CudaEmBackend {
  public:
    CudaEmBackend();
    ~CudaEmBackend();

    CudaEmBackend(CudaEmBackend&&) noexcept;
    CudaEmBackend& operator=(CudaEmBackend&&) noexcept;

    CudaEmBackend(CudaEmBackend const&) = delete;
    CudaEmBackend& operator=(CudaEmBackend const&) = delete;

    void initialize(EnvironmentSnapshot const&, ProposalTableSet const&,
                    GpuEmConfig const&);
    void initialize(
        EnvironmentSnapshot const&, tables::ProposalNativeTableSet const&,
        tables::ProposalNativeAuxData const&, GpuEmConfig const&);
    void beginShower(GpuEmShowerConfig const&);
    bool canTransport(EmParticleState const&) const;
    void enqueue(EmParticleState const&);
    EmBatchResult advanceWavefront();
    void drain();
    bool empty() const;
    GpuEmStatistics const& statistics() const;
    std::size_t minimumBatchSize() const;
    bool hasProposalTable() const;
    std::array<std::uint8_t, 32> proposalTableHash() const;
    bool gpuRadioEnabled() const noexcept;
    radio::GpuRadioWaveforms downloadRadioWaveforms();
    bool gpuProfileEnabled() const noexcept;
    GpuProfileResult downloadProfile();
    std::optional<GpuFirstInteractionSnapshot>
    downloadFirstInteractionSnapshot();

    /**
     * Development bridge for the physical interaction-selection kernel.
     * It does not consume or modify the resident toy particle queue.
     */
    EmInteractionBatchResult selectInteractionsForValidation(
        std::vector<EmParticleState> const&);

    /**
     * Development bridge advancing selected photons to an interaction,
     * atmosphere boundary, observation surface or escape.
     */
    PhotonTransportBatchResult transportPhotonsForValidation(
        std::vector<EmInteractionRecord> const&);

    /**
     * Development bridge for straight charged-lepton transport with
     * continuous PROPOSAL energy loss. It rejects non-zero magnetic fields
     * and deliberately omits Moliere scattering so it remains a
     * geometry/range reference path. Production routing remains separately
     * capability-gated.
     */
    LeptonTransportBatchResult
    transportLeptonsStraightForValidation(
        std::vector<EmInteractionRecord> const&);

    LeptonVertexSelectionBatchResult
    reselectLeptonInteractionsAtVertexForValidation(
        std::vector<EmInteractionRecord> const&);

    /**
     * Device-chained lepton selection, continuous transport, Moliere
     * scattering, vertex process reselection and bremsstrahlung dispatch.
     * Moliere scattering is enabled only when versioned parameters are
     * present in the loaded table.
     */
    LeptonDevicePipelineBatchResult
    runLeptonDevicePipelineForValidation(
        std::vector<EmParticleState> const&,
        std::uint64_t first_secondary_history_id);

    /**
     * Device-chained interaction selection and spherical transport. Selected
     * interaction records are not round-tripped through the host.
     */
    PhotonSelectionTransportBatchResult
    selectAndTransportPhotonsForValidation(
        std::vector<EmParticleState> const&);

    PhotonDevicePipelineBatchResult
    runPhotonDevicePipelineForValidation(
        std::vector<EmParticleState> const&,
        std::uint64_t first_secondary_history_id);

    ResidentPhotonCascadeResult
    runResidentPhotonCascadeForValidation(
        std::vector<EmParticleState> const&,
        std::uint64_t first_secondary_history_id,
        std::size_t maximum_wavefronts = 1024,
        std::size_t minimum_resident_batch_size = 1);

    ResidentLeptonCascadeResult
    runResidentLeptonCascadeForValidation(
        std::vector<EmParticleState> const&,
        std::uint64_t first_secondary_history_id,
        std::size_t maximum_wavefronts = 1024,
        std::uint64_t secondary_history_id_limit_exclusive =
            std::numeric_limits<std::uint64_t>::max(),
        std::size_t minimum_resident_batch_size = 1);

    std::size_t maximumResidentPhotonBatchSize() const;
    std::size_t maximumResidentLeptonBatchSize() const;
    std::size_t maximumResidentInputBatchSize() const;
    std::size_t pendingPhotonCount() const noexcept;
    std::size_t pendingLeptonCount() const noexcept;

    /**
     * Complete physical photon wavefront used while resident HybridCascade
     * integration is under construction.
     */
    PhotonWavefrontBatchResult advancePhotonWavefrontForValidation(
        std::vector<EmParticleState> const&,
        std::uint64_t first_secondary_history_id);

    /**
     * Development bridge for process dispatch and the first physical GPU
     * final state. It does not modify the resident toy particle queue.
     */
    EmFinalStateBatchResult generateFinalStatesForValidation(
        std::vector<EmInteractionRecord> const&,
        std::uint64_t first_secondary_history_id);

    BremsFinalStateBatchResult
    generateBremsFinalStatesForValidation(
        std::vector<EmInteractionRecord> const&,
        std::uint64_t first_secondary_history_id);

    /**
     * Host download intended for validation, checkpointing and diagnostics. Production
     * scheduling must use counts and records returned by advanceWavefront instead.
     */
    std::vector<EmParticleState> downloadActiveParticles() const;

    /**
     * Development-only bridge used by the toy HybridCascade integration. It downloads
     * and removes the current GPU wavefront without resetting identity allocation or
     * accumulated statistics.
     */
    std::vector<EmParticleState> extractActiveParticlesForTesting();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::gpu::em
