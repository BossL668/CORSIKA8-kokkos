/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <boost/core/span.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#include <corsika/kokkos/em/execution/BackendCapabilities.hpp>
#include <corsika/kokkos/em/execution/BackendStatistics.hpp>
#include <corsika/kokkos/em/execution/ShowerConfig.hpp>
#include <corsika/kokkos/em/execution/PhotonCascadeResult.hpp>
#include <corsika/kokkos/em/execution/LeptonCascadeResult.hpp>
#include <corsika/kokkos/em/FirstInteractionRecord.hpp>
#include <corsika/kokkos/output/ProfileResult.hpp>
#include <corsika/kokkos/radio/RadioWaveforms.hpp>
#include <corsika/kokkos/scheduling/SubshowerCallbacks.hpp>

namespace corsika::kokkos {
  class IEmBackend {
  public:
    virtual ~IEmBackend() = default;

    virtual void beginShower(ShowerConfig const&) = 0;
    virtual bool canTransport(ParticleState const&) const = 0;
    virtual bool hasProposalTable() const = 0;

    virtual std::size_t minimumBatchSize() const = 0;
    virtual std::size_t maximumResidentPhotonBatchSize() const = 0;
    virtual std::size_t maximumResidentLeptonBatchSize() const = 0;
    virtual std::size_t maximumResidentInputBatchSize() const = 0;
    virtual std::size_t pendingPhotonCount() const noexcept = 0;
    virtual std::size_t pendingLeptonCount() const noexcept = 0;

    // Opt-in air cooperative protocol. Single endpoints retain the synchronous
    // wavefront API unchanged. All callbacks execute on the calling thread.
    virtual bool independentSubshowersEnabled() const noexcept;
    virtual bool independentSubshowersReady() const noexcept;
    // Coordinator-only policy, never queried from a device kernel. Adaptive
    // endpoints may coalesce specified CPU completions without joining the
    // other endpoint. Legacy independent/single backends retain their policy.
    virtual bool batchIndependentSpecifiedFallbacks() const noexcept;
    virtual std::size_t advanceIndependentSubshowers(std::vector<ParticleState> const&,
                                                     SubshowerCallbacks const&);

    virtual PhotonCascadeResult runPhotonWavefront(
        std::vector<ParticleState> const&, std::uint64_t firstSecondaryHistoryId,
        std::size_t maximumWavefronts, std::size_t minimumResidentBatchSize) = 0;

    virtual LeptonCascadeResult runLeptonWavefront(
        std::vector<ParticleState> const&, std::uint64_t firstSecondaryHistoryId,
        std::size_t maximumWavefronts, std::uint64_t secondaryHistoryIdLimitExclusive,
        std::size_t minimumResidentBatchSize) = 0;

    // Compatibility spelling used by the existing native-CUDA router.  It
    // remains a host-side forwarding call and introduces no device dispatch.
    PhotonCascadeResult runResidentPhotonCascadeForValidation(
        std::vector<ParticleState> const& particles,
        std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
        std::size_t const minimumResidentBatchSize);

    LeptonCascadeResult runResidentLeptonCascadeForValidation(
        std::vector<ParticleState> const& particles,
        std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
        std::uint64_t const secondaryHistoryIdLimitExclusive,
        std::size_t const minimumResidentBatchSize);

    virtual BackendCapabilities capabilities() const = 0;
    virtual BackendStatistics const& statistics() const = 0;

    virtual bool gpuProfileEnabled() const noexcept = 0;
    virtual ProfileResult downloadProfile() = 0;
    virtual bool gpuRadioEnabled() const noexcept = 0;
    /** Copy a borrowed range of CPU-produced tracks into the current shower.
     * The input view is valid only for this call; implementations own the copy
     * and flush the final partial batch before downloading. Empty input is
     * allowed for backends supporting host tracks. The default throws because
     * it does not support host-track accumulation.
     */
    virtual void enqueueRadioTracks(boost::span<LeptonTransportRecord const>);
    virtual radio::RadioWaveforms downloadRadioWaveforms() = 0;
    virtual std::optional<FirstInteractionRecord> downloadFirstInteractionSnapshot() = 0;
  };
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/IEmBackend.inl>
