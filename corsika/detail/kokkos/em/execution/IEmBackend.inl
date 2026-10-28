/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline bool IEmBackend::independentSubshowersEnabled() const noexcept { return false; }

  inline bool IEmBackend::independentSubshowersReady() const noexcept { return false; }

  inline bool IEmBackend::batchIndependentSpecifiedFallbacks() const noexcept {
    return false;
  }

  inline std::size_t IEmBackend::advanceIndependentSubshowers(
      std::vector<ParticleState> const&, SubshowerCallbacks const&) {
    throw std::logic_error("Independent subshowers are unavailable on this backend");
  }

  inline PhotonCascadeResult IEmBackend::runResidentPhotonCascadeForValidation(
      std::vector<ParticleState> const& particles,
      std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
      std::size_t const minimumResidentBatchSize) {
    return runPhotonWavefront(particles, firstSecondaryHistoryId, maximumWavefronts,
                              minimumResidentBatchSize);
  }

  inline LeptonCascadeResult IEmBackend::runResidentLeptonCascadeForValidation(
      std::vector<ParticleState> const& particles,
      std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
      std::uint64_t const secondaryHistoryIdLimitExclusive,
      std::size_t const minimumResidentBatchSize) {
    return runLeptonWavefront(particles, firstSecondaryHistoryId, maximumWavefronts,
                              secondaryHistoryIdLimitExclusive, minimumResidentBatchSize);
  }

  inline void IEmBackend::enqueueRadioTracks(boost::span<LeptonTransportRecord const>) {
    throw std::logic_error("Host radio tracks are unavailable on this backend");
  }
} // namespace corsika::kokkos
