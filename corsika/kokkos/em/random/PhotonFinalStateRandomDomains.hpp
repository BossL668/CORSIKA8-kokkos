/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
namespace corsika::kokkos {
  /** Production photon draw identities, independent of work order.
   * Analytic-candidate/acceptance ranges remain reserved for stream compatibility;
   * this implementation samples pair sharing only through the native CDF.
   */
  inline constexpr std::uint64_t PhotonPairSplitDrawId = 1;
  inline constexpr std::uint64_t PhotonPairAzimuthDrawId = 2;
  inline constexpr std::uint64_t PhotonPairElectronPolarDrawId = 3;
  inline constexpr std::uint64_t PhotonPairPositronPolarDrawId = 4;
  inline constexpr std::uint64_t PhotonPairLpmDrawId = 5;
  inline constexpr std::uint64_t PhotonPairAnalyticCandidateDrawIdBase = 0x20;
  inline constexpr std::uint64_t PhotonPairAnalyticAcceptanceDrawIdBase = 0x60;
  inline constexpr std::uint32_t PhotonPairAnalyticMaximumAttempts = 64;
  inline constexpr std::uint64_t ComptonAzimuthDrawId = 1;
  inline constexpr std::uint64_t PhotonMuonPairSplitDrawId = std::uint64_t{1} << 32;
  inline constexpr std::uint64_t PhotonMuonPairAzimuthDrawId =
      PhotonMuonPairSplitDrawId + 1;
} // namespace corsika::kokkos
