/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::accelerator::em {

  // Keep all photon-final-state counter domains independent of execution
  // order.  Native CUDA and every Kokkos execution space include this same
  // definition so a backend change cannot move a history onto another
  // Philox stream.
  inline constexpr std::uint64_t PhotonPairSplitDrawId = 1;
  inline constexpr std::uint64_t PhotonPairAzimuthDrawId = 2;
  inline constexpr std::uint64_t PhotonPairElectronPolarDrawId = 3;
  inline constexpr std::uint64_t PhotonPairPositronPolarDrawId = 4;
  inline constexpr std::uint64_t PhotonPairLpmDrawId = 5;
  inline constexpr std::uint64_t PhotonPairAnalyticCandidateDrawIdBase = 0x20;
  inline constexpr std::uint64_t PhotonPairAnalyticAcceptanceDrawIdBase = 0x60;
  inline constexpr std::uint32_t PhotonPairAnalyticMaximumAttempts = 64;
  inline constexpr std::uint64_t ComptonAzimuthDrawId = 1;

} // namespace corsika::accelerator::em
