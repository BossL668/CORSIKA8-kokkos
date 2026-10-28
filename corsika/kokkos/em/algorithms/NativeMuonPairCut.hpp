/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>

namespace corsika::kokkos {

  /** Original tag for newly created muons at/below the native transport mass. */
  inline constexpr std::uint32_t NativeMuonPairCutFlag = 1U;
  /** Authorize only the original zero-length compatibility cut, not ordinary transport.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool is_native_muon_pair_cut(
      ParticleState const& particle);
  /** Check the original signed-kinetic-energy deposit in GeV, without clamping it. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool is_native_muon_pair_cut_deposit(
      std::int32_t pid, std::uint32_t tag, double energy, double deposit);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/NativeMuonPairCut.inl>
