/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {
  // Native PROPOSAL IDs. The 20000000xx identities query secondary splits,
  // and must not be registered as additional reaction-rate columns.
  inline constexpr std::int32_t BremsProcessId = 1000000002;
  inline constexpr std::int32_t IonizationProcessId = 1000000003;
  inline constexpr std::int32_t ElectronPairProcessId = 1000000004;
  inline constexpr std::int32_t PhotonuclearProcessId = 1000000005;
  inline constexpr std::int32_t MuonPairProcessId = 1000000006;
  inline constexpr std::int32_t HadronsProcessId = 1000000007;
  inline constexpr std::int32_t WeakInteractionProcessId = 1000000009;
  inline constexpr std::int32_t ComptonProcessId = 1000000010;
  inline constexpr std::int32_t DecayProcessId = 1000000011;
  inline constexpr std::int32_t AnnihilationProcessId = 1000000012;
  inline constexpr std::int32_t PhotonPairProcessId = 1000000013;
  inline constexpr std::int32_t PhotonPairFinalStateProcessId = 2000000013;
  inline constexpr std::int32_t PhotoproductionProcessId = 1000000014;
  inline constexpr std::int32_t PhotonMuonPairProcessId = 1000000015;
  inline constexpr std::int32_t PhotonMuonPairFinalStateProcessId = 2000000015;
  inline constexpr std::int32_t PhotoelectricProcessId = 1000000016;
} // namespace corsika::kokkos
