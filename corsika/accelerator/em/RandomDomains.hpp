/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::gpu::em {

  inline constexpr std::uint32_t InteractionDistanceRandomProcessId =
      0x454d0001U;
  inline constexpr std::uint32_t InteractionColumnRandomProcessId =
      0x454d0002U;
  inline constexpr std::uint32_t ProposalSelectionRandomProcessId =
      0x454d0003U;
  inline constexpr std::uint64_t InteractionDistanceDrawId = 0;
  inline constexpr std::uint64_t InteractionColumnDrawId = 0;
  inline constexpr std::uint64_t ProposalSelectionDrawId = 0;
  inline constexpr std::uint64_t InteractionLossDrawId = 0;

  inline constexpr std::uint32_t ContinuousScatteringRandomProcessId =
      0x454d0003U;
  inline constexpr std::uint64_t MoliereFirstAngleDrawId = 0;
  inline constexpr std::uint64_t MoliereSecondAngleDrawId = 1;
  inline constexpr std::uint64_t MoliereAzimuthDrawId = 2;
  inline constexpr std::uint32_t MuonDecayRandomProcessId = 0x4d554445U;
  inline constexpr std::uint64_t MuonDecayDrawId = 0;
  inline constexpr double MuonMeanLifetimeS = 2.196981e-6;
  inline constexpr double MuonDecaySpeedOfLightMPerS = 299792458.;

} // namespace corsika::gpu::em
