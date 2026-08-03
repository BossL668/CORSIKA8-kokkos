/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <vector>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

  inline constexpr std::uint32_t
      ContinuousScatteringRandomProcessId = 0x454d0003U;
  inline constexpr std::uint64_t
      MoliereFirstAngleDrawId = 0;
  inline constexpr std::uint64_t
      MoliereSecondAngleDrawId = 1;
  inline constexpr std::uint64_t
      MoliereAzimuthDrawId = 2;
  inline constexpr std::uint32_t
      MuonDecayRandomProcessId = 0x4d554445U;
  inline constexpr std::uint64_t MuonDecayDrawId = 0;
  inline constexpr double MuonMeanLifetimeS = 2.196981e-6;
  inline constexpr double MuonDecaySpeedOfLightMPerS =
      299792458.;

  /**
   * Advance electrons/positrons through one spherical-atmosphere segment
   * while applying tabulated PROPOSAL continuous energy loss.
   *
   * This validation bridge deliberately rejects non-zero magnetic fields and
   * omits Moliere scattering, retaining a straight reference against which
   * the device-chained physical path can be checked.
   */
  LeptonTransportBatchResult transportLeptonsStraightForValidation(
      tables::FlatRateTableView device_table,
      EnvironmentSnapshot const& environment,
      std::vector<EmInteractionRecord> const& interactions, int device,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
