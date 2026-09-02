/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <vector>

#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

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
