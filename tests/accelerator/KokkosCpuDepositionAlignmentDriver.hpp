/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/PhotonPairLpm.hpp>
#include <corsika/accelerator/em/common/detail/ProfileProjectionData.hpp>
#include <vector>

namespace corsika::accelerator::em::testing {

  struct CpuDepositionAlignmentResult {
    std::vector<double> energy;
    gpu::em::detail::DeviceProfileCounters counters;
    gpu::em::ProjectedEmStepRecord projected;
    double photoelectron_energy_GeV{};
  };

  CpuDepositionAlignmentResult runCpuDepositionDeviceStep(
      std::vector<double> const& axis_grammage, double axis_step_m,
      gpu::em::LeptonTransportRecord const&, bool photon, double threshold);

  CpuDepositionAlignmentResult runCpuPhotoelectricDeviceStep(
      std::vector<double> const& axis_grammage, double axis_step_m,
      gpu::em::PhotonTransportRecord const&,
      gpu::em::PhotonPairLpmSnapshot const&);

} // namespace corsika::accelerator::em::testing
