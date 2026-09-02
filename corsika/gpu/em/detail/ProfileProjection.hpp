/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/ProfileProjectionData.hpp>

namespace corsika::gpu::em::detail {

  void launchPhotonProfileProjectionOnDevice(
      DeviceProfileProjection const&,
      PhotonTransportRecord const* records, std::size_t count,
      ProjectedEmStepRecord* output);

  void launchLeptonProfileProjectionOnDevice(
      DeviceProfileProjection const&,
      LeptonTransportRecord const* records, std::size_t count,
      ProjectedEmStepRecord* output);

  void launchPhotonProfileAccumulationOnDevice(
      DeviceProfileProjection const&,
      DeviceProfileAccumulator const&,
      PhotonTransportRecord const* records, std::size_t count,
      PhotonPairFinalStateRecord const* final_states,
      std::size_t final_state_count,
      cudaStream_t stream = nullptr);

  void launchLeptonProfileAccumulationOnDevice(
      DeviceProfileProjection const&,
      DeviceProfileAccumulator const&,
      LeptonTransportRecord const* records, std::size_t count,
      BremsFinalStateRecord const* final_states,
      std::size_t final_state_count,
      cudaStream_t stream = nullptr);

} // namespace corsika::gpu::em::detail
