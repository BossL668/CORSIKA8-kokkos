/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <corsika/validation/CudaDecisionReplayTypes.hpp>

namespace corsika::gpu::em {

  struct CudaDecisionReplayVerification {
    std::uint64_t records{};
    std::uint64_t electromagnetic_records{};
    std::uint64_t lepton_records{};
    std::uint64_t photon_records{};
    std::uint64_t invalid_records{};
    std::uint64_t nonfinite_records{};
    std::uint64_t negative_energy_records{};
    std::uint64_t negative_weight_records{};
    std::uint64_t backward_time_records{};
    std::uint64_t invalid_direction_records{};
    std::uint64_t superluminal_records{};
    std::uint64_t byte_hash_mismatches{};
    std::uint64_t ordered_host_hash{};
    std::uint64_t ordered_device_hash{};
    double weighted_track_length_m{};
    double weighted_lepton_track_length_m{};
    double maximum_direction_norm_error{};
    double maximum_speed_over_c{};
  };

  /**
   * Upload a complete scalar decision tape and verify every record on CUDA.
   *
   * The byte hash proves that the device consumed the exact recorded states;
   * physical sanity flags catch corrupt/non-causal segments.  The reported
   * superluminal counter is diagnostic rather than fatal because the legacy
   * curved leapfrog Step can have |dx|/dt > c before its direction
   * renormalisation.  This function intentionally does not claim that CUDA
   * sampled those decisions itself.
   */
  CudaDecisionReplayVerification verifyDecisionReplayOnCuda(
      std::vector<validation::ReplayTransportStep> const& records,
      int device);

} // namespace corsika::gpu::em
