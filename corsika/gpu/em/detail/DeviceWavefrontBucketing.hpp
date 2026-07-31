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

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>

namespace corsika::gpu::em::detail {

  /**
   * Stable PID x medium x energy ordering for one resident EM wavefront.
   *
   * The returned particle array belongs to the supplied DeviceWorkspace and
   * remains valid until that arena is prepared again. Equal keys retain input
   * order, which keeps deterministic source ordering inside every bucket.
   */
  struct DeviceWavefrontBucketBatch {
    EmParticleState const* particles{};
    std::uint64_t const* keys{};
    std::size_t count{};
    bool radix_sorted{};
  };

  struct WavefrontBucketingValidationResult {
    std::vector<EmParticleState> particles;
    std::vector<std::uint64_t> keys;
  };

  /**
   * Energy bins contain 16 equal logarithmic intervals per factor two.
   * Key layout, from most to least significant:
   *
   *   [2 PID bits][32 signed-medium-order bits][16 energy-bin bits]
   */
  inline constexpr int WavefrontBucketKeyBits = 50;
  inline constexpr std::size_t
      MinimumWavefrontRadixSortSize = 256;

  void appendWavefrontBucketingWorkspace(
      WorkspaceSize&, std::size_t count);

  DeviceWavefrontBucketBatch
  launchWavefrontBucketingOnDevice(
      EmParticleState const* input, std::size_t count,
      DeviceWorkspace&);

  WavefrontBucketingValidationResult
  bucketWavefrontForValidation(
      std::vector<EmParticleState> const&, int device,
      DeviceWorkspace&);

} // namespace corsika::gpu::em::detail
