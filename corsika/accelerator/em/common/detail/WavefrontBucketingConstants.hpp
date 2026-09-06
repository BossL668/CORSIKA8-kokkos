/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>

namespace corsika::gpu::em::detail {

  /** Number of significant bits in the PID x medium x energy bucket key. */
  inline constexpr int WavefrontBucketKeyBits = 50;

  /**
   * Below this size, stable radix sorting costs more than it saves in the
   * resident electromagnetic transport kernels.
   */
  inline constexpr std::size_t MinimumWavefrontRadixSortSize = 256;

} // namespace corsika::gpu::em::detail
