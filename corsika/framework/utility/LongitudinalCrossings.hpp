/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <cfloat>
#include <cmath>
#include <cstddef>
namespace corsika {
  /** Observer counting policy only; never changes transport, RNG or cuts.
   * Both counts both directions positively using (low, high]; Forward uses
   * that rule only for increasing depth. OriginalC8 uses ceil(start)..floor(end)
   * inclusively for increasing depth, preserving its shared-endpoint convention.
   */
  enum class ProfileCrossingMode { Both, Forward, OriginalC8 };
  struct LongitudinalCrossingBins {
    std::size_t begin_ = 0;
    std::size_t end_ = 0; // exclusive
  };
  CORSIKA_HOST_DEVICE inline LongitudinalCrossingBins longitudinal_crossing_bins(
      double start, double end, std::size_t count);
  CORSIKA_HOST_DEVICE inline LongitudinalCrossingBins longitudinal_profile_bins(
      double start, double end, std::size_t count, ProfileCrossingMode mode);
} // namespace corsika
#include <corsika/detail/framework/utility/LongitudinalCrossings.inl>
