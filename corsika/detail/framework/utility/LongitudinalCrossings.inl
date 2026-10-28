/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {
  CORSIKA_HOST_DEVICE inline LongitudinalCrossingBins longitudinal_crossing_bins(
      double start, double end, std::size_t count) {
    constexpr double finite_limit = DBL_MAX;
    if (count == 0 || start == end ||
        !(start >= -finite_limit && start <= finite_limit) ||
        !(end >= -finite_limit && end <= finite_limit))
      return {};
    auto const low = start < end ? start : end;
    auto const high = start < end ? end : start;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    auto first = ::floor(low) + 1.;
    auto stop = ::floor(high) + 1.;
#else
    auto first = std::floor(low) + 1.;
    auto stop = std::floor(high) + 1.;
#endif
    auto const limit = static_cast<double>(count);
    first = first < 0. ? 0. : first;
    stop = stop > limit ? limit : stop;
    if (!(first < stop)) return {};
    return {static_cast<std::size_t>(first), static_cast<std::size_t>(stop)};
  }
  CORSIKA_HOST_DEVICE inline LongitudinalCrossingBins longitudinal_profile_bins(
      double start, double end, std::size_t count, ProfileCrossingMode mode) {
    if (mode == ProfileCrossingMode::Both)
      return longitudinal_crossing_bins(start, end, count);
    if (!(end > start)) return {};
    if (mode == ProfileCrossingMode::Forward)
      return longitudinal_crossing_bins(start, end, count);
    if (mode != ProfileCrossingMode::OriginalC8 || count == 0 ||
        !(start >= -DBL_MAX && start <= DBL_MAX) || !(end >= -DBL_MAX && end <= DBL_MAX))
      return {};
      // Compatibility with original C8 ceil(start)..floor(end), inclusive.
      // Unlike half-open modes, an exactly shared endpoint can count twice.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    auto first = ::ceil(start);
    auto stop = ::floor(end) + 1.;
#else
    auto first = std::ceil(start);
    auto stop = std::floor(end) + 1.;
#endif
    first = first < 0. ? 0. : first;
    auto const limit = static_cast<double>(count);
    stop = stop > limit ? limit : stop;
    if (!(first < stop)) return {};
    return {static_cast<std::size_t>(first), static_cast<std::size_t>(stop)};
  }
} // namespace corsika
