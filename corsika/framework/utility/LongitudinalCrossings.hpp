/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <cfloat>
#include <cmath>
#include <cstddef>

namespace corsika {
  // An observer policy only: never changes transport, cuts, RNG or radio tracks.
  enum class ProfileCrossingMode {
    Both,
    Forward,
    OriginalC8
  };
}

namespace corsika::detail {
  struct LongitudinalCrossingBins {
    std::size_t begin = 0;
    std::size_t end = 0; // exclusive
  };

  // Weighted number of plane crossings, NOT signed flux. Both directions
  // count positively. Coordinates are already divided by the profile dX.
  // Use (min(start,end), max(start,end)] so splitting a monotone track at
  // a plane counts it once. This matches C7's LPCTE=floor(X/dX)+1 cursor
  // for interior longitudinal planes, including its backwards branch.
  // Clip in floating point before conversion: negative/out-of-grid tracks
  // must not wrap around through an unsigned bin index.
  C8_ACCELERATOR_INLINE_FUNCTION inline LongitudinalCrossingBins
  longitudinalCrossingBins(double start, double end, std::size_t count) {
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

  C8_ACCELERATOR_INLINE_FUNCTION inline LongitudinalCrossingBins
  longitudinalProfileBins(double start, double end, std::size_t count,
                          ProfileCrossingMode mode) {
    if (mode == ProfileCrossingMode::Both)
      return longitudinalCrossingBins(start, end, count);
    if (!(end > start)) return {};
    if (mode == ProfileCrossingMode::Forward)
      return longitudinalCrossingBins(start, end, count);
    if (mode != ProfileCrossingMode::OriginalC8 || count == 0 ||
        !(start >= -DBL_MAX && start <= DBL_MAX) ||
        !(end >= -DBL_MAX && end <= DBL_MAX)) return {};
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
} // namespace corsika::detail
