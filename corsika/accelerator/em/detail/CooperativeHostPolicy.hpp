/* Host scheduling only; never changes physics or single-endpoint policies. */
#pragma once
#include <algorithm>
#include <cstddef>

namespace corsika::accelerator::em::detail {
struct CooperativeHostPolicy {
  // Keep the laptop footprint; grow the shared arena (not one arena/thread)
  // for large machines, with a strict upper bound even for huge thread counts.
  static std::size_t capacity(std::size_t threads) noexcept {
    auto target=64*std::clamp<std::size_t>(threads,1,256);
    std::size_t result=2048;
    while(result<target) result*=2;
    return result;
  }
  static double initialShare(std::size_t threads) noexcept {
    return std::clamp(static_cast<double>(threads)/320.,1./16.,.5);
  }
};
} // namespace corsika::accelerator::em::detail
