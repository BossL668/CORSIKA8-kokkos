#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace corsika::accelerator::em::detail {
struct FixedAccumulatorLayout {
  // Identity covers algorithm, observers, binning and units, not just length.
  std::string identity;
  double units_per_integer{};
  std::size_t bins{};
};

inline void mergeFixedAccumulators(std::vector<std::int64_t>& destination,
                                  FixedAccumulatorLayout const& a,
                                  std::vector<std::int64_t> const& source,
                                  FixedAccumulatorLayout const& b) {
  if (a.identity.empty() || a.identity != b.identity || a.bins != b.bins ||
      !std::isfinite(a.units_per_integer) || a.units_per_integer <= 0. ||
      a.units_per_integer != b.units_per_integer || destination.size() != a.bins ||
      source.size() != b.bins)
    throw std::invalid_argument("incompatible fixed-point accumulators");
  // Preflight every bin before writing: failed merges cannot partially commit.
  for (std::size_t i = 0; i < source.size(); ++i) {
    auto x = destination[i], y = source[i];
    if ((y > 0 && x > std::numeric_limits<std::int64_t>::max() - y) ||
        (y < 0 && x < std::numeric_limits<std::int64_t>::min() - y))
      throw std::overflow_error("cooperative fixed-point accumulation overflow");
  }
  for (std::size_t i = 0; i < source.size(); ++i) destination[i] += source[i];
}
} // namespace corsika::accelerator::em::detail
