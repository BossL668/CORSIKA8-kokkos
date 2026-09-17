/* Host scheduling only. Does not alter transport, RNG or single-endpoint work. */
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>

namespace corsika::accelerator::em::detail {
struct CpuPrimarySubshowerPolicy {
  // Mirror the legacy primary/auxiliary split: learn ONLY the input share.
  // No time budget, rate-derived input limit, or shrinking resident arena.
  static constexpr double initial_gpu_share = 1. / 16.;
  static double gpuShare(double host_rate, double device_rate) noexcept {
    return host_rate > 0 && device_rate > 0 ? device_rate / (host_rate + device_rate) : initial_gpu_share;
  }
  // The primary must retain a FULL native arena, not just the minimum launch
  // threshold. Donating from that arena fragments the 130-thread wavefront.
  // This protects deliberately assigned input, not in-flight descendant counts.
  static std::size_t hostReserve(std::size_t capacity, std::size_t /*minimum*/) noexcept {
    return capacity;
  }
  // Start auxiliary work only with a useful new batch. Already-owned tails and
  // explicit bounded memory spills still drain; this is never a particle cut.
  static std::size_t helperMinimum(std::size_t capacity, std::size_t minimum) noexcept {
    return std::min(capacity, minimum);
  }
  // The primary uses the ordinary resident lepton horizon; the auxiliary
  // uses the legacy helper horizon. These are safe checkpoints, not cuts.
  static constexpr std::size_t host_lepton_waves = 1024;
  static constexpr std::size_t auxiliary_lepton_waves = 8;

  // An initially sub-minimum front must not checkpoint after every single
  // wave solely because the threshold exceeds its initial population.
  // A quarter-front threshold is a scheduling boundary, NOT a particle cut:
  // all remaining particles rejoin their owned queues at return. Keep the
  // ordinary native threshold once a front is at least four times as large.
  static std::size_t hostCheckpointMinimum(std::size_t input, std::size_t minimum) noexcept {
    return std::max(std::size_t{1}, std::min(minimum, input / 4));
  }

  // Scheduling only: prefer a useful other-species front over a tiny one.
  // Never wait for future arrivals or change the native arena/wave horizon.
  // Count deferrals in completed host epochs, not elapsed time, so a small
  // species receives service after at most eight such choices even if the
  // other front continually replenishes itself. A lone tail drains at once.
  static constexpr unsigned maximum_species_deferrals = 8;
  static unsigned selectHostSpecies(
      unsigned preferred, std::array<std::size_t, 2> const& counts,
      std::array<std::size_t, 2> const& capacities, std::size_t minimum,
      std::array<unsigned, 2>& deferrals) noexcept {
    if (!counts[preferred]) {
      deferrals[preferred] = 0;  // a later arrival is not the old waiting tail
      preferred = 1 - preferred;
    }
    unsigned selected = preferred;
    unsigned other = 1 - preferred;
    if (counts[preferred] &&
        counts[preferred] < std::min(minimum, capacities[preferred]) &&
        counts[other] >= std::min(minimum, capacities[other]) &&
        counts[other] && deferrals[preferred] < maximum_species_deferrals) {
      ++deferrals[preferred];
      selected = other;
    }
    deferrals[selected] = 0;
    return selected;
  }
};
} // namespace corsika::accelerator::em::detail
