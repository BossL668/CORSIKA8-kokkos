/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline double CpuPrimarySubshowerPolicy::gpuShare(double hostRate,
                                                    double deviceRate) noexcept {
    return hostRate > 0 && deviceRate > 0 ? deviceRate / (hostRate + deviceRate)
                                          : InitialGpuShare;
  }

  inline std::size_t CpuPrimarySubshowerPolicy::hostReserve(
      std::size_t capacity, std::size_t /*minimum*/) noexcept {
    return capacity;
  }

  inline std::size_t CpuPrimarySubshowerPolicy::helperMinimum(
      std::size_t capacity, std::size_t minimum) noexcept {
    return std::min(capacity, minimum);
  }

  inline std::size_t CpuPrimarySubshowerPolicy::hostCheckpointMinimum(
      std::size_t input, std::size_t minimum) noexcept {
    return std::max(std::size_t{1}, std::min(minimum, input / 4));
  }

  inline unsigned CpuPrimarySubshowerPolicy::selectHostSpecies(
      unsigned preferred, std::array<std::size_t, 2> const& counts,
      std::array<std::size_t, 2> const& capacities, std::size_t minimum,
      std::array<unsigned, 2>& deferrals) noexcept {
    if (!counts[preferred]) {
      deferrals[preferred] = 0; // a later arrival is not the old waiting tail
      preferred = 1 - preferred;
    }
    unsigned selected = preferred;
    unsigned other = 1 - preferred;
    if (counts[preferred] &&
        counts[preferred] < std::min(minimum, capacities[preferred]) &&
        counts[other] >= std::min(minimum, capacities[other]) && counts[other] &&
        deferrals[preferred] < MaximumSpeciesDeferrals) {
      ++deferrals[preferred];
      selected = other;
    }
    deferrals[selected] = 0;
    return selected;
  }
} // namespace corsika::kokkos
