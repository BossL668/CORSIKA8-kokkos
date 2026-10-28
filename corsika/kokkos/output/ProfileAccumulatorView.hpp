/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/framework/utility/LongitudinalCrossings.hpp>
#include <corsika/kokkos/output/ProfileCounters.hpp>

namespace corsika::kokkos {

  /** Borrowed fixed-point histogram storage in the executing memory space.
   * The owner supplies eight valid bins_-element arrays and a valid counters_;
   * muonParentProductions_ may be null. Zero bins is an empty histogram.
   * Width/scales must be finite and positive and all nonnegative bin indices
   * representable by int (size_t for parent production). Step depths must be
   * finite; equal-depth continuous deposits require a positive threshold.
   * Point deposits deliberately use a separate path, valid at zero threshold.
   * Keep storage alive through kernel completion; this view never allocates.
   */
  struct ProfileAccumulatorView {
    long long* photons_{};
    long long* electrons_{};
    long long* positrons_{};
    long long* muonsMinus_{};
    long long* muonsPlus_{};
    long long* muonParentProductions_{};
    long long* energyLoss_{};
    long long* muonEnergyLoss_{};
    ProfileCounters* counters_{};
    std::size_t bins_{};
    double binWidthGPerCm2_{};
    ProfileCrossingMode crossingMode_{ProfileCrossingMode::Both};
    double energyLossThresholdGPerCm2_{};
    double weightScale_{};
    double inverseWeightScale_{};
    double energyScale_{};
    double inverseEnergyScale_{};
  };
  static_assert(std::is_standard_layout_v<ProfileAccumulatorView>);
  static_assert(std::is_trivially_copyable_v<ProfileAccumulatorView>);
  inline constexpr std::size_t ProfileHistogramCount = 8;
  /** Unchecked production size formula; the owner must check allocation overflow. */
  CORSIKA_HOST_DEVICE inline constexpr std::size_t profile_histogram_bytes(
      std::size_t bins);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/ProfileAccumulatorView.inl>
