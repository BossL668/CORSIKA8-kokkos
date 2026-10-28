/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <corsika/framework/utility/LongitudinalCrossings.hpp>

namespace corsika::kokkos {
  /** Original profile configuration; depths in g/cm^2, distances in metres. */
  struct ProfileProjectionConfig {
    bool enabled_{};
    /**
     * Keep the projected longitudinal profile and energy-loss histograms
     * resident on the device. When enabled, per-step projected records are
     * not copied to the host; only the final O(number-of-bins) histograms
     * are downloaded at end of shower.
     */
    bool accumulateOnDevice_{};
    double axisStartPositionM_[3]{};
    double axisDirection_[3]{};
    double axisStepLengthM_{};
    std::vector<double> axisGrammageGPerCm2_{};
    std::size_t outputBinCount_{};
    double outputBinWidthGPerCm2_{};
    ProfileCrossingMode crossingMode_{ProfileCrossingMode::Both};
    double energyLossThresholdGPerCm2_{1.e-4};
    /**
     * Expected positive ranges for checked deterministic fixed-point sums.
     * Two integer headroom bits are retained; exceeding the representable
     * range is a hard shower error rather than a silent wraparound.
     */
    double fixedPointWeightLimit_{};
    double fixedPointEnergyLimitGeV_{};
  };
} // namespace corsika::kokkos
