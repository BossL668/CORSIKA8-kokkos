/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::firn {
  inline void FrequencyGrid::validate() const {
    if (!(sampleIntervalS_ > 0.0) || !std::isfinite(sampleIntervalS_) ||
        sampleCount_ < 4 || (sampleCount_ & (sampleCount_ - 1)) != 0) {
      throw std::runtime_error(
          "frequency grid requires a finite positive dt and power-of-two N >= 4");
    }
  }

  inline std::size_t FrequencyGrid::frequencyCount() const {
    validate();
    return sampleCount_ / 2 + 1;
  }

  inline double FrequencyGrid::spacingHz() const {
    validate();
    return 1.0 / (sampleIntervalS_ * static_cast<double>(sampleCount_));
  }

  inline double FrequencyGrid::frequencyHz(std::size_t index) const {
    if (index >= frequencyCount()) throw std::out_of_range("frequency bin");
    return static_cast<double>(index) * spacingHz();
  }

  inline double FrequencyGrid::nyquistHz() const { return 0.5 / sampleIntervalS_; }
} // namespace corsika::kokkos::radio::firn
