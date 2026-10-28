/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  namespace thinning_detail {
    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double minimum(double first, double second) {
#if defined(__CUDA_ARCH__)
      return ::fmin(first, second);
#else
      return std::fmin(first, second);
#endif
    }
  } // namespace thinning_detail
  CORSIKA_HOST_DEVICE inline EmThinningResult apply_em_thinning(
      EmThinningConfig const& config, double parent_energy_GeV, double parent_weight,
      double first_energy_GeV, double second_energy_GeV, double first_uniform,
      double second_uniform) {
    using namespace thinning_detail;
    if (!finite(parent_energy_GeV) || !(parent_energy_GeV > 0.) ||
        !finite(parent_weight) || parent_weight < 0. || !finite(first_energy_GeV) ||
        !(first_energy_GeV > 0.) || !finite(second_energy_GeV) ||
        !(second_energy_GeV > 0.) || !finite(first_uniform) || !(first_uniform > 0.) ||
        !(first_uniform < 1.) || !finite(second_uniform) || !(second_uniform > 0.) ||
        !(second_uniform < 1.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    auto unchanged = EmThinningResult{EmThinningStatus::NotApplied, 0x3U, parent_weight,
                                      parent_weight};
    if (config.enabled_ == 0) { return unchanged; }
    if (!finite(config.thresholdGeV_) || !(config.thresholdGeV_ > 0.) ||
        !finite(config.maximumWeight_) || !(config.maximumWeight_ > 0.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    // These are the same early exits and comparisons used by EMThinning.
    if (parent_weight >= config.maximumWeight_ ||
        parent_energy_GeV > config.thresholdGeV_) {
      return unchanged;
    }

    auto const energy_sum = first_energy_GeV + second_energy_GeV;
    if (!finite(energy_sum) || !(energy_sum > 0.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }
    auto const first_probability = first_energy_GeV / energy_sum;
    auto const second_probability = second_energy_GeV / energy_sum;
    auto const first_factor = 1. / first_probability;
    auto const second_factor = 1. / second_probability;
    auto const maximum_factor = config.maximumWeight_ / parent_weight;

    EmThinningResult result{};
    if (first_factor <= maximum_factor && second_factor <= maximum_factor) {
      result.status_ = EmThinningStatus::Hillas;
      if (first_uniform <= first_probability) {
        result.keepMask_ = 0x1U;
        result.firstWeight_ = parent_weight * first_factor;
        result.secondWeight_ = 0.;
      } else {
        result.keepMask_ = 0x2U;
        result.firstWeight_ = 0.;
        result.secondWeight_ = parent_weight * second_factor;
      }
    } else {
      result.status_ = EmThinningStatus::Statistical;
      auto const first_limited_factor = minimum(first_factor, maximum_factor);
      auto const second_limited_factor = minimum(second_factor, maximum_factor);
      if (first_uniform * first_limited_factor <= 1.) {
        result.keepMask_ |= 0x1U;
        result.firstWeight_ = parent_weight * first_limited_factor;
      }
      if (second_uniform * second_limited_factor <= 1.) {
        result.keepMask_ |= 0x2U;
        result.secondWeight_ = parent_weight * second_limited_factor;
      }
    }

    if (config.eraseZeroWeight_ == 0) {
      result.keepMask_ = 0x3U;
    } else {
      // Scalar EMThinning erases by the final weight, not by the random
      // selection bit. A selected child of a zero-weight parent (or an
      // underflowed product) still has to be erased in non-multithin mode.
      if (result.firstWeight_ == 0.) result.keepMask_ &= ~0x1U;
      if (result.secondWeight_ == 0.) result.keepMask_ &= ~0x2U;
    }
    return result;
  }

} // namespace corsika::kokkos
