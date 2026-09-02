/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <corsika/accelerator/AcceleratorMacros.hpp>

#define CORSIKA_GPU_THINNING_HOST_DEVICE C8_ACCELERATOR_INLINE_FUNCTION

namespace corsika::gpu::em {

  // Process-specific final-state draws occupy small consecutive IDs. Keep
  // thinning in a disjoint range so enabling it cannot perturb those streams.
  inline constexpr std::uint64_t EmThinningFirstDrawId = 0x100;
  inline constexpr std::uint64_t EmThinningSecondDrawId = 0x101;

  /**
   * Device representation of the existing CORSIKA EMThinning configuration.
   *
   * The current CORSIKA process applies only to one-to-two EM splittings.
   * threshold_GeV is the parent total-energy threshold, not a secondary cut.
   */
  struct EmThinningConfig {
    double threshold_GeV{};
    double maximum_weight{};
    std::uint32_t enabled{};
    std::uint32_t erase_zero_weight{1};
  };

  enum class EmThinningStatus : std::uint32_t {
    NotApplied = 0,
    Hillas = 1,
    Statistical = 2,
    InvalidInput = 3,
  };

  /**
   * keep_mask uses bit zero for child 0 and bit one for child 1. If
   * erase_zero_weight is false, both bits remain set and discarded particles
   * carry weight zero, matching EMThinning's multithin mode.
   */
  struct EmThinningResult {
    EmThinningStatus status{EmThinningStatus::InvalidInput};
    std::uint32_t keep_mask{};
    double first_weight{};
    double second_weight{};
  };

  static_assert(std::is_standard_layout_v<EmThinningConfig>);
  static_assert(std::is_trivially_copyable_v<EmThinningConfig>);
  static_assert(std::is_standard_layout_v<EmThinningResult>);
  static_assert(std::is_trivially_copyable_v<EmThinningResult>);

  namespace thinning_detail {

    CORSIKA_GPU_THINNING_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    CORSIKA_GPU_THINNING_HOST_DEVICE inline double minimum(
        double first, double second) {
#if defined(__CUDA_ARCH__)
      return ::fmin(first, second);
#else
      return std::fmin(first, second);
#endif
    }

  } // namespace thinning_detail

  /**
   * Apply the exact branching rules of
   * corsika/detail/modules/thinning/EMThinning.inl to an already generated
   * one-to-two electromagnetic final state.
   *
   * Random numbers are explicit so callers can use keyed Philox draws.
   * first_uniform is used for the Hillas choice or the first independent
   * statistical decision; second_uniform is used only for the second
   * statistical decision.
   */
  CORSIKA_GPU_THINNING_HOST_DEVICE inline EmThinningResult
  applyEmThinning(
      EmThinningConfig const& config, double parent_energy_GeV,
      double parent_weight, double first_energy_GeV,
      double second_energy_GeV, double first_uniform,
      double second_uniform) {
    using namespace thinning_detail;
    if (!finite(parent_energy_GeV) ||
        !(parent_energy_GeV > 0.) || !finite(parent_weight) ||
        parent_weight < 0. || !finite(first_energy_GeV) ||
        !(first_energy_GeV > 0.) ||
        !finite(second_energy_GeV) ||
        !(second_energy_GeV > 0.) ||
        !finite(first_uniform) || !(first_uniform > 0.) ||
        !(first_uniform < 1.) ||
        !finite(second_uniform) || !(second_uniform > 0.) ||
        !(second_uniform < 1.)) {
      return {
          EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    auto unchanged = EmThinningResult{
        EmThinningStatus::NotApplied, 0x3U, parent_weight,
        parent_weight};
    if (config.enabled == 0) {
      return unchanged;
    }
    if (!finite(config.threshold_GeV) ||
        !(config.threshold_GeV > 0.) ||
        !finite(config.maximum_weight) ||
        !(config.maximum_weight > 0.)) {
      return {
          EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    // These are the same early exits and comparisons used by EMThinning.
    if (parent_weight >= config.maximum_weight ||
        parent_energy_GeV > config.threshold_GeV) {
      return unchanged;
    }

    auto const energy_sum =
        first_energy_GeV + second_energy_GeV;
    if (!finite(energy_sum) || !(energy_sum > 0.)) {
      return {
          EmThinningStatus::InvalidInput, 0, 0., 0.};
    }
    auto const first_probability =
        first_energy_GeV / energy_sum;
    auto const second_probability =
        second_energy_GeV / energy_sum;
    auto const first_factor = 1. / first_probability;
    auto const second_factor = 1. / second_probability;
    auto const maximum_factor =
        config.maximum_weight / parent_weight;

    EmThinningResult result{};
    if (first_factor <= maximum_factor &&
        second_factor <= maximum_factor) {
      result.status = EmThinningStatus::Hillas;
      if (first_uniform <= first_probability) {
        result.keep_mask = 0x1U;
        result.first_weight =
            parent_weight * first_factor;
        result.second_weight = 0.;
      } else {
        result.keep_mask = 0x2U;
        result.first_weight = 0.;
        result.second_weight =
            parent_weight * second_factor;
      }
    } else {
      result.status = EmThinningStatus::Statistical;
      auto const first_limited_factor =
          minimum(first_factor, maximum_factor);
      auto const second_limited_factor =
          minimum(second_factor, maximum_factor);
      if (first_uniform * first_limited_factor <= 1.) {
        result.keep_mask |= 0x1U;
        result.first_weight =
            parent_weight * first_limited_factor;
      }
      if (second_uniform * second_limited_factor <= 1.) {
        result.keep_mask |= 0x2U;
        result.second_weight =
            parent_weight * second_limited_factor;
      }
    }

    if (config.erase_zero_weight == 0) {
      result.keep_mask = 0x3U;
    }
    return result;
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_THINNING_HOST_DEVICE
