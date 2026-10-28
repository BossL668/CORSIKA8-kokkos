/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Frozen production 41cdc543; only namespace/includes/device annotations adapted.
#include <Kokkos_Macros.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>
namespace final_state_reference {
  struct alignas(16) EmParticleState {
    std::int32_t pid{};
    std::int32_t medium_id{};
    std::uint32_t generation{};
    std::uint32_t reserved{};

    double energy_GeV{};
    double position_m[3]{};
    double direction[3]{};
    double time_s{};
    double weight{1.};

    std::uint64_t history_id{};
    std::uint64_t parent_history_id{};
    std::uint64_t step_id{};
    // Independent of medium_id: distinct logical nodes may share one model.
    // -3 denotes legacy/unassigned, -1 denotes outside the exported tree.
    std::int32_t environment_node{-3};
  };
  struct GpuFirstInteractionSnapshot {
    EmParticleState parent_at_vertex{};
    EmParticleState secondaries[3]{};
    std::int32_t process_id{};
    std::uint32_t secondary_count{};
  };
  struct PhotonFinalStateRecord {
    std::uint64_t input_index{};
    std::uint64_t parent_history_id{};
    std::uint64_t secondary_offset{};
    std::uint32_t secondary_count{};
    std::int32_t process_id{};
    double energy_split_fraction{};
    double split_uniform{};
    double azimuth_uniform{};
    double electron_polar_uniform{};
    double positron_polar_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    std::uint64_t split_draw_id{};
    std::uint64_t azimuth_draw_id{};
    std::uint64_t electron_polar_draw_id{};
    std::uint64_t positron_polar_draw_id{};
    std::uint64_t lpm_draw_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    std::uint64_t thinning_first_draw_id{};
    std::uint64_t thinning_second_draw_id{};
    double weighted_mass_convention_correction_GeV{};
  };
  struct BremsFinalStateRecord {
    std::uint64_t input_index{};
    std::uint64_t parent_history_id{};
    std::uint64_t secondary_offset{};
    std::uint32_t secondary_count{};
    std::int32_t process_id{};
    double photon_energy_fraction{};
    double final_state_uniform{};
    double azimuth_uniform{};
    double auxiliary_uniform{};
    double lpm_survival_probability{};
    double lpm_uniform{};
    std::uint64_t final_state_draw_id{};
    std::uint64_t azimuth_draw_id{};
    std::uint64_t auxiliary_draw_id{};
    std::uint64_t lpm_draw_id{};
    std::uint32_t thinning_status{};
    std::uint32_t thinning_keep_mask{0x3U};
    double thinning_first_uniform{};
    double thinning_second_uniform{};
    std::uint64_t thinning_first_draw_id{};
    std::uint64_t thinning_second_draw_id{};
    double weighted_mass_convention_correction_GeV{};
  };
  struct PhotonPairLpmSuppressionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::uint64_t component_hash{};
    double survival_probability{};
    double uniform{};
    std::uint64_t draw_id{};
  };
  struct BremsLpmSuppressionRecord {
    EmParticleState particle{};
    std::uint64_t input_index{};
    std::uint64_t component_hash{};
    double survival_probability{};
    double uniform{};
    std::uint64_t draw_id{};
  };
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

    KOKKOS_INLINE_FUNCTION bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }

    KOKKOS_INLINE_FUNCTION double minimum(double first, double second) {
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
  KOKKOS_INLINE_FUNCTION EmThinningResult
  applyEmThinning(EmThinningConfig const& config, double parent_energy_GeV,
                  double parent_weight, double first_energy_GeV, double second_energy_GeV,
                  double first_uniform, double second_uniform) {
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
    if (config.enabled == 0) { return unchanged; }
    if (!finite(config.threshold_GeV) || !(config.threshold_GeV > 0.) ||
        !finite(config.maximum_weight) || !(config.maximum_weight > 0.)) {
      return {EmThinningStatus::InvalidInput, 0, 0., 0.};
    }

    // These are the same early exits and comparisons used by EMThinning.
    if (parent_weight >= config.maximum_weight ||
        parent_energy_GeV > config.threshold_GeV) {
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
    auto const maximum_factor = config.maximum_weight / parent_weight;

    EmThinningResult result{};
    if (first_factor <= maximum_factor && second_factor <= maximum_factor) {
      result.status = EmThinningStatus::Hillas;
      if (first_uniform <= first_probability) {
        result.keep_mask = 0x1U;
        result.first_weight = parent_weight * first_factor;
        result.second_weight = 0.;
      } else {
        result.keep_mask = 0x2U;
        result.first_weight = 0.;
        result.second_weight = parent_weight * second_factor;
      }
    } else {
      result.status = EmThinningStatus::Statistical;
      auto const first_limited_factor = minimum(first_factor, maximum_factor);
      auto const second_limited_factor = minimum(second_factor, maximum_factor);
      if (first_uniform * first_limited_factor <= 1.) {
        result.keep_mask |= 0x1U;
        result.first_weight = parent_weight * first_limited_factor;
      }
      if (second_uniform * second_limited_factor <= 1.) {
        result.keep_mask |= 0x2U;
        result.second_weight = parent_weight * second_limited_factor;
      }
    }

    if (config.erase_zero_weight == 0) {
      result.keep_mask = 0x3U;
    } else {
      // Scalar EMThinning erases by the final weight, not by the random
      // selection bit. A selected child of a zero-weight parent (or an
      // underflowed product) still has to be erased in non-multithin mode.
      if (result.first_weight == 0.) result.keep_mask &= ~0x1U;
      if (result.second_weight == 0.) result.keep_mask &= ~0x2U;
    }
    return result;
  }

} // namespace final_state_reference
