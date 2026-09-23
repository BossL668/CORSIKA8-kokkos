/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <corsika/framework/utility/LongitudinalCrossings.hpp>

namespace corsika::gpu::em::detail {

  inline constexpr std::size_t DeviceProfileHistogramCount = 8;

  inline constexpr std::size_t deviceProfileHistogramBytes(
      std::size_t const bins) {
    return DeviceProfileHistogramCount * bins * sizeof(long long);
  }

  /** Backend-neutral POD views used by CUDA and Kokkos profile kernels. */
  struct DeviceProfileProjection {
    double axis_start_position_m[3]{};
    double axis_direction[3]{};
    double axis_step_length_m{};
    double const* axis_grammage_g_per_cm2{};
    std::size_t axis_support_count{};
  };

  struct DeviceProfileCounters {
    unsigned long long steps{};
    unsigned long long deposited_steps{};
    unsigned long long photon_cuts{};
    unsigned long long lepton_limits[8]{};
    unsigned long long moliere_trials{};
    unsigned long long moliere_deflections{};
    unsigned long long moliere_zero_deflections{};
    unsigned long long moliere_newton_iterations{};
    unsigned long long moliere_max_newton_iterations{};
    unsigned long long thinning_hillas_vertices{};
    unsigned long long thinning_statistical_vertices{};
    unsigned long long thinning_particles_discarded{};
    unsigned long long fixed_point_overflows{};
    unsigned long long invalid_records{};
    long long weighted_medium_rest_mass_input{};
    long long weighted_cut_rest_mass_energy{};
    long long weighted_observed_total_energy{};
    long long weighted_escaped_total_energy{};
    long long weighted_unwritten_photoelectric_binding_energy{};
    long long weighted_observation_cut_overlap_energy{};
    long long weighted_mass_convention_correction{};
  };

  struct DeviceProfileAccumulator {
    long long* photons{};
    long long* electrons{};
    long long* positrons{};
    long long* muons_minus{};
    long long* muons_plus{};
    long long* muon_parent_productions{};
    long long* energy_loss{};
    long long* muon_energy_loss{};
    DeviceProfileCounters* counters{};
    std::size_t bins{};
    double bin_width_g_per_cm2{};
    ProfileCrossingMode crossing_mode{ProfileCrossingMode::Both};
    double energy_loss_threshold_g_per_cm2{};
    double weight_scale{};
    double inverse_weight_scale{};
    double energy_scale{};
    double inverse_energy_scale{};
  };

} // namespace corsika::gpu::em::detail
