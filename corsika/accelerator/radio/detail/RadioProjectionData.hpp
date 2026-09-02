/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace corsika::accelerator::radio::detail {

  inline constexpr double SpeedOfLightMPerS = 299792458.;
  inline constexpr double VacuumPermittivityFPerM = 8.8541878128e-12;
  // Keep the scalar CORSIKA constant exactly; do not substitute a newer
  // CODATA rounding here because radio replay uses the scalar result as its
  // oracle.
  inline constexpr double ElementaryChargeC = 1.6021766208e-19;
  inline constexpr double Pi = 3.141592653589793238462643383279502884;
  inline constexpr double EmConstant =
      1. / (4. * Pi * VacuumPermittivityFPerM * SpeedOfLightMPerS);
  inline constexpr double CoREASApproximationThreshold = 1.e-3;
  inline constexpr double FixedPointHeadroom = 0x1p62;
  inline constexpr double SignedIntegerLimit = 0x1p63;
  inline constexpr long long SignedIntegerMaximum = 9223372036854775807LL;
  inline constexpr long long SignedIntegerMinimum =
      (-9223372036854775807LL - 1LL);

  struct DeviceObserver {
    double position_m[3]{};
    double start_time_s{};
    double duration_s{};
    double sample_rate_Hz{};
    double fixed_point_scale{};
    double inverse_fixed_point_scale{};
    std::uint64_t number_of_bins{};
    std::uint64_t waveform_offset{};
  };

  struct DevicePropagation {
    double minimum_height_m{};
    double maximum_height_m{};
    double step_m{};
    double inverse_step_per_m{};
    double slope_refractivity_lower{};
    double slope_integrated_refractivity_lower{};
    double slope_refractivity_upper{};
    double slope_integrated_refractivity_upper{};
    double const* refractivity{};
    double const* integrated_refractivity{};
    std::size_t table_size{};
    std::uint32_t zhs_subtrack_refinement{1};
  };

  struct DeviceWaveforms {
    double* floating_x{};
    double* floating_y{};
    double* floating_z{};
    long long* fixed_x{};
    long long* fixed_y{};
    long long* fixed_z{};
  };

  struct DeviceRadioCounters {
    unsigned long long coreas_contributions{};
    unsigned long long zhs_contributions{};
    unsigned long long zhs_subtracks{};
    unsigned long long valid_tracks{};
    unsigned long long fixed_point_overflows{};
    double weighted_segment_count{};
    double track_length_m{};
    double weighted_track_length_m{};
    double electron_weighted_track_length_m{};
    double positron_weighted_track_length_m{};
    double signed_charge_weighted_track_length_m{};
    double energy_weighted_track_length_GeV_m{};
    double maximum_segment_length_m{};
    double weighted_direction_change_rad{};
    double weighted_direction_change_squared_rad2{};
    double weighted_beta_deficit_track_length_m{};
    double weighted_time_residual_s{};
    double maximum_direction_change_rad{};
    double signed_charge_weighted_direction_change[3]{};
    double weighted_track_length_by_kinetic_energy_m[15]{};
  };

  struct Vec3 {
    double x{};
    double y{};
    double z{};
  };

  struct SignalPath {
    double propagation_time_s{};
    double refractive_index_source{};
    double refractive_index_destination{};
    double distance_m{};
    Vec3 emit{};
  };

  struct RadioTrackKinematics {
    Vec3 start{};
    Vec3 end{};
    Vec3 displacement{};
    Vec3 beta{};
    double start_time_s{};
    double end_time_s{};
    double duration_s{};
    double track_length_m{};
    double beta_module{};
    double constant{};
    std::uint32_t valid{};
  };

} // namespace corsika::accelerator::radio::detail
