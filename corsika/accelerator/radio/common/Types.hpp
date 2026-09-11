/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace corsika::gpu::radio {

  struct RadioObserverSnapshot {
    double position_m[3]{};
    double start_time_s{};
    double duration_s{};
    double sample_rate_Hz{};
    std::uint64_t number_of_bins{};
  };

  /**
   * Host-owned copy of TabulatedFlatAtmospherePropagator's one-dimensional
   * refractivity model. Values intentionally preserve the scalar
   * implementation's nearest-bin lookup and endpoint extrapolation.
   */
  struct FlatAtmosphereRadioSnapshot {
    /** Zero retains the released atmospheric lookup. Positive values opt in
     * to exact homogeneous propagation (n R/c), with no altitude table.
     * The caller must ensure every entire ray stays in that homogeneous medium.
     */
    double homogeneous_refractive_index{};
    double minimum_height_m{};
    double maximum_height_m{};
    double step_m{};
    double inverse_step_per_m{};
    double slope_refractivity_lower{};
    double slope_integrated_refractivity_lower{};
    double slope_refractivity_upper{};
    double slope_integrated_refractivity_upper{};
    std::vector<double> refractivity{};
    std::vector<double> integrated_refractivity{};
  };

  struct GpuRadioConfig {
    bool enabled{};
    bool coreas_enabled{true};
    bool zhs_enabled{true};
    /**
     * Accumulate waveform samples as checked signed fixed-point integers.
     * Integer addition is associative and therefore independent of CUDA
     * warp/block scheduling. This is the production default.
     */
    bool deterministic{true};
    /**
     * Absolute electric-field range reserved by the fixed-point
     * accumulator. ZHS potential samples use the equivalent range after
     * multiplication by each observer's sample rate.
     */
    double fixed_point_field_limit_V_per_m{1.};
    /**
     * Collect scalar-compatible electron/positron track diagnostics on the
     * device. This adds global reductions to every transport batch and is
     * therefore disabled for production runs unless explicitly requested.
     */
    bool track_diagnostics{};
    /**
     * Split every stock ZHS Fraunhofer subtrack into this many equal pieces.
     * The default preserves the released scalar/CUDA projection exactly.
     * Values above one are a validation control: they refine radio projection
     * without changing the transported shower or consuming random numbers.
     */
    std::uint32_t zhs_subtrack_refinement{1};
    FlatAtmosphereRadioSnapshot propagation{};
    std::vector<RadioObserverSnapshot> coreas_observers{};
    std::vector<RadioObserverSnapshot> zhs_observers{};
  };

  struct RadioWaveform {
    std::vector<double> x{};
    std::vector<double> y{};
    std::vector<double> z{};
  };

  /**
   * CoREAS arrays contain electric field samples. ZHS arrays contain vector
   * potential samples; the existing RadioProcess::endOfShower performs the
   * same finite difference as on the scalar path after these arrays are
   * merged into TimeDomainObserver.
   */
  struct GpuRadioWaveforms {
    std::vector<RadioWaveform> coreas{};
    std::vector<RadioWaveform> zhs{};
  };

  struct GpuRadioStatistics {
    std::uint64_t lepton_tracks{};
    std::uint64_t track_observer_pairs{};
    std::uint64_t fused_track_observer_pairs{};
    std::uint64_t coreas_contributions{};
    std::uint64_t zhs_contributions{};
    std::uint64_t zhs_subtracks{};
    std::uint64_t fixed_point_overflows{};
    bool track_diagnostics_enabled{};
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
    std::array<double, 3>
        signed_charge_weighted_direction_change{};
    std::array<double, 15>
        weighted_track_length_by_kinetic_energy_m{};
    std::size_t device_bytes{};
    std::uint64_t host_to_device_bytes{};
    std::uint64_t device_to_host_bytes{};
    double device_time_ms{};
    double kernel_time_ms{};
    double transfer_time_ms{};
    std::uint64_t input_slot_waits{};
    double input_slot_host_wait_time_ms{};
    bool track_precompute_enabled{};
    std::uint32_t track_tile_size{};
    std::uint32_t observer_tile_size{};
    std::uint64_t track_precompute_batches{};
    std::uint64_t track_precomputed_records{};
    std::uint64_t direct_projection_batches{};
    std::uint64_t direct_projection_records{};
    std::uint64_t projection_tiles{};
    std::size_t track_workspace_bytes{};
    std::size_t maximum_track_batch{};
    double track_precompute_device_time_ms{};
    double projection_device_time_ms{};
  };

} // namespace corsika::gpu::radio
