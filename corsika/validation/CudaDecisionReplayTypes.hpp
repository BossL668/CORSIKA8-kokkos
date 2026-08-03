/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

/*
 * POD-only disk/device types for scalar-to-CUDA decision replay.
 */
#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace corsika::validation {

  inline constexpr std::array<char, 8> CudaReplayTapeMagic{
      'C', '8', 'R', 'P', 'T', '0', '0', '1'};
  inline constexpr std::uint32_t CudaReplayTapeVersion = 1;
  inline constexpr std::uint64_t CudaReplayEndianMarker =
      0x0102030405060708ULL;

  enum class ReplayTransportLimit : std::int32_t {
    Continuous = 0,
    Geometry = 1,
    Interaction = 2,
    Decay = 3,
    ForcedInteraction = 4,
    ForcedDecay = 5,
  };

  struct ReplayRadioObserver {
    double position_m[3]{};
    double start_time_s{};
    double duration_s{};
    double sample_rate_Hz{};
    double axis_start_ns{};
    std::uint64_t number_of_bins{};
  };

  struct ReplayPropagationSnapshot {
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

  struct ReplayRadioSnapshot {
    ReplayPropagationSnapshot propagation{};
    std::vector<ReplayRadioObserver> coreas_observers{};
    std::vector<ReplayRadioObserver> zhs_observers{};
  };

  struct ReplayTapeHeader {
    char magic[8]{};
    std::uint32_t format_version{};
    std::uint32_t header_bytes{};
    std::uint64_t endian_marker{};
    std::uint64_t record_bytes{};
    std::uint64_t record_count{};
    std::uint64_t configured_showers{};
    std::uint64_t random_seed{};
    std::uint64_t refractivity_count{};
    std::uint64_t integrated_refractivity_count{};
    std::uint64_t coreas_observer_count{};
    std::uint64_t zhs_observer_count{};
    double minimum_height_m{};
    double maximum_height_m{};
    double propagation_step_m{};
    double inverse_propagation_step_per_m{};
    double slope_refractivity_lower{};
    double slope_integrated_refractivity_lower{};
    double slope_refractivity_upper{};
    double slope_integrated_refractivity_upper{};
  };

  struct ReplayTransportStep {
    std::uint64_t shower{};
    std::uint64_t ordinal{};
    std::uint64_t stack_index{};
    std::int32_t pdg{};
    ReplayTransportLimit transport_limit{ReplayTransportLimit::Continuous};
    std::int32_t process_return{};
    std::int32_t reserved{};
    double start_position_m[3]{};
    double end_position_m[3]{};
    double start_direction[3]{};
    double end_direction[3]{};
    double start_time_s{};
    double end_time_s{};
    double start_total_energy_GeV{};
    double end_total_energy_GeV{};
    double weight{};
  };

  struct ReplayTapeData {
    ReplayTapeHeader header{};
    ReplayRadioSnapshot radio{};
    std::vector<ReplayTransportStep> steps{};
  };

  static_assert(std::is_trivially_copyable_v<ReplayTapeHeader>);
  static_assert(std::is_trivially_copyable_v<ReplayRadioObserver>);
  static_assert(std::is_trivially_copyable_v<ReplayTransportStep>);

} // namespace corsika::validation
