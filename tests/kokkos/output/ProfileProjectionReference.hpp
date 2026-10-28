/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

// Frozen production 41cdc543. No receiver projection/crossing function is called.
#include <tests/kokkos/em/transport/TransportReference.hpp>
#include <tests/kokkos/em/selection/SelectionReference.hpp>
namespace profile_projection_reference {
  using namespace transport_reference;
  using selection_reference::isNativeMuonPairCut;
  using selection_reference::NativeMuonPairCutFlag;
  struct ProjectedEmStepRecord {
    std::uint64_t history_id{};
    std::int32_t pid{};
    std::int32_t process_id{};
    std::int32_t transport_limit{};
    std::int32_t reserved{};
    double start_grammage_g_per_cm2{};
    double end_grammage_g_per_cm2{};
    double end_energy_GeV{};
    double deposited_energy_GeV{};
    double weight{};
    double cut_deposited_energy_GeV{};
    std::uint32_t observation_surface_reached_before_cut{};
    std::uint32_t native_muon_pair_cut{};
  };
  struct DeviceProfileProjection {
    double axis_start_position_m[3]{};
    double axis_direction[3]{};
    double axis_step_length_m{};
    double const* axis_grammage_g_per_cm2{};
    std::size_t axis_support_count{};
  };
  CORSIKA_HOST_DEVICE inline double projectProfileGrammage(
      DeviceProfileProjection const& projection, double const position_m[3]) {
    auto projected_length_m = 0.;
    for (int axis = 0; axis < 3; ++axis)
      projected_length_m += (position_m[axis] - projection.axis_start_position_m[axis]) *
                            projection.axis_direction[axis];
    auto const fractional_bin = projected_length_m / projection.axis_step_length_m;
    if (fractional_bin < 0.) return projection.axis_grammage_g_per_cm2[0];
    auto const lower = static_cast<std::size_t>(fractional_bin);
    auto const upper = lower + 1;
    if (upper >= projection.axis_support_count)
      return projection.axis_grammage_g_per_cm2[projection.axis_support_count - 1];
    auto const fraction = fractional_bin - static_cast<double>(lower);
    return projection.axis_grammage_g_per_cm2[upper] * fraction +
           projection.axis_grammage_g_per_cm2[lower] * (1. - fraction);
  }
  CORSIKA_HOST_DEVICE inline ProjectedEmStepRecord projectPhotonStep(
      DeviceProfileProjection const& projection, PhotonTransportRecord const& record) {
    using namespace transport_reference;
    ProjectedEmStepRecord projected{};
    projected.history_id = record.start.history_id;
    projected.pid = record.start.pid;
    projected.process_id = record.limit == PhotonTransportLimit::Interaction
                               ? record.interaction.process_id
                               : 0;
    projected.transport_limit = static_cast<std::int32_t>(record.limit);
    projected.start_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.start.position_m);
    projected.end_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.end.position_m);
    projected.end_energy_GeV = record.end.energy_GeV;
    projected.deposited_energy_GeV = record.cut_deposited_energy_GeV;
    projected.cut_deposited_energy_GeV = record.cut_deposited_energy_GeV;
    projected.observation_surface_reached_before_cut =
        record.observation_surface_reached_before_cut;
    projected.weight = record.start.weight;
    return projected;
  }
  CORSIKA_HOST_DEVICE inline ProjectedEmStepRecord projectLeptonStep(
      DeviceProfileProjection const& projection, LeptonTransportRecord const& record) {
    using namespace transport_reference;
    ProjectedEmStepRecord projected{};
    projected.history_id = record.start.history_id;
    projected.pid = record.start.pid;
    projected.process_id =
        (record.limit == LeptonTransportLimit::InteractionCandidate ||
         record.interaction.status == EmInteractionStatus::AtRestAnnihilation)
            ? record.interaction.process_id
            : 0;
    projected.transport_limit = static_cast<std::int32_t>(record.limit);
    projected.reserved = static_cast<std::int32_t>(
        static_cast<std::uint32_t>(record.multiple_scattering_applied) |
        (static_cast<std::uint32_t>(record.multiple_scattering_status) << 1) |
        (record.multiple_scattering_iterations << 8));
    projected.native_muon_pair_cut =
        isNativeMuonPairCut(record.start) ? NativeMuonPairCutFlag : 0U;
    projected.start_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.start.position_m);
    projected.end_grammage_g_per_cm2 =
        projectProfileGrammage(projection, record.end.position_m);
    projected.end_energy_GeV = record.end.energy_GeV;
    projected.deposited_energy_GeV =
        record.continuous_deposited_energy_GeV + record.cut_deposited_energy_GeV;
    projected.cut_deposited_energy_GeV = record.cut_deposited_energy_GeV;
    projected.observation_surface_reached_before_cut =
        record.observation_surface_reached_before_cut;
    projected.weight = record.start.weight;
    return projected;
  }
  enum class ProfileCrossingMode { Both, Forward, OriginalC8 };
  struct LongitudinalCrossingBins {
    std::size_t begin = 0;
    std::size_t end = 0; // exclusive
  };
  CORSIKA_HOST_DEVICE inline LongitudinalCrossingBins longitudinalCrossingBins(
      double start, double end, std::size_t count) {
    constexpr double finite_limit = DBL_MAX;
    if (count == 0 || start == end ||
        !(start >= -finite_limit && start <= finite_limit) ||
        !(end >= -finite_limit && end <= finite_limit))
      return {};
    auto const low = start < end ? start : end;
    auto const high = start < end ? end : start;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    auto first = ::floor(low) + 1.;
    auto stop = ::floor(high) + 1.;
#else
    auto first = std::floor(low) + 1.;
    auto stop = std::floor(high) + 1.;
#endif
    auto const limit = static_cast<double>(count);
    first = first < 0. ? 0. : first;
    stop = stop > limit ? limit : stop;
    if (!(first < stop)) return {};
    return {static_cast<std::size_t>(first), static_cast<std::size_t>(stop)};
  }
  CORSIKA_HOST_DEVICE inline LongitudinalCrossingBins longitudinalProfileBins(
      double start, double end, std::size_t count, ProfileCrossingMode mode) {
    if (mode == ProfileCrossingMode::Both)
      return longitudinalCrossingBins(start, end, count);
    if (!(end > start)) return {};
    if (mode == ProfileCrossingMode::Forward)
      return longitudinalCrossingBins(start, end, count);
    if (mode != ProfileCrossingMode::OriginalC8 || count == 0 ||
        !(start >= -DBL_MAX && start <= DBL_MAX) || !(end >= -DBL_MAX && end <= DBL_MAX))
      return {};
      // Compatibility with original C8 ceil(start)..floor(end), inclusive.
      // Unlike half-open modes, an exactly shared endpoint can count twice.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    auto first = ::ceil(start);
    auto stop = ::floor(end) + 1.;
#else
    auto first = std::ceil(start);
    auto stop = std::floor(end) + 1.;
#endif
    first = first < 0. ? 0. : first;
    auto const limit = static_cast<double>(count);
    stop = stop > limit ? limit : stop;
    if (!(first < stop)) return {};
    return {static_cast<std::size_t>(first), static_cast<std::size_t>(stop)};
  }
} // namespace profile_projection_reference
