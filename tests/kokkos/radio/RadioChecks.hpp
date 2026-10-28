/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "RadioProjectionDriver.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
namespace radio_test {
  inline void compare(Result<Current> const& a, Result<Reference> const& b) {
    REQUIRE(a.coreasFixed_ == b.coreasFixed_);
    REQUIRE(a.zhsFixed_ == b.zhsFixed_);
    REQUIRE(a.coreas_ == b.coreas_);
    REQUIRE(a.zhs_ == b.zhs_);
    REQUIRE(a.counters_.size() == b.counters_.size());
    for (std::size_t i = 0; i < a.counters_.size(); ++i) {
      REQUIRE(a.counters_[i].coreasContributions_ == b.counters_[i].coreas_contributions);
      REQUIRE(a.counters_[i].zhsContributions_ == b.counters_[i].zhs_contributions);
      REQUIRE(a.counters_[i].zhsSubtracks_ == b.counters_[i].zhs_subtracks);
      REQUIRE(a.counters_[i].validTracks_ == b.counters_[i].valid_tracks);
      REQUIRE(a.counters_[i].fixedPointOverflows_ ==
              b.counters_[i].fixed_point_overflows);
      REQUIRE(a.counters_[i].weightedSegmentCount_ ==
              b.counters_[i].weighted_segment_count);
      REQUIRE(a.counters_[i].trackLengthM_ == b.counters_[i].track_length_m);
      REQUIRE(a.counters_[i].weightedTrackLengthM_ ==
              b.counters_[i].weighted_track_length_m);
      REQUIRE(a.counters_[i].electronWeightedTrackLengthM_ ==
              b.counters_[i].electron_weighted_track_length_m);
      REQUIRE(a.counters_[i].positronWeightedTrackLengthM_ ==
              b.counters_[i].positron_weighted_track_length_m);
      REQUIRE(a.counters_[i].signedChargeWeightedTrackLengthM_ ==
              b.counters_[i].signed_charge_weighted_track_length_m);
      REQUIRE(a.counters_[i].energyWeightedTrackLengthGeVM_ ==
              b.counters_[i].energy_weighted_track_length_GeV_m);
      REQUIRE(a.counters_[i].maximumSegmentLengthM_ ==
              b.counters_[i].maximum_segment_length_m);
      REQUIRE(a.counters_[i].weightedDirectionChangeRad_ ==
              b.counters_[i].weighted_direction_change_rad);
      REQUIRE(a.counters_[i].weightedDirectionChangeSquaredRad2_ ==
              b.counters_[i].weighted_direction_change_squared_rad2);
      REQUIRE(a.counters_[i].weightedBetaDeficitTrackLengthM_ ==
              b.counters_[i].weighted_beta_deficit_track_length_m);
      REQUIRE(a.counters_[i].weightedTimeResidualS_ ==
              b.counters_[i].weighted_time_residual_s);
      REQUIRE(a.counters_[i].maximumDirectionChangeRad_ ==
              b.counters_[i].maximum_direction_change_rad);
      for (unsigned j = 0; j < 3; ++j)
        REQUIRE(a.counters_[i].signedChargeWeightedDirectionChange_[j] ==
                b.counters_[i].signed_charge_weighted_direction_change[j]);
      for (unsigned j = 0; j < 15; ++j)
        REQUIRE(a.counters_[i].weightedTrackLengthByKineticEnergyM_[j] ==
                b.counters_[i].weighted_track_length_by_kinetic_energy_m[j]);
    }
    for (std::size_t i = 0; i < a.tracks_.size(); ++i) {
      REQUIRE(a.tracks_[i].start_.x_ == b.tracks_[i].start.x);
      REQUIRE(a.tracks_[i].start_.y_ == b.tracks_[i].start.y);
      REQUIRE(a.tracks_[i].start_.z_ == b.tracks_[i].start.z);
      REQUIRE(a.tracks_[i].end_.x_ == b.tracks_[i].end.x);
      REQUIRE(a.tracks_[i].end_.y_ == b.tracks_[i].end.y);
      REQUIRE(a.tracks_[i].end_.z_ == b.tracks_[i].end.z);
      REQUIRE(a.tracks_[i].displacement_.x_ == b.tracks_[i].displacement.x);
      REQUIRE(a.tracks_[i].displacement_.y_ == b.tracks_[i].displacement.y);
      REQUIRE(a.tracks_[i].displacement_.z_ == b.tracks_[i].displacement.z);
      REQUIRE(a.tracks_[i].beta_.x_ == b.tracks_[i].beta.x);
      REQUIRE(a.tracks_[i].beta_.y_ == b.tracks_[i].beta.y);
      REQUIRE(a.tracks_[i].beta_.z_ == b.tracks_[i].beta.z);
      REQUIRE(a.tracks_[i].startTimeS_ == b.tracks_[i].start_time_s);
      REQUIRE(a.tracks_[i].endTimeS_ == b.tracks_[i].end_time_s);
      REQUIRE(a.tracks_[i].durationS_ == b.tracks_[i].duration_s);
      REQUIRE(a.tracks_[i].trackLengthM_ == b.tracks_[i].track_length_m);
      REQUIRE(a.tracks_[i].betaModule_ == b.tracks_[i].beta_module);
      REQUIRE(a.tracks_[i].constant_ == b.tracks_[i].constant);
      REQUIRE(a.tracks_[i].valid_ == b.tracks_[i].valid);
    }
  }
  static_assert(sizeof(cr::RadioObserverView) == sizeof(rr::DeviceObserver));
  static_assert(offsetof(cr::RadioObserverView, positionM_) ==
                offsetof(rr::DeviceObserver, position_m));
  static_assert(offsetof(cr::RadioObserverView, startTimeS_) ==
                offsetof(rr::DeviceObserver, start_time_s));
  static_assert(offsetof(cr::RadioObserverView, durationS_) ==
                offsetof(rr::DeviceObserver, duration_s));
  static_assert(offsetof(cr::RadioObserverView, sampleRateHz_) ==
                offsetof(rr::DeviceObserver, sample_rate_Hz));
  static_assert(offsetof(cr::RadioObserverView, fixedPointScale_) ==
                offsetof(rr::DeviceObserver, fixed_point_scale));
  static_assert(offsetof(cr::RadioObserverView, inverseFixedPointScale_) ==
                offsetof(rr::DeviceObserver, inverse_fixed_point_scale));
  static_assert(offsetof(cr::RadioObserverView, numberOfBins_) ==
                offsetof(rr::DeviceObserver, number_of_bins));
  static_assert(offsetof(cr::RadioObserverView, waveformOffset_) ==
                offsetof(rr::DeviceObserver, waveform_offset));
  static_assert(sizeof(cr::RadioPropagationView) == sizeof(rr::DevicePropagation));
  static_assert(offsetof(cr::RadioPropagationView, homogeneousRefractiveIndex_) ==
                offsetof(rr::DevicePropagation, homogeneous_refractive_index));
  static_assert(offsetof(cr::RadioPropagationView, minimumHeightM_) ==
                offsetof(rr::DevicePropagation, minimum_height_m));
  static_assert(offsetof(cr::RadioPropagationView, maximumHeightM_) ==
                offsetof(rr::DevicePropagation, maximum_height_m));
  static_assert(offsetof(cr::RadioPropagationView, stepM_) ==
                offsetof(rr::DevicePropagation, step_m));
  static_assert(offsetof(cr::RadioPropagationView, inverseStepPerM_) ==
                offsetof(rr::DevicePropagation, inverse_step_per_m));
  static_assert(offsetof(cr::RadioPropagationView, slopeRefractivityLower_) ==
                offsetof(rr::DevicePropagation, slope_refractivity_lower));
  static_assert(offsetof(cr::RadioPropagationView, slopeIntegratedRefractivityLower_) ==
                offsetof(rr::DevicePropagation, slope_integrated_refractivity_lower));
  static_assert(offsetof(cr::RadioPropagationView, slopeRefractivityUpper_) ==
                offsetof(rr::DevicePropagation, slope_refractivity_upper));
  static_assert(offsetof(cr::RadioPropagationView, slopeIntegratedRefractivityUpper_) ==
                offsetof(rr::DevicePropagation, slope_integrated_refractivity_upper));
  static_assert(offsetof(cr::RadioPropagationView, refractivity_) ==
                offsetof(rr::DevicePropagation, refractivity));
  static_assert(offsetof(cr::RadioPropagationView, integratedRefractivity_) ==
                offsetof(rr::DevicePropagation, integrated_refractivity));
  static_assert(offsetof(cr::RadioPropagationView, tableSize_) ==
                offsetof(rr::DevicePropagation, table_size));
  static_assert(offsetof(cr::RadioPropagationView, zhsSubtrackRefinement_) ==
                offsetof(rr::DevicePropagation, zhs_subtrack_refinement));
  static_assert(sizeof(cr::RadioWaveformView) == sizeof(rr::DeviceWaveforms));
  static_assert(offsetof(cr::RadioWaveformView, floatingX_) ==
                offsetof(rr::DeviceWaveforms, floating_x));
  static_assert(offsetof(cr::RadioWaveformView, floatingY_) ==
                offsetof(rr::DeviceWaveforms, floating_y));
  static_assert(offsetof(cr::RadioWaveformView, floatingZ_) ==
                offsetof(rr::DeviceWaveforms, floating_z));
  static_assert(offsetof(cr::RadioWaveformView, fixedX_) ==
                offsetof(rr::DeviceWaveforms, fixed_x));
  static_assert(offsetof(cr::RadioWaveformView, fixedY_) ==
                offsetof(rr::DeviceWaveforms, fixed_y));
  static_assert(offsetof(cr::RadioWaveformView, fixedZ_) ==
                offsetof(rr::DeviceWaveforms, fixed_z));
  static_assert(sizeof(cr::RadioCounters) == sizeof(rr::DeviceRadioCounters));
  static_assert(offsetof(cr::RadioCounters, coreasContributions_) ==
                offsetof(rr::DeviceRadioCounters, coreas_contributions));
  static_assert(offsetof(cr::RadioCounters, zhsContributions_) ==
                offsetof(rr::DeviceRadioCounters, zhs_contributions));
  static_assert(offsetof(cr::RadioCounters, zhsSubtracks_) ==
                offsetof(rr::DeviceRadioCounters, zhs_subtracks));
  static_assert(offsetof(cr::RadioCounters, validTracks_) ==
                offsetof(rr::DeviceRadioCounters, valid_tracks));
  static_assert(offsetof(cr::RadioCounters, fixedPointOverflows_) ==
                offsetof(rr::DeviceRadioCounters, fixed_point_overflows));
  static_assert(offsetof(cr::RadioCounters, weightedSegmentCount_) ==
                offsetof(rr::DeviceRadioCounters, weighted_segment_count));
  static_assert(offsetof(cr::RadioCounters, trackLengthM_) ==
                offsetof(rr::DeviceRadioCounters, track_length_m));
  static_assert(offsetof(cr::RadioCounters, weightedTrackLengthM_) ==
                offsetof(rr::DeviceRadioCounters, weighted_track_length_m));
  static_assert(offsetof(cr::RadioCounters, electronWeightedTrackLengthM_) ==
                offsetof(rr::DeviceRadioCounters, electron_weighted_track_length_m));
  static_assert(offsetof(cr::RadioCounters, positronWeightedTrackLengthM_) ==
                offsetof(rr::DeviceRadioCounters, positron_weighted_track_length_m));
  static_assert(offsetof(cr::RadioCounters, signedChargeWeightedTrackLengthM_) ==
                offsetof(rr::DeviceRadioCounters, signed_charge_weighted_track_length_m));
  static_assert(offsetof(cr::RadioCounters, energyWeightedTrackLengthGeVM_) ==
                offsetof(rr::DeviceRadioCounters, energy_weighted_track_length_GeV_m));
  static_assert(offsetof(cr::RadioCounters, maximumSegmentLengthM_) ==
                offsetof(rr::DeviceRadioCounters, maximum_segment_length_m));
  static_assert(offsetof(cr::RadioCounters, weightedDirectionChangeRad_) ==
                offsetof(rr::DeviceRadioCounters, weighted_direction_change_rad));
  static_assert(offsetof(cr::RadioCounters, weightedDirectionChangeSquaredRad2_) ==
                offsetof(rr::DeviceRadioCounters,
                         weighted_direction_change_squared_rad2));
  static_assert(offsetof(cr::RadioCounters, weightedBetaDeficitTrackLengthM_) ==
                offsetof(rr::DeviceRadioCounters, weighted_beta_deficit_track_length_m));
  static_assert(offsetof(cr::RadioCounters, weightedTimeResidualS_) ==
                offsetof(rr::DeviceRadioCounters, weighted_time_residual_s));
  static_assert(offsetof(cr::RadioCounters, maximumDirectionChangeRad_) ==
                offsetof(rr::DeviceRadioCounters, maximum_direction_change_rad));
  static_assert(offsetof(cr::RadioCounters, signedChargeWeightedDirectionChange_) ==
                offsetof(rr::DeviceRadioCounters,
                         signed_charge_weighted_direction_change));
  static_assert(offsetof(cr::RadioCounters, weightedTrackLengthByKineticEnergyM_) ==
                offsetof(rr::DeviceRadioCounters,
                         weighted_track_length_by_kinetic_energy_m));
  static_assert(sizeof(cr::RadioVector) == sizeof(rr::Vec3));
  static_assert(offsetof(cr::RadioVector, x_) == offsetof(rr::Vec3, x));
  static_assert(offsetof(cr::RadioVector, y_) == offsetof(rr::Vec3, y));
  static_assert(offsetof(cr::RadioVector, z_) == offsetof(rr::Vec3, z));
  static_assert(sizeof(cr::RadioSignalPath) == sizeof(rr::SignalPath));
  static_assert(offsetof(cr::RadioSignalPath, propagationTimeS_) ==
                offsetof(rr::SignalPath, propagation_time_s));
  static_assert(offsetof(cr::RadioSignalPath, refractiveIndexSource_) ==
                offsetof(rr::SignalPath, refractive_index_source));
  static_assert(offsetof(cr::RadioSignalPath, refractiveIndexDestination_) ==
                offsetof(rr::SignalPath, refractive_index_destination));
  static_assert(offsetof(cr::RadioSignalPath, distanceM_) ==
                offsetof(rr::SignalPath, distance_m));
  static_assert(offsetof(cr::RadioSignalPath, emit_) == offsetof(rr::SignalPath, emit));
  static_assert(sizeof(cr::RadioTrackKinematics) == sizeof(rr::RadioTrackKinematics));
  static_assert(offsetof(cr::RadioTrackKinematics, start_) ==
                offsetof(rr::RadioTrackKinematics, start));
  static_assert(offsetof(cr::RadioTrackKinematics, end_) ==
                offsetof(rr::RadioTrackKinematics, end));
  static_assert(offsetof(cr::RadioTrackKinematics, displacement_) ==
                offsetof(rr::RadioTrackKinematics, displacement));
  static_assert(offsetof(cr::RadioTrackKinematics, beta_) ==
                offsetof(rr::RadioTrackKinematics, beta));
  static_assert(offsetof(cr::RadioTrackKinematics, startTimeS_) ==
                offsetof(rr::RadioTrackKinematics, start_time_s));
  static_assert(offsetof(cr::RadioTrackKinematics, endTimeS_) ==
                offsetof(rr::RadioTrackKinematics, end_time_s));
  static_assert(offsetof(cr::RadioTrackKinematics, durationS_) ==
                offsetof(rr::RadioTrackKinematics, duration_s));
  static_assert(offsetof(cr::RadioTrackKinematics, trackLengthM_) ==
                offsetof(rr::RadioTrackKinematics, track_length_m));
  static_assert(offsetof(cr::RadioTrackKinematics, betaModule_) ==
                offsetof(rr::RadioTrackKinematics, beta_module));
  static_assert(offsetof(cr::RadioTrackKinematics, constant_) ==
                offsetof(rr::RadioTrackKinematics, constant));
  static_assert(offsetof(cr::RadioTrackKinematics, valid_) ==
                offsetof(rr::RadioTrackKinematics, valid));
} // namespace radio_test
