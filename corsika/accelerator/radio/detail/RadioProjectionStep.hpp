/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <corsika/accelerator/AcceleratorMacros.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionData.hpp>
#include <corsika/gpu/em/Types.hpp>

namespace corsika::accelerator::radio::detail {

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioAbs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioSqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(value);
#else
    return std::sqrt(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioFloor(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::floor(value);
#else
    return std::floor(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioCeil(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::ceil(value);
#else
    return std::ceil(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioAcos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::acos(value);
#else
    return std::acos(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool radioFinite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline long long radioRound(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return static_cast<long long>(::rint(value));
#else
    return static_cast<long long>(std::rint(value));
#endif
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioMin(double left,
                                                         double right) {
    return left < right ? left : right;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double radioMax(double left,
                                                         double right) {
    return left > right ? left : right;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Vec3 operator*(Vec3 a, double value) {
    return {a.x * value, a.y * value, a.z * value};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Vec3 operator/(Vec3 a, double value) {
    return {a.x / value, a.y / value, a.z / value};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline double norm(Vec3 value) {
    return radioSqrt(dot(value, value));
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool makeRadioTrackKinematics(
      gpu::em::LeptonTransportRecord const& record,
      RadioTrackKinematics& track) {
    if (!gpu::em::isElectronOrPositronPid(record.start.pid)) return false;
    track.start = {record.start.position_m[0], record.start.position_m[1],
                   record.start.position_m[2]};
    track.end = {record.end.position_m[0], record.end.position_m[1],
                 record.end.position_m[2]};
    track.start_time_s = record.start.time_s;
    track.end_time_s = record.end.time_s;
    track.duration_s = track.end_time_s - track.start_time_s;
    track.displacement = track.end - track.start;
    track.track_length_m = norm(track.displacement);
    if (!(track.duration_s > 0.) || !(track.track_length_m > 0.)) return false;
    track.beta =
        track.displacement / (SpeedOfLightMPerS * track.duration_s);
    track.beta_module =
        track.track_length_m / (SpeedOfLightMPerS * track.duration_s);
    auto const charge =
        record.start.pid == static_cast<std::int32_t>(gpu::em::EmPid::Electron)
            ? -ElementaryChargeC
            : ElementaryChargeC;
    track.constant = charge * EmConstant * record.start.weight;
    track.valid = 1;
    return true;
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline SignalPath propagate(
      DevicePropagation const& table, Vec3 source,
      DeviceObserver const& observer) {
    auto const destination = Vec3{observer.position_m[0], observer.position_m[1],
                                  observer.position_m[2]};
    auto const displacement = destination - source;
    auto const distance = norm(displacement);
    auto const emit = displacement / distance;
    auto const source_height =
        (source.z - table.minimum_height_m) * table.inverse_step_per_m;
    auto const destination_height =
        (destination.z - table.minimum_height_m) * table.inverse_step_per_m;
    auto const last = table.table_size - 1;
    auto const destination_index =
        static_cast<std::size_t>(destination_height + 0.5);

    double refractive_index_source = 1.;
    double integrated_source = 1.;
    auto const refractive_index_destination =
        table.refractivity[destination_index] + 1.;
    auto const integrated_destination =
        table.integrated_refractivity[destination_index];
    double height = 1.;

    if (source_height + 0.5 >= static_cast<double>(last)) {
      refractive_index_source =
          table.refractivity[last] +
          table.slope_refractivity_upper *
              radioAbs((source.z - table.maximum_height_m) *
                       table.inverse_step_per_m) +
          1.;
      integrated_source =
          table.integrated_refractivity[last] +
          table.slope_integrated_refractivity_upper *
              radioAbs(table.maximum_height_m - source.z);
      height = source_height - destination_height;
    } else if (source_height + 0.5 < static_cast<double>(last) &&
               source_height > 0.) {
      auto const source_index =
          static_cast<std::size_t>(source_height + 0.5);
      refractive_index_source = table.refractivity[source_index] + 1.;
      integrated_source = table.integrated_refractivity[source_index];
      height =
          static_cast<double>(static_cast<std::int64_t>(source_index) -
                              static_cast<std::int64_t>(destination_index)) *
          table.step_m;
      if (height == 0.) height = 1.;
    } else if (source_height == 0.) {
      refractive_index_source = table.refractivity[0] + 1.;
      integrated_source = table.integrated_refractivity[0];
      height = destination_height - source_height;
    } else {
      refractive_index_source =
          table.refractivity[0] +
          table.slope_refractivity_lower * radioAbs(source_height) + 1.;
      integrated_source =
          table.integrated_refractivity[0] +
          table.slope_integrated_refractivity_lower * radioAbs(source_height);
      height = destination_height - radioAbs(source_height);
    }

    auto const propagation_time =
        (1. + (integrated_source - integrated_destination) / height) *
        distance / SpeedOfLightMPerS;
    return {propagation_time, refractive_index_source,
            refractive_index_destination, distance, emit};
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline Vec3 transverseEndpoint(Vec3 emit,
                                                                 Vec3 beta) {
    return cross(emit, cross(emit, beta));
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline bool checkedRadioAtomicAdd(
      long long* address, long long increment) {
    auto const previous = AtomicOperations::add(address, increment);
    return !((increment > 0 &&
              previous > SignedIntegerMaximum - increment) ||
             (increment < 0 &&
              previous < SignedIntegerMinimum - increment));
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline bool addRadioFixedPoint(
      long long* address, double contribution, double scale) {
    auto const scaled = contribution * scale;
    if (!radioFinite(scaled) || radioAbs(scaled) >= SignedIntegerLimit)
      return false;
    return checkedRadioAtomicAdd<AtomicOperations>(address,
                                                    radioRound(scaled));
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void addSample(
      DeviceObserver const& observer, DeviceWaveforms const& waveforms,
      double time_s, Vec3 contribution,
      unsigned long long* contribution_counter,
      DeviceRadioCounters* counters) {
    if (time_s < observer.start_time_s ||
        time_s > observer.start_time_s + observer.duration_s)
      return;
    auto const bin_value =
        radioFloor((time_s - observer.start_time_s) * observer.sample_rate_Hz +
                   0.5);
    if (!(bin_value >= 0.)) return;
    auto const bin = static_cast<std::uint64_t>(bin_value);
    if (bin >= observer.number_of_bins) return;
    auto const offset = observer.waveform_offset + bin;
    if (waveforms.fixed_x != nullptr) {
      auto const valid_x = addRadioFixedPoint<AtomicOperations>(
          waveforms.fixed_x + offset, contribution.x,
          observer.fixed_point_scale);
      auto const valid_y = addRadioFixedPoint<AtomicOperations>(
          waveforms.fixed_y + offset, contribution.y,
          observer.fixed_point_scale);
      auto const valid_z = addRadioFixedPoint<AtomicOperations>(
          waveforms.fixed_z + offset, contribution.z,
          observer.fixed_point_scale);
      if (!(valid_x && valid_y && valid_z)) {
        AtomicOperations::add(&counters->fixed_point_overflows, 1ULL);
        return;
      }
    } else {
      AtomicOperations::add(waveforms.floating_x + offset, contribution.x);
      AtomicOperations::add(waveforms.floating_y + offset, contribution.y);
      AtomicOperations::add(waveforms.floating_z + offset, contribution.z);
    }
    AtomicOperations::add(contribution_counter, 1ULL);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline void separateSameBinEndpoints(
      double& start_time, double& end_time, double grid_resolution,
      bool signed_order) {
    auto const start_bin = static_cast<long long>(
        radioFloor(start_time / grid_resolution + 0.5));
    auto const end_bin = static_cast<long long>(
        radioFloor(end_time / grid_resolution + 0.5));
    if (start_bin != end_bin) return;
    auto const start_fraction =
        start_time / grid_resolution - radioFloor(start_time / grid_resolution);
    auto const end_fraction =
        end_time / grid_resolution - radioFloor(end_time / grid_resolution);
    auto const forward = !signed_order || end_time >= start_time;
    if (forward) {
      if (start_fraction >= 0.5 && end_fraction >= 0.5)
        start_time -= grid_resolution;
      else if (start_fraction < 0.5 && end_fraction < 0.5)
        end_time += grid_resolution;
      else if (end_fraction >= 1. - start_fraction)
        end_time += grid_resolution;
      else
        start_time -= grid_resolution;
    } else {
      if (start_fraction >= 0.5 && end_fraction >= 0.5)
        end_time -= grid_resolution;
      else if (start_fraction < 0.5 && end_fraction < 0.5)
        start_time += grid_resolution;
      else if (start_fraction >= 1. - end_fraction)
        start_time += grid_resolution;
      else
        end_time -= grid_resolution;
    }
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateCoREAS(
      RadioTrackKinematics const& track, DevicePropagation const& propagation,
      DeviceObserver const& observer, DeviceWaveforms const& waveforms,
      DeviceRadioCounters* counters) {
    auto const path_start = propagate(propagation, track.start, observer);
    auto const path_end = propagate(propagation, track.end, observer);
    auto const pre_doppler =
        1. - path_start.refractive_index_source *
                 dot(track.beta, path_start.emit);
    auto const post_doppler =
        1. - path_end.refractive_index_source * dot(track.beta, path_end.emit);
    auto start_receive = track.start_time_s + path_start.propagation_time_s;
    auto end_receive = track.end_time_s + path_end.propagation_time_s;

    if (path_start.refractive_index_destination > 1. &&
        (radioAbs(pre_doppler) < CoREASApproximationThreshold ||
         radioAbs(post_doppler) < CoREASApproximationThreshold)) {
      auto const midpoint = (track.start + track.end) * 0.5;
      auto const middle_time = (track.start_time_s + track.end_time_s) * 0.5;
      auto const path_middle = propagate(propagation, midpoint, observer);
      auto const middle_receive = middle_time + path_middle.propagation_time_s;
      auto const middle_doppler =
          1. - path_middle.refractive_index_source *
                   dot(track.beta, path_middle.emit);
      auto field_start = transverseEndpoint(path_middle.emit, track.beta) *
                         (track.constant * observer.sample_rate_Hz /
                          (middle_doppler * path_middle.distance_m));
      auto field_end = field_start * -1.;
      auto delta_time = track.track_length_m /
                        (SpeedOfLightMPerS * track.beta_module) *
                        radioAbs(middle_doppler);
      if (start_receive < end_receive) {
        start_receive = middle_receive - 0.5 * delta_time;
        end_receive = middle_receive + 0.5 * delta_time;
      } else {
        start_receive = middle_receive + 0.5 * delta_time;
        end_receive = middle_receive - 0.5 * delta_time;
      }
      auto const grid_resolution = 1. / observer.sample_rate_Hz;
      delta_time = end_receive - start_receive;
      if (radioAbs(delta_time) < grid_resolution) {
        auto const scale = radioAbs(delta_time / grid_resolution);
        field_start = field_start * scale;
        field_end = field_end * scale;
        separateSameBinEndpoints(start_receive, end_receive, grid_resolution,
                                 true);
      }
      addSample<AtomicOperations>(
          observer, waveforms, start_receive, field_start,
          &counters->coreas_contributions, counters);
      addSample<AtomicOperations>(observer, waveforms, end_receive, field_end,
                                  &counters->coreas_contributions, counters);
      return;
    }

    auto field_start = transverseEndpoint(path_start.emit, track.beta) *
                       (track.constant * observer.sample_rate_Hz /
                        (pre_doppler * path_start.distance_m));
    auto field_end = transverseEndpoint(path_end.emit, track.beta) *
                     (-track.constant * observer.sample_rate_Hz /
                      (post_doppler * path_end.distance_m));
    if (pre_doppler < 1.e-9 || post_doppler < 1.e-9) {
      auto const grid_resolution = 1. / observer.sample_rate_Hz;
      auto const delta_time = end_receive - start_receive;
      if (radioAbs(delta_time) < grid_resolution) {
        auto const scale = radioAbs(delta_time / grid_resolution);
        field_start = field_start * scale;
        field_end = field_end * scale;
        separateSameBinEndpoints(start_receive, end_receive, grid_resolution,
                                 false);
      }
    }
    addSample<AtomicOperations>(observer, waveforms, start_receive,
                                field_start,
                                &counters->coreas_contributions, counters);
    addSample<AtomicOperations>(observer, waveforms, end_receive, field_end,
                                &counters->coreas_contributions, counters);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateZhsSegment(
      Vec3 point1, Vec3 point2, double time1, double time2, Vec3 beta,
      double constant, DevicePropagation const& propagation,
      DeviceObserver const& observer, DeviceWaveforms const& waveforms,
      DeviceRadioCounters* counters, bool subdivided) {
    auto const midpoint = (point1 + point2) * 0.5;
    auto const path = propagate(propagation, midpoint, observer);
    auto const n_source = path.refractive_index_source;
    auto const beta_times_k = dot(beta, path.emit);
    auto const middle_time = (time1 + time2) * 0.5;
    auto detection_time1 =
        time1 + path.propagation_time_s -
        n_source * beta_times_k * (time1 - middle_time);
    auto detection_time2 =
        time2 + path.propagation_time_s -
        n_source * beta_times_k * (time2 - middle_time);
    auto sign = 1.;
    if (detection_time1 > detection_time2) {
      detection_time1 =
          time2 + path.propagation_time_s -
          n_source * beta_times_k * (time2 - middle_time);
      detection_time2 =
          time1 + path.propagation_time_s -
          n_source * beta_times_k * (time1 - middle_time);
      sign = -1.;
    }
    auto const start_bin = radioFloor(
        (detection_time1 - observer.start_time_s) * observer.sample_rate_Hz +
        0.5);
    auto const end_bin = radioFloor(
        (detection_time2 - observer.start_time_s) * observer.sample_rate_Hz +
        0.5);
    auto const beta_perpendicular = cross(path.emit, cross(beta, path.emit));
    auto const denominator = 1. - n_source * beta_times_k;

    if (start_bin == end_bin) {
      Vec3 potential{};
      if (radioAbs(denominator) > 1.e-15) {
        auto const fraction = radioAbs(
            detection_time2 * observer.sample_rate_Hz -
            detection_time1 * observer.sample_rate_Hz);
        potential = beta_perpendicular *
                    (sign * constant * fraction /
                     (denominator * path.distance_m));
      } else {
        auto const fraction = (time2 - time1) * observer.sample_rate_Hz;
        potential = beta_perpendicular *
                    (sign * constant * fraction / path.distance_m);
      }
      addSample<AtomicOperations>(observer, waveforms, detection_time2,
                                  potential, &counters->zhs_contributions,
                                  counters);
      return;
    }

    auto const number_of_bins = static_cast<int>(end_bin - start_bin);
    auto fraction = radioAbs(
        start_bin + 0.5 -
        (detection_time1 - observer.start_time_s) * observer.sample_rate_Hz);
    auto potential = beta_perpendicular *
                     (sign * constant * fraction /
                      (denominator * path.distance_m));
    addSample<AtomicOperations>(observer, waveforms, detection_time1,
                                potential, &counters->zhs_contributions,
                                counters);
    for (int index = 1; index < number_of_bins; ++index) {
      potential = beta_perpendicular *
                  ((subdivided ? 1. : sign) * constant /
                   (denominator * path.distance_m));
      addSample<AtomicOperations>(
          observer, waveforms,
          detection_time1 +
              static_cast<double>(index) / observer.sample_rate_Hz,
          potential, &counters->zhs_contributions, counters);
    }
    fraction = radioAbs(
        (detection_time2 - observer.start_time_s) * observer.sample_rate_Hz +
        0.5 - end_bin);
    potential = beta_perpendicular *
                (sign * constant * fraction /
                 (denominator * path.distance_m));
    addSample<AtomicOperations>(observer, waveforms, detection_time2,
                                potential, &counters->zhs_contributions,
                                counters);
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline void accumulateZHS(
      RadioTrackKinematics const& track, DevicePropagation const& propagation,
      DeviceObserver const& observer, DeviceWaveforms const& waveforms,
      DeviceRadioCounters* counters) {
    auto const midpoint = (track.start + track.end) * 0.5;
    auto const middle_path = propagate(propagation, midpoint, observer);
    auto const u_times_k = dot(track.beta, middle_path.emit) /
                           track.beta_module;
    auto const sin_theta_squared = 1. - u_times_k * u_times_k;
    auto const wavelength = SpeedOfLightMPerS / observer.sample_rate_Hz;
    auto const fraunhofer = sin_theta_squared * track.track_length_m *
                            track.track_length_m / middle_path.distance_m /
                            wavelength * 2. * Pi;
    auto subtrack_divisor = 1.;
    auto number_of_subtracks = 1;
    if (fraunhofer > 1.) {
      // Preserve the released scalar beta2 semantics: the loop bound is the
      // ceil of this value, while step sizes use the unrounded double.
      subtrack_divisor = radioSqrt(fraunhofer) + 1.;
      number_of_subtracks = static_cast<int>(radioCeil(subtrack_divisor));
    }
    AtomicOperations::add(
        &counters->zhs_subtracks,
        static_cast<unsigned long long>(number_of_subtracks));
    auto const spatial_step = track.displacement / subtrack_divisor;
    auto const time_step = track.duration_s / subtrack_divisor;
    auto const refinement =
        static_cast<int>(propagation.zhs_subtrack_refinement);
    auto const refined_spatial_step =
        spatial_step / static_cast<double>(refinement);
    auto const refined_time_step = time_step / static_cast<double>(refinement);
    auto point1 = track.start;
    auto time1 = track.start_time_s;
    for (int index = 0; index < number_of_subtracks; ++index) {
      for (int refined_index = 0; refined_index < refinement;
           ++refined_index) {
        auto const point2 = point1 + refined_spatial_step;
        auto const time2 = time1 + refined_time_step;
        accumulateZhsSegment<AtomicOperations>(
            point1, point2, time1, time2, track.beta, track.constant,
            propagation, observer, waveforms, counters,
            number_of_subtracks > 1 || refinement > 1);
        point1 = point2;
        time1 = time2;
      }
    }
    if (refinement > 1) {
      AtomicOperations::add(
          &counters->zhs_subtracks,
          static_cast<unsigned long long>(number_of_subtracks *
                                          (refinement - 1)));
    }
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline std::size_t diagnosticEnergyBin(
      double kinetic_energy_GeV) {
    constexpr double upper_edges[14]{
        1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2, 1.e-1,
        2.e-1, 5.e-1, 1.,   2.,    5.,    10.,   100.};
    std::size_t index = 0;
    while (index < 14 && kinetic_energy_GeV > upper_edges[index]) ++index;
    return index;
  }

  template <class AtomicOperations>
  C8_ACCELERATOR_INLINE_FUNCTION inline bool recordTrack(
      gpu::em::LeptonTransportRecord const& record,
      RadioTrackKinematics* output, DeviceRadioCounters* counters,
      bool collect_diagnostics) {
    RadioTrackKinematics track{};
    if (!makeRadioTrackKinematics(record, track)) {
      if (output != nullptr) *output = {};
      return false;
    }
    if (output != nullptr) *output = track;
    AtomicOperations::add(&counters->valid_tracks, 1ULL);
    if (!collect_diagnostics) return true;

    auto const length_m = track.track_length_m;
    auto const weight = record.start.weight;
    auto const kinetic_energy_GeV =
        record.start.energy_GeV - gpu::em::ElectronMassGeV;
    if (!radioFinite(length_m) || !(length_m > 0.) || !radioFinite(weight) ||
        weight < 0. || !radioFinite(kinetic_energy_GeV) ||
        kinetic_energy_GeV < 0.)
      return true;

    auto const weighted_length_m = weight * length_m;
    AtomicOperations::add(&counters->weighted_segment_count, weight);
    AtomicOperations::add(&counters->track_length_m, length_m);
    AtomicOperations::add(&counters->weighted_track_length_m,
                          weighted_length_m);
    AtomicOperations::add(&counters->energy_weighted_track_length_GeV_m,
                          weighted_length_m * kinetic_energy_GeV);
    AtomicOperations::maximum(&counters->maximum_segment_length_m, length_m);
    if (record.start.pid ==
        static_cast<std::int32_t>(gpu::em::EmPid::Electron)) {
      AtomicOperations::add(&counters->electron_weighted_track_length_m,
                            weighted_length_m);
      AtomicOperations::add(&counters->signed_charge_weighted_track_length_m,
                            -weighted_length_m);
    } else {
      AtomicOperations::add(&counters->positron_weighted_track_length_m,
                            weighted_length_m);
      AtomicOperations::add(&counters->signed_charge_weighted_track_length_m,
                            weighted_length_m);
    }
    AtomicOperations::add(
        &counters->weighted_track_length_by_kinetic_energy_m
             [diagnosticEnergyBin(kinetic_energy_GeV)],
        weighted_length_m);

    auto direction_dot = 0.;
    double direction_delta[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      auto const pre = record.start.direction[axis];
      auto const post = record.end.direction[axis];
      direction_dot += pre * post;
      direction_delta[axis] = post - pre;
    }
    auto const direction_change_rad = radioAcos(
        radioMin(1., radioMax(-1., direction_dot)));
    auto const time_residual_s =
        track.duration_s - length_m / SpeedOfLightMPerS;
    AtomicOperations::add(&counters->weighted_direction_change_rad,
                          weight * direction_change_rad);
    AtomicOperations::add(&counters->weighted_direction_change_squared_rad2,
                          weight * direction_change_rad *
                              direction_change_rad);
    AtomicOperations::add(&counters->weighted_beta_deficit_track_length_m,
                          weighted_length_m * (1. - track.beta_module));
    AtomicOperations::add(&counters->weighted_time_residual_s,
                          weight * time_residual_s);
    AtomicOperations::maximum(&counters->maximum_direction_change_rad,
                              direction_change_rad);
    auto const charge_sign =
        record.start.pid == static_cast<std::int32_t>(gpu::em::EmPid::Electron)
            ? -1.
            : 1.;
    for (int axis = 0; axis < 3; ++axis)
      AtomicOperations::add(
          &counters->signed_charge_weighted_direction_change[axis],
          charge_sign * weight * direction_delta[axis]);
    return true;
  }

} // namespace corsika::accelerator::radio::detail
