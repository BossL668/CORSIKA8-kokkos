/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <corsika/gpu/em/PhotonPairKinematics.hpp>
#include <corsika/gpu/radio/CudaRadioAccumulator.hpp>

namespace corsika::gpu::radio {

  namespace {

    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr std::size_t RadioInputSlotCount = 2;
    constexpr double SpeedOfLightMPerS = 299792458.;
    constexpr double VacuumPermittivityFPerM =
        8.8541878128e-12;
    constexpr double ElementaryChargeC =
        // Match corsika::constants::e exactly. The core project currently
        // uses the pre-2019 CODATA value and scalar radio output is the
        // validation authority for this backend.
        1.6021766208e-19;
    constexpr double Pi =
        3.141592653589793238462643383279502884;
    constexpr double EmConstant =
        1. / (4. * Pi * VacuumPermittivityFPerM *
              SpeedOfLightMPerS);
    constexpr double CoREASApproximationThreshold = 1.e-3;
    // Reserve one sign bit and one headroom bit. This leaves a symmetric
    // range of approximately +/-2*fixed_point_field_limit while retaining
    // 62 fractional bits at the configured limit.
    constexpr double FixedPointHeadroom = 0x1p62;
    constexpr double SignedIntegerLimit = 0x1p63;
    constexpr long long SignedIntegerMaximum =
        9223372036854775807LL;
    constexpr long long SignedIntegerMinimum =
        -SignedIntegerMaximum - 1LL;

    void checkCuda(cudaError_t status, char const* operation) {
      if (status == cudaSuccess) {
        return;
      }
      std::ostringstream message;
      message << operation << " failed: "
              << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }

    std::size_t checkedAdd(std::size_t left, std::size_t right,
                           char const* message) {
      if (right >
          std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(message);
      }
      return left + right;
    }

    std::size_t checkedMultiply(std::size_t left,
                                std::size_t right,
                                char const* message) {
      if (left != 0 &&
          right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(message);
      }
      return left * right;
    }

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

    struct RadioInputSlot {
      cudaEvent_t input_ready{};
      cudaEvent_t radio_start{};
      cudaEvent_t radio_done{};
      bool active{};
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
    };

    __host__ __device__ Vec3 operator+(Vec3 a, Vec3 b) {
      return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    __host__ __device__ Vec3 operator-(Vec3 a, Vec3 b) {
      return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    __host__ __device__ Vec3 operator*(Vec3 a, double value) {
      return {a.x * value, a.y * value, a.z * value};
    }

    __host__ __device__ Vec3 operator/(Vec3 a, double value) {
      return {a.x / value, a.y / value, a.z / value};
    }

    __host__ __device__ double dot(Vec3 a, Vec3 b) {
      return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
      return {a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
    }

    __host__ __device__ double norm(Vec3 value) {
      return sqrt(dot(value, value));
    }

    __device__ bool makeRadioTrackKinematics(
        em::LeptonTransportRecord const& record,
        RadioTrackKinematics& track) {
      if (!em::isElectronOrPositronPid(record.start.pid)) {
        return false;
      }
      track.start = {
          record.start.position_m[0],
          record.start.position_m[1],
          record.start.position_m[2]};
      track.end = {
          record.end.position_m[0],
          record.end.position_m[1],
          record.end.position_m[2]};
      track.start_time_s = record.start.time_s;
      track.end_time_s = record.end.time_s;
      track.duration_s =
          track.end_time_s - track.start_time_s;
      track.displacement = track.end - track.start;
      track.track_length_m = norm(track.displacement);
      if (!(track.duration_s > 0.) ||
          !(track.track_length_m > 0.)) {
        return false;
      }
      track.beta =
          track.displacement /
          (SpeedOfLightMPerS * track.duration_s);
      track.beta_module =
          track.track_length_m /
          (SpeedOfLightMPerS * track.duration_s);
      auto const charge =
          record.start.pid ==
                  static_cast<std::int32_t>(
                      em::EmPid::Electron)
              ? -ElementaryChargeC
              : ElementaryChargeC;
      track.constant =
          charge * EmConstant * record.start.weight;
      return true;
    }

    __device__ SignalPath propagate(
        DevicePropagation const& table, Vec3 source,
        DeviceObserver const& observer) {
      auto const destination =
          Vec3{observer.position_m[0], observer.position_m[1],
               observer.position_m[2]};
      auto const displacement = destination - source;
      auto const distance = norm(displacement);
      auto const emit = displacement / distance;
      auto const source_height =
          (source.z - table.minimum_height_m) *
          table.inverse_step_per_m;
      auto const destination_height =
          (destination.z - table.minimum_height_m) *
          table.inverse_step_per_m;
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

      if (source_height + 0.5 >=
          static_cast<double>(last)) {
        refractive_index_source =
            table.refractivity[last] +
            table.slope_refractivity_upper *
                fabs((source.z - table.maximum_height_m) *
                     table.inverse_step_per_m) +
            1.;
        integrated_source =
            table.integrated_refractivity[last] +
            table.slope_integrated_refractivity_upper *
                fabs(table.maximum_height_m - source.z);
        height = source_height - destination_height;
      } else if (source_height + 0.5 <
                     static_cast<double>(last) &&
                 source_height > 0.) {
        auto const source_index =
            static_cast<std::size_t>(source_height + 0.5);
        refractive_index_source =
            table.refractivity[source_index] + 1.;
        integrated_source =
            table.integrated_refractivity[source_index];
        height =
            static_cast<double>(
                static_cast<std::int64_t>(source_index) -
                static_cast<std::int64_t>(destination_index)) *
            table.step_m;
        if (height == 0.) {
          height = 1.;
        }
      } else if (source_height == 0.) {
        refractive_index_source =
            table.refractivity[0] + 1.;
        integrated_source =
            table.integrated_refractivity[0];
        height = destination_height - source_height;
      } else {
        refractive_index_source =
            table.refractivity[0] +
            table.slope_refractivity_lower *
                fabs(source_height) +
            1.;
        integrated_source =
            table.integrated_refractivity[0] +
            table.slope_integrated_refractivity_lower *
                fabs(source_height);
        height = destination_height - fabs(source_height);
      }

      auto const propagation_time =
          (1. +
           (integrated_source - integrated_destination) /
               height) *
          distance / SpeedOfLightMPerS;
      return {propagation_time, refractive_index_source,
              refractive_index_destination, distance, emit};
    }

    __device__ Vec3 transverseEndpoint(Vec3 emit, Vec3 beta) {
      return cross(emit, cross(emit, beta));
    }

    __device__ bool checkedAtomicAdd(
        long long* address, long long increment) {
      auto* storage =
          reinterpret_cast<unsigned long long*>(address);
      // One hardware atomic addition is sufficient.  Its returned old value
      // is the serialization point used for the signed-overflow check.
      // Should an overflow occur the shower is rejected after the counters
      // are downloaded, so the wrapped destination is never accepted.
      auto const previous_bits = atomicAdd(
          storage, static_cast<unsigned long long>(increment));
      auto const previous =
          static_cast<long long>(previous_bits);
      return !(
          (increment > 0 &&
           previous > SignedIntegerMaximum - increment) ||
          (increment < 0 &&
           previous < SignedIntegerMinimum - increment));
    }

    __device__ bool addFixedPoint(
        long long* address, double contribution, double scale) {
      auto const scaled = contribution * scale;
      if (!isfinite(scaled) || fabs(scaled) >= SignedIntegerLimit) {
        return false;
      }
      return checkedAtomicAdd(
          address, __double2ll_rn(scaled));
    }

    __device__ void addSample(
        DeviceObserver const& observer,
        DeviceWaveforms const& waveforms, double time_s,
        Vec3 contribution,
        unsigned long long* contribution_counter,
        DeviceRadioCounters* counters) {
      if (time_s < observer.start_time_s ||
          time_s >
              observer.start_time_s + observer.duration_s) {
        return;
      }
      auto const bin_value =
          floor((time_s - observer.start_time_s) *
                    observer.sample_rate_Hz +
                0.5);
      if (!(bin_value >= 0.)) {
        return;
      }
      auto const bin =
          static_cast<std::uint64_t>(bin_value);
      if (bin >= observer.number_of_bins) {
        return;
      }
      auto const offset = observer.waveform_offset + bin;
      if (waveforms.fixed_x != nullptr) {
        auto const valid_x = addFixedPoint(
            waveforms.fixed_x + offset, contribution.x,
            observer.fixed_point_scale);
        auto const valid_y = addFixedPoint(
            waveforms.fixed_y + offset, contribution.y,
            observer.fixed_point_scale);
        auto const valid_z = addFixedPoint(
            waveforms.fixed_z + offset, contribution.z,
            observer.fixed_point_scale);
        if (!(valid_x && valid_y && valid_z)) {
          atomicAdd(&counters->fixed_point_overflows, 1ULL);
          return;
        }
      } else {
        atomicAdd(
            waveforms.floating_x + offset, contribution.x);
        atomicAdd(
            waveforms.floating_y + offset, contribution.y);
        atomicAdd(
            waveforms.floating_z + offset, contribution.z);
      }
      atomicAdd(contribution_counter, 1ULL);
    }

    __device__ void separateSameBinEndpoints(
        double& start_time, double& end_time,
        double grid_resolution, bool signed_order) {
      auto const start_bin = static_cast<long long>(
          floor(start_time / grid_resolution + 0.5));
      auto const end_bin = static_cast<long long>(
          floor(end_time / grid_resolution + 0.5));
      if (start_bin != end_bin) {
        return;
      }
      auto const start_fraction =
          start_time / grid_resolution -
          floor(start_time / grid_resolution);
      auto const end_fraction =
          end_time / grid_resolution -
          floor(end_time / grid_resolution);
      auto const forward = !signed_order || end_time >= start_time;
      if (forward) {
        if (start_fraction >= 0.5 && end_fraction >= 0.5) {
          start_time -= grid_resolution;
        } else if (start_fraction < 0.5 &&
                   end_fraction < 0.5) {
          end_time += grid_resolution;
        } else if (end_fraction >=
                   1. - start_fraction) {
          end_time += grid_resolution;
        } else {
          start_time -= grid_resolution;
        }
      } else {
        if (start_fraction >= 0.5 && end_fraction >= 0.5) {
          end_time -= grid_resolution;
        } else if (start_fraction < 0.5 &&
                   end_fraction < 0.5) {
          start_time += grid_resolution;
        } else if (start_fraction >=
                   1. - end_fraction) {
          start_time += grid_resolution;
        } else {
          end_time -= grid_resolution;
        }
      }
    }

    __device__ void accumulateCoREAS(
        RadioTrackKinematics const& track,
        DevicePropagation const& propagation,
        DeviceObserver const& observer,
        DeviceWaveforms const& waveforms,
        DeviceRadioCounters* counters) {
      auto const path_start =
          propagate(propagation, track.start, observer);
      auto const path_end =
          propagate(propagation, track.end, observer);
      auto const pre_doppler =
          1. - path_start.refractive_index_source *
                   dot(track.beta, path_start.emit);
      auto const post_doppler =
          1. - path_end.refractive_index_source *
                   dot(track.beta, path_end.emit);
      auto start_receive =
          track.start_time_s +
          path_start.propagation_time_s;
      auto end_receive =
          track.end_time_s +
          path_end.propagation_time_s;

      if (path_start.refractive_index_destination > 1. &&
          (fabs(pre_doppler) < CoREASApproximationThreshold ||
           fabs(post_doppler) <
               CoREASApproximationThreshold)) {
        auto const midpoint =
            (track.start + track.end) * 0.5;
        auto const middle_time =
            (track.start_time_s +
             track.end_time_s) *
            0.5;
        auto const path_middle =
            propagate(propagation, midpoint, observer);
        auto const middle_receive =
            middle_time + path_middle.propagation_time_s;
        auto const middle_doppler =
            1. - path_middle.refractive_index_source *
                     dot(track.beta, path_middle.emit);
        auto field_start =
            transverseEndpoint(
                path_middle.emit, track.beta) *
            (track.constant * observer.sample_rate_Hz /
             (middle_doppler * path_middle.distance_m));
        auto field_end = field_start * -1.;
        auto delta_time =
            track.track_length_m /
            (SpeedOfLightMPerS *
             track.beta_module) *
            fabs(middle_doppler);
        if (start_receive < end_receive) {
          start_receive =
              middle_receive - 0.5 * delta_time;
          end_receive =
              middle_receive + 0.5 * delta_time;
        } else {
          start_receive =
              middle_receive + 0.5 * delta_time;
          end_receive =
              middle_receive - 0.5 * delta_time;
        }
        auto const grid_resolution =
            1. / observer.sample_rate_Hz;
        delta_time = end_receive - start_receive;
        if (fabs(delta_time) < grid_resolution) {
          auto const scale =
              fabs(delta_time / grid_resolution);
          field_start = field_start * scale;
          field_end = field_end * scale;
          separateSameBinEndpoints(
              start_receive, end_receive, grid_resolution, true);
        }
        addSample(observer, waveforms, start_receive,
                  field_start,
                  &counters->coreas_contributions, counters);
        addSample(observer, waveforms, end_receive, field_end,
                  &counters->coreas_contributions, counters);
        return;
      }

      auto field_start =
          transverseEndpoint(
              path_start.emit, track.beta) *
          (track.constant * observer.sample_rate_Hz /
           (pre_doppler * path_start.distance_m));
      auto field_end =
          transverseEndpoint(
              path_end.emit, track.beta) *
          (-track.constant * observer.sample_rate_Hz /
           (post_doppler * path_end.distance_m));
      if (pre_doppler < 1.e-9 || post_doppler < 1.e-9) {
        auto const grid_resolution =
            1. / observer.sample_rate_Hz;
        auto const delta_time = end_receive - start_receive;
        if (fabs(delta_time) < grid_resolution) {
          auto const scale =
              fabs(delta_time / grid_resolution);
          field_start = field_start * scale;
          field_end = field_end * scale;
          separateSameBinEndpoints(
              start_receive, end_receive, grid_resolution, false);
        }
      }
      addSample(observer, waveforms, start_receive, field_start,
                &counters->coreas_contributions, counters);
      addSample(observer, waveforms, end_receive, field_end,
                &counters->coreas_contributions, counters);
    }

    __device__ void accumulateZhsSegment(
        Vec3 point1, Vec3 point2, double time1, double time2,
        Vec3 beta, double constant,
        DevicePropagation const& propagation,
        DeviceObserver const& observer,
        DeviceWaveforms const& waveforms,
        DeviceRadioCounters* counters, bool subdivided) {
      auto const midpoint = (point1 + point2) * 0.5;
      auto const path =
          propagate(propagation, midpoint, observer);
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
      auto const start_bin = floor(
          (detection_time1 - observer.start_time_s) *
              observer.sample_rate_Hz +
          0.5);
      auto const end_bin = floor(
          (detection_time2 - observer.start_time_s) *
              observer.sample_rate_Hz +
          0.5);
      auto const beta_perpendicular =
          cross(path.emit, cross(beta, path.emit));
      auto const denominator =
          1. - n_source * beta_times_k;

      if (start_bin == end_bin) {
        Vec3 potential{};
        if (fabs(denominator) > 1.e-15) {
          auto const fraction = fabs(
              detection_time2 * observer.sample_rate_Hz -
              detection_time1 * observer.sample_rate_Hz);
          potential =
              beta_perpendicular *
              (sign * constant * fraction /
               (denominator * path.distance_m));
        } else {
          auto const fraction =
              (time2 - time1) * observer.sample_rate_Hz;
          potential =
              beta_perpendicular *
              (sign * constant * fraction / path.distance_m);
        }
        addSample(observer, waveforms, detection_time2,
                  potential, &counters->zhs_contributions,
                  counters);
        return;
      }

      auto const number_of_bins =
          static_cast<int>(end_bin - start_bin);
      auto fraction = fabs(
          start_bin + 0.5 -
          (detection_time1 - observer.start_time_s) *
              observer.sample_rate_Hz);
      auto potential =
          beta_perpendicular *
          (sign * constant * fraction /
           (denominator * path.distance_m));
      addSample(observer, waveforms, detection_time1,
                potential, &counters->zhs_contributions,
                counters);
      for (int index = 1; index < number_of_bins; ++index) {
        // The scalar ZHS subdivided-track branch omits sign for intermediate
        // bins, whereas its whole-track branch retains it.
        potential =
            beta_perpendicular *
            ((subdivided ? 1. : sign) * constant /
             (denominator * path.distance_m));
        addSample(
            observer, waveforms,
            detection_time1 +
                static_cast<double>(index) /
                    observer.sample_rate_Hz,
            potential, &counters->zhs_contributions, counters);
      }
      fraction = fabs(
          (detection_time2 - observer.start_time_s) *
                  observer.sample_rate_Hz +
              0.5 -
          end_bin);
      potential =
          beta_perpendicular *
          (sign * constant * fraction /
           (denominator * path.distance_m));
      addSample(observer, waveforms, detection_time2,
                potential, &counters->zhs_contributions,
                counters);
    }

    __device__ void accumulateZHS(
        RadioTrackKinematics const& track,
        DevicePropagation const& propagation,
        DeviceObserver const& observer,
        DeviceWaveforms const& waveforms,
        DeviceRadioCounters* counters) {
      auto const midpoint =
          (track.start + track.end) * 0.5;
      auto const middle_path =
          propagate(propagation, midpoint, observer);
      auto const u_times_k =
          dot(track.beta, middle_path.emit) /
          track.beta_module;
      auto const sin_theta_squared =
          1. - u_times_k * u_times_k;
      auto const wavelength =
          SpeedOfLightMPerS / observer.sample_rate_Hz;
      auto const fraunhofer =
          sin_theta_squared *
          track.track_length_m *
          track.track_length_m /
          middle_path.distance_m / wavelength * 2. * Pi;
      auto subtrack_divisor = 1.;
      auto number_of_subtracks = 1;
      if (fraunhofer > 1.) {
        // CORSIKA 8 icrc2025-beta2 deliberately keeps the Fraunhofer
        // subdivision count as a double.  Its scalar loop compares an int
        // index with that double while the spatial and time steps are divided
        // by the unrounded value.  Preserve those exact semantics here so the
        // CUDA and released scalar ZHS projections remain equivalent.
        subtrack_divisor = sqrt(fraunhofer) + 1.;
        number_of_subtracks =
            static_cast<int>(ceil(subtrack_divisor));
      }
      atomicAdd(&counters->zhs_subtracks,
                static_cast<unsigned long long>(
                    number_of_subtracks));
      auto const spatial_step =
          track.displacement /
          subtrack_divisor;
      auto const time_step =
          track.duration_s /
          subtrack_divisor;
      auto const refinement =
          static_cast<int>(propagation.zhs_subtrack_refinement);
      auto const refined_spatial_step =
          spatial_step / static_cast<double>(refinement);
      auto const refined_time_step =
          time_step / static_cast<double>(refinement);
      auto point1 = track.start;
      auto time1 = track.start_time_s;
      for (int index = 0; index < number_of_subtracks; ++index) {
        for (int refined_index = 0;
             refined_index < refinement; ++refined_index) {
          auto const point2 = point1 + refined_spatial_step;
          auto const time2 = time1 + refined_time_step;
          accumulateZhsSegment(
              point1, point2, time1, time2, track.beta,
              track.constant, propagation, observer, waveforms, counters,
              number_of_subtracks > 1 || refinement > 1);
          point1 = point2;
          time1 = time2;
        }
      }
      if (refinement > 1) {
        atomicAdd(
            &counters->zhs_subtracks,
            static_cast<unsigned long long>(
                number_of_subtracks * (refinement - 1)));
      }
    }

    __device__ bool validRadioTrack(
        em::LeptonTransportRecord const& record) {
      if (!em::isElectronOrPositronPid(record.start.pid)) {
        return false;
      }
      auto const duration =
          record.end.time_s - record.start.time_s;
      auto const dx =
          record.end.position_m[0] -
          record.start.position_m[0];
      auto const dy =
          record.end.position_m[1] -
          record.start.position_m[1];
      auto const dz =
          record.end.position_m[2] -
          record.start.position_m[2];
      return duration > 0. &&
             (dx * dx + dy * dy + dz * dz) > 0.;
    }

    __device__ void atomicMaxPositiveDouble(
        double* address, double value) {
      // IEEE-754 bit order is monotonic for finite non-negative doubles.
      auto* bits =
          reinterpret_cast<unsigned long long*>(address);
      atomicMax(bits, __double_as_longlong(value));
    }

    __device__ std::size_t diagnosticEnergyBin(
        double kinetic_energy_GeV) {
      constexpr double upper_edges[14]{
          1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2,
          5.e-2, 1.e-1, 2.e-1, 5.e-1, 1.,
          2., 5., 10., 100.};
      std::size_t index = 0;
      while (index < 14 &&
             kinetic_energy_GeV > upper_edges[index]) {
        ++index;
      }
      return index;
    }

    __global__ void countValidTracksKernel(
        em::LeptonTransportRecord const* records,
        std::size_t record_count,
        DeviceRadioCounters* counters,
        bool collect_diagnostics) {
      auto const record_index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      if (record_index >= record_count) {
        return;
      }
      auto const& record = records[record_index];
      if (!validRadioTrack(record)) {
        return;
      }
      atomicAdd(&counters->valid_tracks, 1ULL);
      if (!collect_diagnostics) {
        return;
      }

      auto const dx =
          record.end.position_m[0] -
          record.start.position_m[0];
      auto const dy =
          record.end.position_m[1] -
          record.start.position_m[1];
      auto const dz =
          record.end.position_m[2] -
          record.start.position_m[2];
      auto const length_m =
          sqrt(dx * dx + dy * dy + dz * dz);
      auto const weight = record.start.weight;
      auto const kinetic_energy_GeV =
          record.start.energy_GeV - em::ElectronMassGeV;
      if (!isfinite(length_m) || !(length_m > 0.) ||
          !isfinite(weight) || weight < 0. ||
          !isfinite(kinetic_energy_GeV) ||
          kinetic_energy_GeV < 0.) {
        return;
      }

      auto const weighted_length_m = weight * length_m;
      atomicAdd(
          &counters->weighted_segment_count, weight);
      atomicAdd(&counters->track_length_m, length_m);
      atomicAdd(
          &counters->weighted_track_length_m,
          weighted_length_m);
      atomicAdd(
          &counters->energy_weighted_track_length_GeV_m,
          weighted_length_m * kinetic_energy_GeV);
      atomicMaxPositiveDouble(
          &counters->maximum_segment_length_m, length_m);
      if (record.start.pid ==
          static_cast<std::int32_t>(em::EmPid::Electron)) {
        atomicAdd(
            &counters->electron_weighted_track_length_m,
            weighted_length_m);
        atomicAdd(
            &counters->signed_charge_weighted_track_length_m,
            -weighted_length_m);
      } else {
        atomicAdd(
            &counters->positron_weighted_track_length_m,
            weighted_length_m);
        atomicAdd(
            &counters->signed_charge_weighted_track_length_m,
            weighted_length_m);
      }
      atomicAdd(
          &counters
               ->weighted_track_length_by_kinetic_energy_m
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
      auto const direction_change_rad =
          acos(fmin(1., fmax(-1., direction_dot)));
      auto const duration_s =
          record.end.time_s - record.start.time_s;
      auto const beta_module =
          length_m / (SpeedOfLightMPerS * duration_s);
      auto const time_residual_s =
          duration_s - length_m / SpeedOfLightMPerS;
      atomicAdd(
          &counters->weighted_direction_change_rad,
          weight * direction_change_rad);
      atomicAdd(
          &counters->weighted_direction_change_squared_rad2,
          weight * direction_change_rad * direction_change_rad);
      atomicAdd(
          &counters->weighted_beta_deficit_track_length_m,
          weighted_length_m * (1. - beta_module));
      atomicAdd(
          &counters->weighted_time_residual_s,
          weight * time_residual_s);
      atomicMaxPositiveDouble(
          &counters->maximum_direction_change_rad,
          direction_change_rad);
      auto const charge_sign =
          record.start.pid ==
                  static_cast<std::int32_t>(em::EmPid::Electron)
              ? -1.
              : 1.;
      for (int axis = 0; axis < 3; ++axis) {
        atomicAdd(
            &counters
                 ->signed_charge_weighted_direction_change[axis],
            charge_sign * weight * direction_delta[axis]);
      }
    }

    __global__ void coreasKernel(
        em::LeptonTransportRecord const* records,
        std::size_t record_count,
        DevicePropagation propagation,
        DeviceObserver const* observers,
        std::size_t observer_count, DeviceWaveforms waveforms,
        DeviceRadioCounters* counters) {
      auto const pair_index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      auto const pair_count = record_count * observer_count;
      if (pair_index >= pair_count) {
        return;
      }
      auto const record_index = pair_index / observer_count;
      auto const observer_index = pair_index % observer_count;
      RadioTrackKinematics track{};
      if (!makeRadioTrackKinematics(
              records[record_index], track)) {
        return;
      }
      accumulateCoREAS(
          track, propagation,
          observers[observer_index], waveforms, counters);
    }

    __global__ void zhsKernel(
        em::LeptonTransportRecord const* records,
        std::size_t record_count,
        DevicePropagation propagation,
        DeviceObserver const* observers,
        std::size_t observer_count, DeviceWaveforms waveforms,
        DeviceRadioCounters* counters) {
      auto const pair_index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      auto const pair_count = record_count * observer_count;
      if (pair_index >= pair_count) {
        return;
      }
      auto const record_index = pair_index / observer_count;
      auto const observer_index = pair_index % observer_count;
      RadioTrackKinematics track{};
      if (!makeRadioTrackKinematics(
              records[record_index], track)) {
        return;
      }
      accumulateZHS(
          track, propagation,
          observers[observer_index], waveforms, counters);
    }

    __global__ void coreasZhsKernel(
        em::LeptonTransportRecord const* records,
        std::size_t record_count,
        DevicePropagation propagation,
        DeviceObserver const* coreas_observers,
        DeviceObserver const* zhs_observers,
        std::size_t observer_count,
        DeviceWaveforms coreas_waveforms,
        DeviceWaveforms zhs_waveforms,
        DeviceRadioCounters* counters) {
      auto const pair_index =
          static_cast<std::size_t>(blockIdx.x) * blockDim.x +
          threadIdx.x;
      auto const pair_count = record_count * observer_count;
      if (pair_index >= pair_count) {
        return;
      }
      auto const record_index = pair_index / observer_count;
      auto const observer_index = pair_index % observer_count;
      RadioTrackKinematics track{};
      if (!makeRadioTrackKinematics(
              records[record_index], track)) {
        return;
      }
      accumulateCoREAS(
          track, propagation,
          coreas_observers[observer_index],
          coreas_waveforms, counters);
      accumulateZHS(
          track, propagation,
          zhs_observers[observer_index],
          zhs_waveforms, counters);
    }

    std::vector<DeviceObserver> makeDeviceObservers(
        std::vector<RadioObserverSnapshot> const& input,
        std::size_t& total_bins, bool deterministic,
        bool zhs_potential,
        double fixed_point_field_limit_V_per_m) {
      std::vector<DeviceObserver> result;
      result.reserve(input.size());
      total_bins = 0;
      for (auto const& observer : input) {
        DeviceObserver device{};
        for (int axis = 0; axis < 3; ++axis) {
          device.position_m[axis] = observer.position_m[axis];
        }
        device.start_time_s = observer.start_time_s;
        device.duration_s = observer.duration_s;
        device.sample_rate_Hz = observer.sample_rate_Hz;
        auto const potential_factor =
            zhs_potential ? observer.sample_rate_Hz : 1.;
        device.fixed_point_scale =
            deterministic
                ? FixedPointHeadroom * potential_factor /
                      fixed_point_field_limit_V_per_m
                : 1.;
        device.inverse_fixed_point_scale =
            1. / device.fixed_point_scale;
        device.number_of_bins = observer.number_of_bins;
        device.waveform_offset = total_bins;
        total_bins = checkedAdd(
            total_bins,
            static_cast<std::size_t>(observer.number_of_bins),
            "GPU radio waveform bin count overflow");
        result.push_back(device);
      }
      return result;
    }

    bool observersCanShareTrackKinematics(
        std::vector<DeviceObserver> const& coreas,
        std::vector<DeviceObserver> const& zhs) {
      if (coreas.size() != zhs.size() ||
          coreas.empty()) {
        return false;
      }
      for (std::size_t index = 0;
           index < coreas.size(); ++index) {
        auto const& lhs = coreas[index];
        auto const& rhs = zhs[index];
        if (lhs.start_time_s != rhs.start_time_s ||
            lhs.duration_s != rhs.duration_s ||
            lhs.sample_rate_Hz != rhs.sample_rate_Hz ||
            lhs.number_of_bins != rhs.number_of_bins) {
          return false;
        }
        for (int axis = 0; axis < 3; ++axis) {
          if (lhs.position_m[axis] !=
              rhs.position_m[axis]) {
            return false;
          }
        }
      }
      return true;
    }

  } // namespace

  class CudaRadioAccumulator::Impl {
  public:
    ~Impl() { release(); }

    void initialize(GpuRadioConfig const& requested, int device,
                    std::size_t memory_budget_bytes) {
      if (initialized_) {
        throw std::logic_error(
            "CUDA radio accumulator is already initialized");
      }
      if (!requested.enabled) {
        config_ = requested;
        return;
      }
      validate(requested);
      config_ = requested;
      device_ = device;
      checkCuda(cudaSetDevice(device_), "set CUDA radio device");

      auto const coreas_host = makeDeviceObservers(
          config_.coreas_observers, coreas_bins_,
          config_.deterministic, false,
          config_.fixed_point_field_limit_V_per_m);
      auto const zhs_host = makeDeviceObservers(
          config_.zhs_observers, zhs_bins_,
          config_.deterministic, true,
          config_.fixed_point_field_limit_V_per_m);
      fuse_coreas_zhs_ =
          config_.coreas_enabled &&
          config_.zhs_enabled &&
          observersCanShareTrackKinematics(
              coreas_host, zhs_host);
      auto const table_bytes =
          checkedMultiply(
              config_.propagation.refractivity.size(), sizeof(double),
              "GPU radio propagation table byte overflow");
      auto const observer_bytes = checkedAdd(
          checkedMultiply(coreas_host.size(), sizeof(DeviceObserver),
                          "GPU CoREAS observer byte overflow"),
          checkedMultiply(zhs_host.size(), sizeof(DeviceObserver),
                          "GPU ZHS observer byte overflow"),
          "GPU radio observer byte overflow");
      auto const waveform_values =
          checkedMultiply(
              checkedAdd(coreas_bins_, zhs_bins_,
                         "GPU radio waveform count overflow"),
              std::size_t{3},
              "GPU radio waveform polarization overflow");
      auto const waveform_bytes =
          checkedMultiply(waveform_values, sizeof(double),
                          "GPU radio waveform byte overflow");
      device_bytes_ = checkedAdd(
          checkedAdd(checkedMultiply(table_bytes, 2,
                                     "GPU radio table pair overflow"),
                     observer_bytes,
                     "GPU radio static byte overflow"),
          checkedAdd(waveform_bytes, sizeof(DeviceRadioCounters),
                     "GPU radio output byte overflow"),
          "GPU radio allocation overflow");
      if (device_bytes_ > memory_budget_bytes) {
        throw std::runtime_error(
            "GPU radio buffers exceed configured device memory budget");
      }

      initialized_ = true;
      try {
        createAsyncInfrastructure();
        allocateAndUpload(
            coreas_host, zhs_host, table_bytes, waveform_bytes);
        statistics_.device_bytes = device_bytes_;
      } catch (...) {
        release();
        throw;
      }
    }

    bool enabled() const noexcept {
      return initialized_ && config_.enabled;
    }

    void waitForInputSlot(std::size_t input_slot) {
      if (!enabled()) {
        return;
      }
      if (input_slot >= input_slots_.size()) {
        throw std::out_of_range(
            "CUDA radio input slot is out of range");
      }
      auto& slot = input_slots_[input_slot];
      if (!slot.active) {
        return;
      }
      auto const wait_start = std::chrono::steady_clock::now();
      checkCuda(
          cudaEventSynchronize(slot.radio_done),
          "wait for CUDA radio input slot");
      auto const wait_stop = std::chrono::steady_clock::now();
      ++statistics_.input_slot_waits;
      statistics_.input_slot_host_wait_time_ms +=
          std::chrono::duration<double, std::milli>(
              wait_stop - wait_start)
              .count();
      float elapsed_ms = 0.;
      checkCuda(
          cudaEventElapsedTime(
              &elapsed_ms, slot.radio_start,
              slot.radio_done),
          "measure CUDA radio device time");
      statistics_.device_time_ms += elapsed_ms;
      slot.active = false;
    }

    void drain() {
      if (!enabled()) {
        return;
      }
      for (std::size_t slot = 0;
           slot < input_slots_.size(); ++slot) {
        waitForInputSlot(slot);
      }
    }

    void accumulateLeptonTracksOnDevice(
        em::LeptonTransportRecord const* records,
        std::size_t count, std::size_t input_slot) {
      if (!enabled() || count == 0) {
        return;
      }
      if (records == nullptr) {
        throw std::invalid_argument(
            "CUDA radio received a null track array");
      }
      if (input_slot >= input_slots_.size()) {
        throw std::out_of_range(
            "CUDA radio input slot is out of range");
      }
      waitForInputSlot(input_slot);
      auto& slot = input_slots_[input_slot];
      auto const start = std::chrono::steady_clock::now();
      checkCuda(
          cudaEventRecord(slot.input_ready),
          "record CUDA radio input readiness");
      checkCuda(
          cudaStreamWaitEvent(
              radio_stream_, slot.input_ready, 0),
          "wait for CUDA radio input readiness");
      checkCuda(
          cudaEventRecord(
              slot.radio_start, radio_stream_),
          "record CUDA radio batch start");
      auto const track_blocks = static_cast<unsigned int>(
          (count + ThreadsPerBlock - 1) /
          ThreadsPerBlock);
      countValidTracksKernel<<<track_blocks, ThreadsPerBlock, 0,
                               radio_stream_>>>(
          records, count, device_counters_,
          config_.track_diagnostics);
      checkCuda(cudaGetLastError(),
                "launch CUDA radio valid-track counter");
      auto launch = [&](auto kernel, std::size_t observers,
                        DeviceObserver const* device_observers,
                        DeviceWaveforms waveforms,
                        char const* operation) {
        if (observers == 0) {
          return;
        }
        auto const pairs = checkedMultiply(
            count, observers,
            "GPU radio track-observer pair count overflow");
        auto const blocks = static_cast<unsigned int>(
            (pairs + ThreadsPerBlock - 1) / ThreadsPerBlock);
        kernel<<<blocks, ThreadsPerBlock, 0, radio_stream_>>>(
            records, count, device_propagation_, device_observers,
            observers, waveforms, device_counters_);
        checkCuda(cudaGetLastError(), operation);
        statistics_.track_observer_pairs += pairs;
      };
      if (fuse_coreas_zhs_) {
        auto const observers =
            config_.coreas_observers.size();
        auto const pairs = checkedMultiply(
            count, observers,
            "GPU fused radio track-observer pair count overflow");
        auto const blocks = static_cast<unsigned int>(
            (pairs + ThreadsPerBlock - 1) /
            ThreadsPerBlock);
        coreasZhsKernel
            <<<blocks, ThreadsPerBlock, 0, radio_stream_>>>(
                records, count, device_propagation_,
                device_coreas_observers_,
                device_zhs_observers_, observers,
                coreas_waveforms_, zhs_waveforms_,
                device_counters_);
        checkCuda(
            cudaGetLastError(),
            "launch fused CUDA CoREAS/ZHS kernel");
        statistics_.fused_track_observer_pairs +=
            pairs;
        statistics_.track_observer_pairs +=
            checkedMultiply(
                pairs, std::size_t{2},
                "GPU logical radio pair count overflow");
      } else {
        if (config_.coreas_enabled) {
          launch(
              coreasKernel,
              config_.coreas_observers.size(),
              device_coreas_observers_,
              coreas_waveforms_,
              "launch CUDA CoREAS kernel");
        }
        if (config_.zhs_enabled) {
          launch(
              zhsKernel,
              config_.zhs_observers.size(),
              device_zhs_observers_,
              zhs_waveforms_,
              "launch CUDA ZHS kernel");
        }
      }
      checkCuda(
          cudaEventRecord(slot.radio_done, radio_stream_),
          "record CUDA radio batch completion");
      slot.active = true;
      // Input readiness is an explicit event dependency. Do not add a
      // per-radio-kernel stream or device synchronization here.
      auto const stop = std::chrono::steady_clock::now();
      statistics_.kernel_time_ms +=
          std::chrono::duration<double, std::milli>(stop - start)
              .count();
    }

    void accumulateLeptonTracksFromHost(
        std::vector<em::LeptonTransportRecord> const& records) {
      if (!enabled() || records.empty()) {
        return;
      }
      auto const bytes = checkedMultiply(
          records.size(), sizeof(em::LeptonTransportRecord),
          "CUDA replay radio track byte overflow");
      checkCuda(cudaSetDevice(device_),
                "set CUDA replay radio device");
      em::LeptonTransportRecord* device_records = nullptr;
      auto const start = std::chrono::steady_clock::now();
      checkCuda(
          cudaMalloc(
              reinterpret_cast<void**>(&device_records), bytes),
          "allocate CUDA replay radio tracks");
      try {
        checkCuda(
            cudaMemcpy(
                device_records, records.data(), bytes,
                cudaMemcpyHostToDevice),
            "upload CUDA replay radio tracks");
        auto const copied = std::chrono::steady_clock::now();
        statistics_.host_to_device_bytes += bytes;
        statistics_.transfer_time_ms +=
            std::chrono::duration<double, std::milli>(
                copied - start)
                .count();
        accumulateLeptonTracksOnDevice(
            device_records, records.size(), 0);
        // The device buffer must remain alive until both projection kernels
        // have consumed it.
        drain();
      } catch (...) {
        cudaFree(device_records);
        throw;
      }
      checkCuda(
          cudaFree(device_records),
          "free CUDA replay radio tracks");
    }

    GpuRadioWaveforms downloadWaveforms() {
      GpuRadioWaveforms result{};
      if (!enabled()) {
        return result;
      }
      drain();
      auto const start = std::chrono::steady_clock::now();
      DeviceRadioCounters counters{};
      checkCuda(cudaMemcpy(&counters, device_counters_,
                           sizeof(counters), cudaMemcpyDeviceToHost),
                "download CUDA radio counters");
      statistics_.lepton_tracks = counters.valid_tracks;
      statistics_.coreas_contributions =
          counters.coreas_contributions;
      statistics_.zhs_contributions =
          counters.zhs_contributions;
      statistics_.zhs_subtracks = counters.zhs_subtracks;
      statistics_.fixed_point_overflows =
          counters.fixed_point_overflows;
      statistics_.track_diagnostics_enabled =
          config_.track_diagnostics;
      statistics_.weighted_segment_count =
          counters.weighted_segment_count;
      statistics_.track_length_m =
          counters.track_length_m;
      statistics_.weighted_track_length_m =
          counters.weighted_track_length_m;
      statistics_.electron_weighted_track_length_m =
          counters.electron_weighted_track_length_m;
      statistics_.positron_weighted_track_length_m =
          counters.positron_weighted_track_length_m;
      statistics_.signed_charge_weighted_track_length_m =
          counters.signed_charge_weighted_track_length_m;
      statistics_.energy_weighted_track_length_GeV_m =
          counters.energy_weighted_track_length_GeV_m;
      statistics_.maximum_segment_length_m =
          counters.maximum_segment_length_m;
      statistics_.weighted_direction_change_rad =
          counters.weighted_direction_change_rad;
      statistics_.weighted_direction_change_squared_rad2 =
          counters.weighted_direction_change_squared_rad2;
      statistics_.weighted_beta_deficit_track_length_m =
          counters.weighted_beta_deficit_track_length_m;
      statistics_.weighted_time_residual_s =
          counters.weighted_time_residual_s;
      statistics_.maximum_direction_change_rad =
          counters.maximum_direction_change_rad;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        statistics_.signed_charge_weighted_direction_change[axis] =
            counters
                .signed_charge_weighted_direction_change[axis];
      }
      for (std::size_t index = 0; index < 15; ++index) {
        statistics_
            .weighted_track_length_by_kinetic_energy_m[index] =
            counters
                .weighted_track_length_by_kinetic_energy_m[index];
      }
      statistics_.device_to_host_bytes += sizeof(counters);
      if (counters.fixed_point_overflows != 0) {
        throw std::runtime_error(
            "CUDA radio fixed-point accumulator overflowed; "
            "increase fixed_point_field_limit_V_per_m");
      }
      result.coreas =
          downloadSet(config_.coreas_observers, coreas_bins_,
                      coreas_waveforms_, false,
                      "download CoREAS waveform");
      result.zhs =
          downloadSet(config_.zhs_observers, zhs_bins_,
                      zhs_waveforms_, true,
                      "download ZHS waveform");
      auto const stop = std::chrono::steady_clock::now();
      statistics_.transfer_time_ms +=
          std::chrono::duration<double, std::milli>(stop - start)
              .count();
      return result;
    }

    void reset() {
      if (!enabled()) {
        return;
      }
      drain();
      auto clear = [](DeviceWaveforms const& waveforms,
                      std::size_t bins,
                      char const* operation) {
        if (bins == 0) {
          return;
        }
        void* allocation =
            waveforms.fixed_x != nullptr
                ? static_cast<void*>(waveforms.fixed_x)
                : static_cast<void*>(waveforms.floating_x);
        auto const element_bytes =
            waveforms.fixed_x != nullptr
                ? sizeof(long long)
                : sizeof(double);
        checkCuda(
            cudaMemset(allocation, 0,
                       3 * bins * element_bytes),
            operation);
      };
      clear(coreas_waveforms_, coreas_bins_,
            "reset CUDA CoREAS waveform");
      clear(zhs_waveforms_, zhs_bins_,
            "reset CUDA ZHS waveform");
      checkCuda(cudaMemset(device_counters_, 0,
                           sizeof(DeviceRadioCounters)),
                "reset CUDA radio counters");
      // reset() is the shower-lifecycle boundary. Static allocations remain
      // resident, while activity counters restart for the next event.
      statistics_ = {};
      statistics_.device_bytes = device_bytes_;
    }

    void shutdown() noexcept { release(); }

    std::size_t deviceBytes() const noexcept { return device_bytes_; }

    GpuRadioStatistics const& statistics() const noexcept {
      return statistics_;
    }

  private:
    void createAsyncInfrastructure() {
      checkCuda(
          cudaStreamCreateWithFlags(
              &radio_stream_, cudaStreamNonBlocking),
          "create CUDA radio stream");
      for (auto& slot : input_slots_) {
        checkCuda(
            cudaEventCreateWithFlags(
                &slot.input_ready, cudaEventDisableTiming),
            "create CUDA radio input-ready event");
        checkCuda(
            cudaEventCreate(&slot.radio_start),
            "create CUDA radio start event");
        checkCuda(
            cudaEventCreate(&slot.radio_done),
            "create CUDA radio completion event");
      }
    }

    static void validate(GpuRadioConfig const& config) {
      auto const& propagation = config.propagation;
      if (!config.coreas_enabled && !config.zhs_enabled) {
        throw std::invalid_argument(
            "enabled CUDA radio config has no algorithm");
      }
      if (!std::isfinite(
              config.fixed_point_field_limit_V_per_m) ||
          !(config.fixed_point_field_limit_V_per_m > 0.)) {
        throw std::invalid_argument(
            "invalid CUDA radio fixed-point field limit");
      }
      if (config.zhs_subtrack_refinement < 1 ||
          config.zhs_subtrack_refinement > 64) {
        throw std::invalid_argument(
            "CUDA ZHS subtrack refinement must be in [1,64]");
      }
      if (propagation.refractivity.size() < 11 ||
          propagation.refractivity.size() !=
              propagation.integrated_refractivity.size() ||
          !std::isfinite(propagation.minimum_height_m) ||
          !std::isfinite(propagation.maximum_height_m) ||
          !(propagation.maximum_height_m >
            propagation.minimum_height_m) ||
          !std::isfinite(propagation.step_m) ||
          !(propagation.step_m > 0.) ||
          !std::isfinite(propagation.inverse_step_per_m) ||
          !(propagation.inverse_step_per_m > 0.)) {
        throw std::invalid_argument(
            "invalid CUDA flat-atmosphere radio snapshot");
      }
      auto validate_observers =
          [&](std::vector<RadioObserverSnapshot> const& observers) {
            for (auto const& observer : observers) {
              if (!std::isfinite(observer.start_time_s) ||
                  !std::isfinite(observer.duration_s) ||
                  !(observer.duration_s >= 0.) ||
                  !std::isfinite(observer.sample_rate_Hz) ||
                  !(observer.sample_rate_Hz > 0.) ||
                  observer.number_of_bins == 0) {
                throw std::invalid_argument(
                    "invalid CUDA radio observer");
              }
              if (config.deterministic &&
                  (!std::isfinite(
                       FixedPointHeadroom *
                       observer.sample_rate_Hz /
                       config.fixed_point_field_limit_V_per_m))) {
                throw std::invalid_argument(
                    "CUDA radio fixed-point scale is not finite");
              }
              auto const height =
                  (observer.position_m[2] -
                   propagation.minimum_height_m) *
                  propagation.inverse_step_per_m;
              if (!std::isfinite(height) || height < 0. ||
                  height + 0.5 >=
                      static_cast<double>(
                          propagation.refractivity.size())) {
                throw std::invalid_argument(
                    "CUDA radio observer lies outside propagation table");
              }
              for (double coordinate : observer.position_m) {
                if (!std::isfinite(coordinate)) {
                  throw std::invalid_argument(
                      "CUDA radio observer position is not finite");
                }
              }
            }
          };
      validate_observers(config.coreas_observers);
      validate_observers(config.zhs_observers);
    }

    void allocateAndUpload(
        std::vector<DeviceObserver> const& coreas_host,
        std::vector<DeviceObserver> const& zhs_host,
        std::size_t table_bytes, std::size_t) {
      checkCuda(cudaMalloc(
                    reinterpret_cast<void**>(&device_refractivity_),
                    table_bytes),
                "allocate CUDA radio refractivity");
      checkCuda(cudaMalloc(
                    reinterpret_cast<void**>(
                        &device_integrated_refractivity_),
                    table_bytes),
                "allocate CUDA radio integrated refractivity");
      checkCuda(cudaMemcpy(
                    device_refractivity_,
                    config_.propagation.refractivity.data(),
                    table_bytes, cudaMemcpyHostToDevice),
                "upload CUDA radio refractivity");
      checkCuda(cudaMemcpy(
                    device_integrated_refractivity_,
                    config_.propagation.integrated_refractivity.data(),
                    table_bytes, cudaMemcpyHostToDevice),
                "upload CUDA radio integrated refractivity");
      statistics_.host_to_device_bytes += 2 * table_bytes;

      uploadObservers(coreas_host, device_coreas_observers_,
                      "upload CUDA CoREAS observers");
      uploadObservers(zhs_host, device_zhs_observers_,
                      "upload CUDA ZHS observers");
      allocateWaveforms(coreas_bins_, config_.deterministic,
                        coreas_waveforms_,
                        "allocate CUDA CoREAS waveforms");
      allocateWaveforms(zhs_bins_, config_.deterministic,
                        zhs_waveforms_,
                        "allocate CUDA ZHS waveforms");
      checkCuda(cudaMalloc(
                    reinterpret_cast<void**>(&device_counters_),
                    sizeof(DeviceRadioCounters)),
                "allocate CUDA radio counters");
      checkCuda(cudaMemset(device_counters_, 0,
                           sizeof(DeviceRadioCounters)),
                "initialize CUDA radio counters");

      auto const& source = config_.propagation;
      device_propagation_.minimum_height_m =
          source.minimum_height_m;
      device_propagation_.maximum_height_m =
          source.maximum_height_m;
      device_propagation_.step_m = source.step_m;
      device_propagation_.inverse_step_per_m =
          source.inverse_step_per_m;
      device_propagation_.slope_refractivity_lower =
          source.slope_refractivity_lower;
      device_propagation_.slope_integrated_refractivity_lower =
          source.slope_integrated_refractivity_lower;
      device_propagation_.slope_refractivity_upper =
          source.slope_refractivity_upper;
      device_propagation_.slope_integrated_refractivity_upper =
          source.slope_integrated_refractivity_upper;
      device_propagation_.refractivity = device_refractivity_;
      device_propagation_.integrated_refractivity =
          device_integrated_refractivity_;
      device_propagation_.table_size =
          source.refractivity.size();
      device_propagation_.zhs_subtrack_refinement =
          config_.zhs_subtrack_refinement;
    }

    void uploadObservers(
        std::vector<DeviceObserver> const& host,
        DeviceObserver*& device, char const* operation) {
      if (host.empty()) {
        return;
      }
      auto const bytes = host.size() * sizeof(DeviceObserver);
      checkCuda(cudaMalloc(reinterpret_cast<void**>(&device), bytes),
                operation);
      checkCuda(cudaMemcpy(device, host.data(), bytes,
                           cudaMemcpyHostToDevice),
                operation);
      statistics_.host_to_device_bytes += bytes;
    }

    static void allocateWaveforms(
        std::size_t bins, bool deterministic,
        DeviceWaveforms& waveforms,
        char const* operation) {
      if (bins == 0) {
        return;
      }
      if (deterministic) {
        long long* allocation = nullptr;
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&allocation),
                      3 * bins * sizeof(long long)),
                  operation);
        checkCuda(cudaMemset(
                      allocation, 0,
                      3 * bins * sizeof(long long)),
                  operation);
        waveforms.fixed_x = allocation;
        waveforms.fixed_y = allocation + bins;
        waveforms.fixed_z = allocation + 2 * bins;
      } else {
        double* allocation = nullptr;
        checkCuda(cudaMalloc(
                      reinterpret_cast<void**>(&allocation),
                      3 * bins * sizeof(double)),
                  operation);
        checkCuda(cudaMemset(
                      allocation, 0,
                      3 * bins * sizeof(double)),
                  operation);
        waveforms.floating_x = allocation;
        waveforms.floating_y = allocation + bins;
        waveforms.floating_z = allocation + 2 * bins;
      }
    }

    std::vector<RadioWaveform> downloadSet(
        std::vector<RadioObserverSnapshot> const& observers,
        std::size_t total_bins, DeviceWaveforms const& waveforms,
        bool zhs_potential, char const* operation) {
      std::vector<RadioWaveform> result;
      result.reserve(observers.size());
      if (total_bins == 0) {
        return result;
      }
      std::vector<double> host(3 * total_bins);
      if (waveforms.fixed_x != nullptr) {
        std::vector<long long> fixed(3 * total_bins);
        checkCuda(cudaMemcpy(
                      fixed.data(), waveforms.fixed_x,
                      fixed.size() * sizeof(long long),
                      cudaMemcpyDeviceToHost),
                  operation);
        std::size_t offset = 0;
        for (auto const& observer : observers) {
          auto const count =
              static_cast<std::size_t>(
                  observer.number_of_bins);
          auto const potential_factor =
              zhs_potential ? observer.sample_rate_Hz : 1.;
          auto const inverse_scale =
              config_.fixed_point_field_limit_V_per_m /
              (FixedPointHeadroom * potential_factor);
          for (std::size_t bin = 0; bin < count; ++bin) {
            auto const local = offset + bin;
            host[local] =
                static_cast<double>(fixed[local]) *
                inverse_scale;
            host[total_bins + local] =
                static_cast<double>(
                    fixed[total_bins + local]) *
                inverse_scale;
            host[2 * total_bins + local] =
                static_cast<double>(
                    fixed[2 * total_bins + local]) *
                inverse_scale;
          }
          offset += count;
        }
        statistics_.device_to_host_bytes +=
            fixed.size() * sizeof(long long);
      } else {
        checkCuda(cudaMemcpy(
                      host.data(), waveforms.floating_x,
                      host.size() * sizeof(double),
                      cudaMemcpyDeviceToHost),
                  operation);
        statistics_.device_to_host_bytes +=
            host.size() * sizeof(double);
      }
      std::size_t offset = 0;
      for (auto const& observer : observers) {
        auto const count =
            static_cast<std::size_t>(observer.number_of_bins);
        RadioWaveform waveform{};
        waveform.x.assign(host.begin() + offset,
                          host.begin() + offset + count);
        waveform.y.assign(host.begin() + total_bins + offset,
                          host.begin() + total_bins + offset + count);
        waveform.z.assign(host.begin() + 2 * total_bins + offset,
                          host.begin() + 2 * total_bins + offset + count);
        result.push_back(std::move(waveform));
        offset += count;
      }
      return result;
    }

    void release() noexcept {
      if (initialized_) {
        cudaSetDevice(device_);
      }
      if (radio_stream_ != nullptr) {
        cudaStreamSynchronize(radio_stream_);
      }
      cudaFree(device_counters_);
      cudaFree(
          zhs_waveforms_.fixed_x != nullptr
              ? static_cast<void*>(zhs_waveforms_.fixed_x)
              : static_cast<void*>(
                    zhs_waveforms_.floating_x));
      cudaFree(
          coreas_waveforms_.fixed_x != nullptr
              ? static_cast<void*>(coreas_waveforms_.fixed_x)
              : static_cast<void*>(
                    coreas_waveforms_.floating_x));
      cudaFree(device_zhs_observers_);
      cudaFree(device_coreas_observers_);
      cudaFree(device_integrated_refractivity_);
      cudaFree(device_refractivity_);
      for (auto& slot : input_slots_) {
        if (slot.radio_done != nullptr) {
          cudaEventDestroy(slot.radio_done);
        }
        if (slot.radio_start != nullptr) {
          cudaEventDestroy(slot.radio_start);
        }
        if (slot.input_ready != nullptr) {
          cudaEventDestroy(slot.input_ready);
        }
        slot = {};
      }
      if (radio_stream_ != nullptr) {
        cudaStreamDestroy(radio_stream_);
      }
      radio_stream_ = nullptr;
      device_counters_ = nullptr;
      zhs_waveforms_ = {};
      coreas_waveforms_ = {};
      device_zhs_observers_ = nullptr;
      device_coreas_observers_ = nullptr;
      device_integrated_refractivity_ = nullptr;
      device_refractivity_ = nullptr;
      device_propagation_ = {};
      fuse_coreas_zhs_ = false;
      coreas_bins_ = 0;
      zhs_bins_ = 0;
      device_bytes_ = 0;
      initialized_ = false;
    }

    GpuRadioConfig config_{};
    int device_{};
    bool initialized_{};
    bool fuse_coreas_zhs_{};
    std::size_t coreas_bins_{};
    std::size_t zhs_bins_{};
    std::size_t device_bytes_{};
    double* device_refractivity_{};
    double* device_integrated_refractivity_{};
    DeviceObserver* device_coreas_observers_{};
    DeviceObserver* device_zhs_observers_{};
    DeviceWaveforms coreas_waveforms_{};
    DeviceWaveforms zhs_waveforms_{};
    DeviceRadioCounters* device_counters_{};
    DevicePropagation device_propagation_{};
    cudaStream_t radio_stream_{};
    std::array<RadioInputSlot, RadioInputSlotCount>
        input_slots_{};
    GpuRadioStatistics statistics_{};
  };

  CudaRadioAccumulator::CudaRadioAccumulator()
      : impl_(std::make_unique<Impl>()) {}

  CudaRadioAccumulator::~CudaRadioAccumulator() = default;

  CudaRadioAccumulator::CudaRadioAccumulator(
      CudaRadioAccumulator&&) noexcept = default;

  CudaRadioAccumulator&
  CudaRadioAccumulator::operator=(CudaRadioAccumulator&&) noexcept =
      default;

  void CudaRadioAccumulator::initialize(
      GpuRadioConfig const& config, int device,
      std::size_t memory_budget_bytes) {
    impl_->initialize(config, device, memory_budget_bytes);
  }

  bool CudaRadioAccumulator::enabled() const noexcept {
    return impl_->enabled();
  }

  void CudaRadioAccumulator::accumulateLeptonTracksOnDevice(
      em::LeptonTransportRecord const* records,
      std::size_t count, std::size_t input_slot) {
    impl_->accumulateLeptonTracksOnDevice(
        records, count, input_slot);
  }

  void CudaRadioAccumulator::accumulateLeptonTracksFromHost(
      std::vector<em::LeptonTransportRecord> const& records) {
    impl_->accumulateLeptonTracksFromHost(records);
  }

  void CudaRadioAccumulator::waitForInputSlot(
      std::size_t input_slot) {
    impl_->waitForInputSlot(input_slot);
  }

  void CudaRadioAccumulator::drain() { impl_->drain(); }

  GpuRadioWaveforms CudaRadioAccumulator::downloadWaveforms() {
    return impl_->downloadWaveforms();
  }

  void CudaRadioAccumulator::reset() { impl_->reset(); }

  void CudaRadioAccumulator::release() noexcept { impl_->shutdown(); }

  std::size_t CudaRadioAccumulator::deviceBytes() const noexcept {
    return impl_->deviceBytes();
  }

  GpuRadioStatistics const&
  CudaRadioAccumulator::statistics() const noexcept {
    return impl_->statistics();
  }

} // namespace corsika::gpu::radio
