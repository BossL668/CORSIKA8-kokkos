/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  CORSIKA_HOST_DEVICE inline double radio_abs(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double radio_sqrt(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(value);
#else
    return std::sqrt(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double radio_floor(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::floor(value);
#else
    return std::floor(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double radio_ceil(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::ceil(value);
#else
    return std::ceil(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline double radio_acos(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::acos(value);
#else
    return std::acos(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline bool radio_finite(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
  }

  CORSIKA_HOST_DEVICE inline long long radio_round(double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return static_cast<long long>(::rint(value));
#else
    return static_cast<long long>(std::rint(value));
#endif
  }

  CORSIKA_HOST_DEVICE inline double radio_min(double left, double right) {
    return left < right ? left : right;
  }

  CORSIKA_HOST_DEVICE inline double radio_max(double left, double right) {
    return left > right ? left : right;
  }

  CORSIKA_HOST_DEVICE inline double observer_nearest_bin(double scaled_time) {
    // Scalar TimeDomainObserver/CoREAS add 0.5L, not a double 0.5.  A
    // double addition rounds nextafter(0.5,0)+0.5 to 1 and incorrectly
    // moves that contribution into the next bin. Split integer/fraction
    // to preserve scalar's extended-precision half-bin decision on GPUs.
    auto const integral = radio_floor(scaled_time);
    return integral + (scaled_time - integral >= 0.5 ? 1. : 0.);
  }

  CORSIKA_HOST_DEVICE inline RadioVector operator+(RadioVector a, RadioVector b) {
    return {a.x_ + b.x_, a.y_ + b.y_, a.z_ + b.z_};
  }

  CORSIKA_HOST_DEVICE inline RadioVector operator-(RadioVector a, RadioVector b) {
    return {a.x_ - b.x_, a.y_ - b.y_, a.z_ - b.z_};
  }

  CORSIKA_HOST_DEVICE inline RadioVector operator*(RadioVector a, double value) {
    return {a.x_ * value, a.y_ * value, a.z_ * value};
  }

  CORSIKA_HOST_DEVICE inline RadioVector operator/(RadioVector a, double value) {
    return {a.x_ / value, a.y_ / value, a.z_ / value};
  }

  CORSIKA_HOST_DEVICE inline double dot(RadioVector a, RadioVector b) {
    return a.x_ * b.x_ + a.y_ * b.y_ + a.z_ * b.z_;
  }

  CORSIKA_HOST_DEVICE inline RadioVector cross(RadioVector a, RadioVector b) {
    return {a.y_ * b.z_ - a.z_ * b.y_, a.z_ * b.x_ - a.x_ * b.z_,
            a.x_ * b.y_ - a.y_ * b.x_};
  }

  CORSIKA_HOST_DEVICE inline double norm(RadioVector value) {
    return radio_sqrt(dot(value, value));
  }

  CORSIKA_HOST_DEVICE inline double co_reas_doppler(double index, RadioVector beta,
                                                    RadioVector emit) {
    auto const ordinary = 1. - index * dot(beta, emit);
    if (ordinary != 0.) return ordinary;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !defined(__SYCL_DEVICE_ONLY__)
    return static_cast<double>(1.L - static_cast<long double>(index) *
                                         (static_cast<long double>(beta.x_) * emit.x_ +
                                          static_cast<long double>(beta.y_) * emit.y_ +
                                          static_cast<long double>(beta.z_) * emit.z_));
#else
    auto const px = beta.x_ * emit.x_;
    auto const py = beta.y_ * emit.y_;
    auto const pz = beta.z_ * emit.z_;
    auto const ex = ::fma(beta.x_, emit.x_, -px);
    auto const ey = ::fma(beta.y_, emit.y_, -py);
    auto const ez = ::fma(beta.z_, emit.z_, -pz);
    auto const xy = px + py;
    auto const xy_virtual = xy - px;
    auto const xy_error = (px - (xy - xy_virtual)) + (py - xy_virtual);
    auto const xyz = xy + pz;
    auto const xyz_virtual = xyz - xy;
    auto const xyz_error = (xy - (xyz - xyz_virtual)) + (pz - xyz_virtual);
    auto const low = ex + ey + ez + xy_error + xyz_error;
    auto const product = index * xyz;
    auto const product_error = ::fma(index, xyz, -product) + index * low;
    // Here product rounded to 1 in the ordinary calculation.  Sterbenz's
    // lemma makes this subtraction exact before restoring its residual.
    return (1. - product) - product_error;
#endif
  }

  CORSIKA_HOST_DEVICE inline bool make_radio_track_kinematics(
      ::corsika::kokkos::LeptonTransportRecord const& record,
      RadioTrackKinematics& track) {
    if (!::corsika::kokkos::is_electron_or_positron_pid(record.start_.pid_)) return false;
    track.start_ = {record.start_.positionM_[0], record.start_.positionM_[1],
                    record.start_.positionM_[2]};
    track.end_ = {record.end_.positionM_[0], record.end_.positionM_[1],
                  record.end_.positionM_[2]};
    track.startTimeS_ = record.start_.timeS_;
    track.endTimeS_ = record.end_.timeS_;
    track.durationS_ = track.endTimeS_ - track.startTimeS_;
    track.displacement_ = track.end_ - track.start_;
    track.trackLengthM_ = norm(track.displacement_);
    if (!(track.durationS_ > 0.) || !(track.trackLengthM_ > 0.)) return false;
    track.beta_ = track.displacement_ / (SpeedOfLightMPerS * track.durationS_);
    track.betaModule_ = track.trackLengthM_ / (SpeedOfLightMPerS * track.durationS_);
    auto const charge = record.start_.pid_ == static_cast<std::int32_t>(
                                                  ::corsika::kokkos::EmPid::Electron)
                            ? -ElementaryChargeC
                            : ElementaryChargeC;
    track.constant_ = charge * EmConstant * record.start_.weight_;
    track.valid_ = 1;
    return true;
  }

  CORSIKA_HOST_DEVICE inline RadioSignalPath propagate(
      RadioPropagationView const& table, RadioVector source,
      RadioObserverView const& observer) {
    auto const destination = RadioVector{observer.positionM_[0], observer.positionM_[1],
                                         observer.positionM_[2]};
    auto const displacement = destination - source;
    auto const distance = norm(displacement);
    auto const emit = displacement / distance;
    if (table.homogeneousRefractiveIndex_ > 0.) {
      auto const index = table.homogeneousRefractiveIndex_;
      return {index * distance / SpeedOfLightMPerS, index, index, distance, emit};
    }
    auto const source_height =
        (source.z_ - table.minimumHeightM_) * table.inverseStepPerM_;
    auto const destination_height =
        (destination.z_ - table.minimumHeightM_) * table.inverseStepPerM_;
    auto const last = table.tableSize_ - 1;
    auto const destination_index = static_cast<std::size_t>(destination_height + 0.5);

    double refractive_index_source = 1.;
    double integrated_source = 1.;
    auto refractive_index_destination = table.refractivity_[destination_index] + 1.;
    auto const integrated_destination = table.integratedRefractivity_[destination_index];
    double height = 1.;

    if (source_height + 0.5 >= static_cast<double>(last)) {
      refractive_index_source =
          table.refractivity_[last] +
          table.slopeRefractivityUpper_ *
              radio_abs((source.z_ - table.maximumHeightM_) * table.inverseStepPerM_) +
          1.;
      integrated_source = table.integratedRefractivity_[last] +
                          table.slopeIntegratedRefractivityUpper_ *
                              radio_abs(table.maximumHeightM_ - source.z_);
      height = source_height - destination_height;
    } else if (source_height + 0.5 < static_cast<double>(last) && source_height > 0.) {
      auto const source_index = static_cast<std::size_t>(source_height + 0.5);
      refractive_index_source = table.refractivity_[source_index] + 1.;
      integrated_source = table.integratedRefractivity_[source_index];
      height = static_cast<double>(static_cast<std::int64_t>(source_index) -
                                   static_cast<std::int64_t>(destination_index)) *
               table.stepM_;
      if (height == 0.) height = 1.;
    } else if (source_height == 0.) {
      // The scalar propagator shadows both index variables at its exact
      // lower edge and returns their initial value, 1.  Preserve that
      // released behavior rather than silently changing it on devices.
      refractive_index_source = 1.;
      refractive_index_destination = 1.;
      integrated_source = table.integratedRefractivity_[0];
      height = destination_height - source_height;
    } else {
      refractive_index_source = table.refractivity_[0] +
                                table.slopeRefractivityLower_ * radio_abs(source_height) +
                                1.;
      integrated_source =
          table.integratedRefractivity_[0] +
          table.slopeIntegratedRefractivityLower_ * radio_abs(source_height);
      // Scalar's below-table branch likewise shadows the destination n.
      refractive_index_destination = 1.;
      height = destination_height - radio_abs(source_height);
    }

    auto const propagation_time =
        (1. + (integrated_source - integrated_destination) / height) * distance /
        SpeedOfLightMPerS;
    return {propagation_time, refractive_index_source, refractive_index_destination,
            distance, emit};
  }

  CORSIKA_HOST_DEVICE inline RadioVector transverse_endpoint(RadioVector emit,
                                                             RadioVector beta) {
    return cross(emit, cross(emit, beta));
  }

  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline bool checked_radio_atomic_add(long long* address,
                                                           long long increment) {
    auto const previous = TAtomicOperations::add(address, increment);
    return !((increment > 0 && previous > SignedIntegerMaximum - increment) ||
             (increment < 0 && previous < SignedIntegerMinimum - increment));
  }

  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline bool add_radio_fixed_point(long long* address,
                                                        double contribution,
                                                        double scale) {
    auto const scaled = contribution * scale;
    if (!radio_finite(scaled) || radio_abs(scaled) >= SignedIntegerLimit) return false;
    return checked_radio_atomic_add<TAtomicOperations>(address, radio_round(scaled));
  }

  template <class TAtomicOperations, bool IsVectorPotential>
  CORSIKA_HOST_DEVICE inline void add_sample(RadioObserverView const& observer,
                                             RadioWaveformView const& waveforms,
                                             double time_s, RadioVector contribution,
                                             unsigned long long* contribution_counter,
                                             RadioCounters* counters) {
    // Match TimeDomainObserver's distinct receive overloads. ZHS potentials
    // need the full support of each stored bin, including the edge half-bins,
    // before differentiation. CoREAS instantaneous fields retain time gating.
    if constexpr (!IsVectorPotential) {
      if (time_s < observer.startTimeS_ ||
          time_s > observer.startTimeS_ + observer.durationS_)
        return;
    }
    auto const bin_value =
        observer_nearest_bin((time_s - observer.startTimeS_) * observer.sampleRateHz_);
    // Validate both bounds before the unsigned conversion (also rejects NaN
    // and infinities). No waveform allocation or output shape is changed.
    if (!(bin_value >= 0. && bin_value < observer.numberOfBins_)) return;
    auto const bin = static_cast<std::uint64_t>(bin_value);
    if (bin >= observer.numberOfBins_) return;
    auto const offset = observer.waveformOffset_ + bin;
    if (waveforms.fixedX_ != nullptr) {
      auto const valid_x = add_radio_fixed_point<TAtomicOperations>(
          waveforms.fixedX_ + offset, contribution.x_, observer.fixedPointScale_);
      auto const valid_y = add_radio_fixed_point<TAtomicOperations>(
          waveforms.fixedY_ + offset, contribution.y_, observer.fixedPointScale_);
      auto const valid_z = add_radio_fixed_point<TAtomicOperations>(
          waveforms.fixedZ_ + offset, contribution.z_, observer.fixedPointScale_);
      if (!(valid_x && valid_y && valid_z)) {
        TAtomicOperations::add(&counters->fixedPointOverflows_, 1ULL);
        return;
      }
    } else {
      TAtomicOperations::add(waveforms.floatingX_ + offset, contribution.x_);
      TAtomicOperations::add(waveforms.floatingY_ + offset, contribution.y_);
      TAtomicOperations::add(waveforms.floatingZ_ + offset, contribution.z_);
    }
    TAtomicOperations::add(contribution_counter, 1ULL);
  }

  CORSIKA_HOST_DEVICE inline void separate_same_bin_endpoints(double& start_time,
                                                              double& end_time,
                                                              double grid_resolution,
                                                              bool signed_order) {
    auto const start_bin =
        static_cast<long long>(observer_nearest_bin(start_time / grid_resolution));
    auto const end_bin =
        static_cast<long long>(observer_nearest_bin(end_time / grid_resolution));
    if (start_bin != end_bin) return;
    auto const start_fraction =
        start_time / grid_resolution - radio_floor(start_time / grid_resolution);
    auto const end_fraction =
        end_time / grid_resolution - radio_floor(end_time / grid_resolution);
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

  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_co_reas(
      RadioTrackKinematics const& track, RadioPropagationView const& propagation,
      RadioObserverView const& observer, RadioWaveformView const& waveforms,
      RadioCounters* counters) {
    auto const path_start = propagate(propagation, track.start_, observer);
    auto const path_end = propagate(propagation, track.end_, observer);
    auto const pre_doppler =
        co_reas_doppler(path_start.refractiveIndexSource_, track.beta_, path_start.emit_);
    auto const post_doppler =
        co_reas_doppler(path_end.refractiveIndexSource_, track.beta_, path_end.emit_);
    auto start_receive = track.startTimeS_ + path_start.propagationTimeS_;
    auto end_receive = track.endTimeS_ + path_end.propagationTimeS_;

    if (path_start.refractiveIndexDestination_ > 1. &&
        (radio_abs(pre_doppler) < CoREASApproximationThreshold ||
         radio_abs(post_doppler) < CoREASApproximationThreshold)) {
      auto const midpoint = (track.start_ + track.end_) * 0.5;
      auto const middle_time = (track.startTimeS_ + track.endTimeS_) * 0.5;
      auto const path_middle = propagate(propagation, midpoint, observer);
      auto const middle_receive = middle_time + path_middle.propagationTimeS_;
      auto const middle_doppler = co_reas_doppler(path_middle.refractiveIndexSource_,
                                                  track.beta_, path_middle.emit_);
      auto field_start = transverse_endpoint(path_middle.emit_, track.beta_) *
                         (track.constant_ * observer.sampleRateHz_ /
                          (middle_doppler * path_middle.distanceM_));
      auto field_end = field_start * -1.;
      auto delta_time = track.trackLengthM_ / (SpeedOfLightMPerS * track.betaModule_) *
                        radio_abs(middle_doppler);
      if (start_receive < end_receive) {
        start_receive = middle_receive - 0.5 * delta_time;
        end_receive = middle_receive + 0.5 * delta_time;
      } else {
        start_receive = middle_receive + 0.5 * delta_time;
        end_receive = middle_receive - 0.5 * delta_time;
      }
      auto const grid_resolution = 1. / observer.sampleRateHz_;
      delta_time = end_receive - start_receive;
      if (radio_abs(delta_time) < grid_resolution) {
        auto const scale = radio_abs(delta_time / grid_resolution);
        field_start = field_start * scale;
        field_end = field_end * scale;
        separate_same_bin_endpoints(start_receive, end_receive, grid_resolution, true);
      }
      add_sample<TAtomicOperations>(observer, waveforms, start_receive, field_start,
                                    &counters->coreasContributions_, counters);
      add_sample<TAtomicOperations>(observer, waveforms, end_receive, field_end,
                                    &counters->coreasContributions_, counters);
      return;
    }

    auto field_start = transverse_endpoint(path_start.emit_, track.beta_) *
                       (track.constant_ * observer.sampleRateHz_ /
                        (pre_doppler * path_start.distanceM_));
    auto field_end = transverse_endpoint(path_end.emit_, track.beta_) *
                     (-track.constant_ * observer.sampleRateHz_ /
                      (post_doppler * path_end.distanceM_));
    if (pre_doppler < 1.e-9 || post_doppler < 1.e-9) {
      auto const grid_resolution = 1. / observer.sampleRateHz_;
      auto const delta_time = end_receive - start_receive;
      if (radio_abs(delta_time) < grid_resolution) {
        auto const scale = radio_abs(delta_time / grid_resolution);
        field_start = field_start * scale;
        field_end = field_end * scale;
        separate_same_bin_endpoints(start_receive, end_receive, grid_resolution, false);
      }
    }
    add_sample<TAtomicOperations>(observer, waveforms, start_receive, field_start,
                                  &counters->coreasContributions_, counters);
    add_sample<TAtomicOperations>(observer, waveforms, end_receive, field_end,
                                  &counters->coreasContributions_, counters);
  }

  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_zhs_segment(
      RadioVector point1, RadioVector point2, double time1, double time2,
      RadioVector beta, double constant, RadioPropagationView const& propagation,
      RadioObserverView const& observer, RadioWaveformView const& waveforms,
      RadioCounters* counters, bool subdivided) {
    auto const midpoint = (point1 + point2) * 0.5;
    auto const path = propagate(propagation, midpoint, observer);
    auto const n_source = path.refractiveIndexSource_;
    auto const beta_times_k = dot(beta, path.emit_);
    auto const middle_time = (time1 + time2) * 0.5;
    auto detection_time1 =
        time1 + path.propagationTimeS_ - n_source * beta_times_k * (time1 - middle_time);
    auto detection_time2 =
        time2 + path.propagationTimeS_ - n_source * beta_times_k * (time2 - middle_time);
    auto sign = 1.;
    if (detection_time1 > detection_time2) {
      detection_time1 = time2 + path.propagationTimeS_ -
                        n_source * beta_times_k * (time2 - middle_time);
      detection_time2 = time1 + path.propagationTimeS_ -
                        n_source * beta_times_k * (time1 - middle_time);
      sign = -1.;
    }
    auto const start_bin = radio_floor(
        (detection_time1 - observer.startTimeS_) * observer.sampleRateHz_ + 0.5);
    auto const end_bin = radio_floor(
        (detection_time2 - observer.startTimeS_) * observer.sampleRateHz_ + 0.5);
    auto const beta_perpendicular = cross(path.emit_, cross(beta, path.emit_));
    auto const denominator = 1. - n_source * beta_times_k;

    if (start_bin == end_bin) {
      RadioVector potential{};
      if (radio_abs(denominator) > 1.e-15) {
        auto const fraction = radio_abs(detection_time2 * observer.sampleRateHz_ -
                                        detection_time1 * observer.sampleRateHz_);
        potential = beta_perpendicular *
                    (sign * constant * fraction / (denominator * path.distanceM_));
      } else {
        auto const fraction = (time2 - time1) * observer.sampleRateHz_;
        potential = beta_perpendicular * (sign * constant * fraction / path.distanceM_);
      }
      add_sample<TAtomicOperations, true>(observer, waveforms, detection_time2, potential,
                                          &counters->zhsContributions_, counters);
      return;
    }

    auto const number_of_bins = static_cast<int>(end_bin - start_bin);
    auto fraction =
        radio_abs(start_bin + 0.5 -
                  (detection_time1 - observer.startTimeS_) * observer.sampleRateHz_);
    auto potential = beta_perpendicular *
                     (sign * constant * fraction / (denominator * path.distanceM_));
    add_sample<TAtomicOperations, true>(observer, waveforms, detection_time1, potential,
                                        &counters->zhsContributions_, counters);
    for (int index = 1; index < number_of_bins; ++index) {
      potential = beta_perpendicular *
                  ((subdivided ? 1. : sign) * constant / (denominator * path.distanceM_));
      add_sample<TAtomicOperations, true>(
          observer, waveforms,
          detection_time1 + static_cast<double>(index) / observer.sampleRateHz_,
          potential, &counters->zhsContributions_, counters);
    }
    fraction =
        radio_abs((detection_time2 - observer.startTimeS_) * observer.sampleRateHz_ +
                  0.5 - end_bin);
    potential = beta_perpendicular *
                (sign * constant * fraction / (denominator * path.distanceM_));
    add_sample<TAtomicOperations, true>(observer, waveforms, detection_time2, potential,
                                        &counters->zhsContributions_, counters);
  }

  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_zhs(RadioTrackKinematics const& track,
                                                 RadioPropagationView const& propagation,
                                                 RadioObserverView const& observer,
                                                 RadioWaveformView const& waveforms,
                                                 RadioCounters* counters) {
    auto const midpoint = (track.start_ + track.end_) * 0.5;
    auto const middle_path = propagate(propagation, midpoint, observer);
    auto const u_times_k = dot(track.beta_, middle_path.emit_) / track.betaModule_;
    auto const sin_theta_squared = 1. - u_times_k * u_times_k;
    auto const wavelength = SpeedOfLightMPerS / observer.sampleRateHz_;
    auto const fraunhofer = sin_theta_squared * track.trackLengthM_ *
                            track.trackLengthM_ / middle_path.distanceM_ / wavelength *
                            2. * Pi;
    auto subtrack_divisor = 1.;
    auto number_of_subtracks = 1;
    if (fraunhofer > 1.) {
      // Preserve the released scalar beta2 semantics: the loop bound is the
      // ceil of this value, while step sizes use the unrounded double.
      subtrack_divisor = radio_sqrt(fraunhofer) + 1.;
      number_of_subtracks = static_cast<int>(radio_ceil(subtrack_divisor));
    }
    TAtomicOperations::add(&counters->zhsSubtracks_,
                           static_cast<unsigned long long>(number_of_subtracks));
    auto const spatial_step = track.displacement_ / subtrack_divisor;
    auto const time_step = track.durationS_ / subtrack_divisor;
    auto const refinement = static_cast<int>(propagation.zhsSubtrackRefinement_);
    auto const refined_spatial_step = spatial_step / static_cast<double>(refinement);
    auto const refined_time_step = time_step / static_cast<double>(refinement);
    auto point1 = track.start_;
    auto time1 = track.startTimeS_;
    for (int index = 0; index < number_of_subtracks; ++index) {
      for (int refined_index = 0; refined_index < refinement; ++refined_index) {
        auto const point2 = point1 + refined_spatial_step;
        auto const time2 = time1 + refined_time_step;
        accumulate_zhs_segment<TAtomicOperations>(
            point1, point2, time1, time2, track.beta_, track.constant_, propagation,
            observer, waveforms, counters, number_of_subtracks > 1 || refinement > 1);
        point1 = point2;
        time1 = time2;
      }
    }
    if (refinement > 1) {
      TAtomicOperations::add(
          &counters->zhsSubtracks_,
          static_cast<unsigned long long>(number_of_subtracks * (refinement - 1)));
    }
  }

  CORSIKA_HOST_DEVICE inline std::size_t diagnostic_energy_bin(
      double kinetic_energy_GeV) {
    constexpr double upper_edges[14]{1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2, 1.e-1,
                                     2.e-1, 5.e-1, 1.,    2.,    5.,    10.,   100.};
    std::size_t index = 0;
    while (index < 14 && kinetic_energy_GeV > upper_edges[index]) ++index;
    return index;
  }

  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline bool record_track(
      ::corsika::kokkos::LeptonTransportRecord const& record,
      RadioTrackKinematics* output, RadioCounters* counters, bool collect_diagnostics) {
    RadioTrackKinematics track{};
    if (!make_radio_track_kinematics(record, track)) {
      if (output != nullptr) *output = {};
      return false;
    }
    if (output != nullptr) *output = track;
    TAtomicOperations::add(&counters->validTracks_, 1ULL);
    if (!collect_diagnostics) return true;

    auto const length_m = track.trackLengthM_;
    auto const weight = record.start_.weight_;
    auto const kinetic_energy_GeV =
        record.start_.energyGeV_ - ::corsika::kokkos::TransportElectronMassGeV;
    if (!radio_finite(length_m) || !(length_m > 0.) || !radio_finite(weight) ||
        weight < 0. || !radio_finite(kinetic_energy_GeV) || kinetic_energy_GeV < 0.)
      return true;

    auto const weighted_length_m = weight * length_m;
    TAtomicOperations::add(&counters->weightedSegmentCount_, weight);
    TAtomicOperations::add(&counters->trackLengthM_, length_m);
    TAtomicOperations::add(&counters->weightedTrackLengthM_, weighted_length_m);
    TAtomicOperations::add(&counters->energyWeightedTrackLengthGeVM_,
                           weighted_length_m * kinetic_energy_GeV);
    TAtomicOperations::maximum(&counters->maximumSegmentLengthM_, length_m);
    if (record.start_.pid_ ==
        static_cast<std::int32_t>(::corsika::kokkos::EmPid::Electron)) {
      TAtomicOperations::add(&counters->electronWeightedTrackLengthM_, weighted_length_m);
      TAtomicOperations::add(&counters->signedChargeWeightedTrackLengthM_,
                             -weighted_length_m);
    } else {
      TAtomicOperations::add(&counters->positronWeightedTrackLengthM_, weighted_length_m);
      TAtomicOperations::add(&counters->signedChargeWeightedTrackLengthM_,
                             weighted_length_m);
    }
    TAtomicOperations::add(
        &counters->weightedTrackLengthByKineticEnergyM_[diagnostic_energy_bin(
            kinetic_energy_GeV)],
        weighted_length_m);

    auto direction_dot = 0.;
    double direction_delta[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      auto const pre = record.start_.direction_[axis];
      auto const post = record.end_.direction_[axis];
      direction_dot += pre * post;
      direction_delta[axis] = post - pre;
    }
    auto const direction_change_rad =
        radio_acos(radio_min(1., radio_max(-1., direction_dot)));
    auto const time_residual_s = track.durationS_ - length_m / SpeedOfLightMPerS;
    TAtomicOperations::add(&counters->weightedDirectionChangeRad_,
                           weight * direction_change_rad);
    TAtomicOperations::add(&counters->weightedDirectionChangeSquaredRad2_,
                           weight * direction_change_rad * direction_change_rad);
    TAtomicOperations::add(&counters->weightedBetaDeficitTrackLengthM_,
                           weighted_length_m * (1. - track.betaModule_));
    TAtomicOperations::add(&counters->weightedTimeResidualS_, weight * time_residual_s);
    TAtomicOperations::maximum(&counters->maximumDirectionChangeRad_,
                               direction_change_rad);
    auto const charge_sign = record.start_.pid_ == static_cast<std::int32_t>(
                                                       ::corsika::kokkos::EmPid::Electron)
                                 ? -1.
                                 : 1.;
    for (int axis = 0; axis < 3; ++axis)
      TAtomicOperations::add(&counters->signedChargeWeightedDirectionChange_[axis],
                             charge_sign * weight * direction_delta[axis]);
    return true;
  }
} // namespace corsika::kokkos::radio
