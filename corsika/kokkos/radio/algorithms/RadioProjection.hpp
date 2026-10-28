/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/radio/RadioProjectionData.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <cmath>

namespace corsika::kokkos::radio {
  CORSIKA_HOST_DEVICE inline double radio_abs(double value);
  CORSIKA_HOST_DEVICE inline double radio_sqrt(double value);
  CORSIKA_HOST_DEVICE inline double radio_floor(double value);
  CORSIKA_HOST_DEVICE inline double radio_ceil(double value);
  CORSIKA_HOST_DEVICE inline double radio_acos(double value);
  CORSIKA_HOST_DEVICE inline bool radio_finite(double value);
  CORSIKA_HOST_DEVICE inline long long radio_round(double value);
  CORSIKA_HOST_DEVICE inline double radio_min(double left, double right);
  CORSIKA_HOST_DEVICE inline double radio_max(double left, double right);
  CORSIKA_HOST_DEVICE inline double observer_nearest_bin(double scaled_time);
  CORSIKA_HOST_DEVICE inline RadioVector operator+(RadioVector a, RadioVector b);
  CORSIKA_HOST_DEVICE inline RadioVector operator-(RadioVector a, RadioVector b);
  CORSIKA_HOST_DEVICE inline RadioVector operator*(RadioVector a, double value);
  CORSIKA_HOST_DEVICE inline RadioVector operator/(RadioVector a, double value);
  CORSIKA_HOST_DEVICE inline double dot(RadioVector a, RadioVector b);
  CORSIKA_HOST_DEVICE inline RadioVector cross(RadioVector a, RadioVector b);
  CORSIKA_HOST_DEVICE inline double norm(RadioVector value);
  CORSIKA_HOST_DEVICE inline double co_reas_doppler(double index, RadioVector beta,
                                                    RadioVector emit);
  /** Prepare one electron/positron track without changing the transport record.
   * A nonpositive duration/length is rejected; valid input has finite coordinates.
   * The caller clears the output before use. No random numbers are consumed. */
  CORSIKA_HOST_DEVICE inline bool make_radio_track_kinematics(
      ::corsika::kokkos::LeptonTransportRecord const& record,
      RadioTrackKinematics& track);
  /** Evaluate the unchanged straight-ray production propagation model.
   * The source must differ from the observer. For a tabulated atmosphere the
   * observer's rounded bin must be in [0, tableSize_); source heights may use
   * the original endpoint extrapolation. Homogeneous mode needs no table. */
  CORSIKA_HOST_DEVICE inline RadioSignalPath propagate(RadioPropagationView const& table,
                                                       RadioVector source,
                                                       RadioObserverView const& observer);
  CORSIKA_HOST_DEVICE inline RadioVector transverse_endpoint(RadioVector emit,
                                                             RadioVector beta);
  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline bool checked_radio_atomic_add(long long* address,
                                                           long long increment);
  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline bool add_radio_fixed_point(long long* address,
                                                        double contribution,
                                                        double scale);
  /** Accumulate one sample; fixed-point overflow increments the diagnostic.
   * The caller treats an overflow as a failed waveform. Storage and counters
   * must be initialized and large enough for waveformOffset_ + numberOfBins_. */
  template <class TAtomicOperations, bool IsVectorPotential = false>
  CORSIKA_HOST_DEVICE inline void add_sample(RadioObserverView const& observer,
                                             RadioWaveformView const& waveforms,
                                             double time_s, RadioVector contribution,
                                             unsigned long long* contribution_counter,
                                             RadioCounters* counters);
  CORSIKA_HOST_DEVICE inline void separate_same_bin_endpoints(double& start_time,
                                                              double& end_time,
                                                              double grid_resolution,
                                                              bool signed_order);
  /** Add endpoint fields into caller-owned waveforms using the supplied atomic
   * operations. Preserve production Cherenkov handling and observer time gating. */
  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_co_reas(
      RadioTrackKinematics const& track, RadioPropagationView const& propagation,
      RadioObserverView const& observer, RadioWaveformView const& waveforms,
      RadioCounters* counters);
  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_zhs_segment(
      RadioVector point1, RadioVector point2, double time1, double time2,
      RadioVector beta, double constant, RadioPropagationView const& propagation,
      RadioObserverView const& observer, RadioWaveformView const& waveforms,
      RadioCounters* counters, bool subdivided);
  /** Add ZHS vector potentials with production Fraunhofer subdivision.
   * zhsSubtrackRefinement_ must be at least one. Differentiation belongs to output. */
  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline void accumulate_zhs(RadioTrackKinematics const& track,
                                                 RadioPropagationView const& propagation,
                                                 RadioObserverView const& observer,
                                                 RadioWaveformView const& waveforms,
                                                 RadioCounters* counters);
  CORSIKA_HOST_DEVICE inline std::size_t diagnostic_energy_bin(double kinetic_energy_GeV);
  /** Record the unchanged production track diagnostics. Projection is separate.
   * Diagnostic maxima require finite nonnegative values and zero initialization. */
  template <class TAtomicOperations>
  CORSIKA_HOST_DEVICE inline bool record_track(
      ::corsika::kokkos::LeptonTransportRecord const& record,
      RadioTrackKinematics* output, RadioCounters* counters, bool collect_diagnostics);
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/algorithms/RadioProjection.inl>
