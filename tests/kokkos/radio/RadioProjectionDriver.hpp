/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "RadioProjectionReference.hpp"
#include <corsika/kokkos/radio/RadioAtomicOperations.hpp>
#include <corsika/kokkos/radio/algorithms/RadioProjection.hpp>
#include <corsika/kokkos/radio/RadioConfig.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <vector>
#include <algorithm>
namespace radio_test {
  namespace cp = corsika::kokkos;
  namespace cr = cp::radio;
  namespace rr = radio_reference;
  using execution_type = Kokkos::DefaultExecutionSpace;
  using memory_type = execution_type::memory_space;
  template <class T>
  auto upload(std::vector<T> const& values) {
    Kokkos::View<T*, memory_type> view("radio input", values.size());
    auto host = Kokkos::create_mirror_view(view);
    for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
    Kokkos::deep_copy(view, host);
    return view;
  }
  template <class T>
  auto download(T const& view) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    std::vector<typename T::non_const_value_type> out(host.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = host(i);
    return out;
  }
  inline void ensure_runtime() { static cp::Runtime runtime; }

  struct Reference {
    using record_type = transport_reference::LeptonTransportRecord;
    using counter_type = rr::DeviceRadioCounters;
    using track_type = rr::RadioTrackKinematics;
    using propagation_type = rr::DevicePropagation;
    using observer_type = rr::DeviceObserver;
    using waveform_type = rr::DeviceWaveforms;
    using atomic_type = rr::KokkosRadioAtomicOperations;
    static record_type convert(cp::LeptonTransportRecord const& input) {
      record_type out;
      out.start.pid = input.start_.pid_;
      out.end.pid = input.end_.pid_;
      out.start.time_s = input.start_.timeS_;
      out.end.time_s = input.end_.timeS_;
      out.start.weight = input.start_.weight_;
      out.end.weight = input.end_.weight_;
      out.start.energy_GeV = input.start_.energyGeV_;
      out.end.energy_GeV = input.end_.energyGeV_;
      for (unsigned j = 0; j < 3; ++j) {
        out.start.position_m[j] = input.start_.positionM_[j];
        out.end.position_m[j] = input.end_.positionM_[j];
        out.start.direction[j] = input.start_.direction_[j];
        out.end.direction[j] = input.end_.direction_[j];
      }
      return out;
    }
    static propagation_type propagation(cr::RadioConfig const& c, double const* n,
                                        double const* integral) {
      auto const& p = c.propagation_;
      return {p.homogeneousRefractiveIndex_,
              p.minimumHeightM_,
              p.maximumHeightM_,
              p.stepM_,
              p.inverseStepPerM_,
              p.slopeRefractivityLower_,
              p.slopeIntegratedRefractivityLower_,
              p.slopeRefractivityUpper_,
              p.slopeIntegratedRefractivityUpper_,
              n,
              integral,
              p.refractivity_.size(),
              c.zhsSubtrackRefinement_};
    }
    static observer_type observer(cr::RadioObserverSnapshot const& p, std::size_t offset,
                                  bool potential, bool integer, double limit) {
      observer_type v;
      for (unsigned j = 0; j < 3; ++j) v.position_m[j] = p.positionM_[j];
      v.start_time_s = p.startTimeS_;
      v.duration_s = p.durationS_;
      v.sample_rate_Hz = p.sampleRateHz_;
      v.number_of_bins = p.numberOfBins_;
      v.waveform_offset = offset;
      v.fixed_point_scale =
          integer ? rr::FixedPointHeadroom * (potential ? p.sampleRateHz_ : 1.) / limit
                  : 1.;
      v.inverse_fixed_point_scale = 1. / v.fixed_point_scale;
      return v;
    }
    KOKKOS_FUNCTION static waveform_type waveform(double* d, long long* fixed,
                                                  std::size_t size, bool integer) {
      waveform_type w;
      if (integer) {
        w.fixed_x = fixed;
        w.fixed_y = fixed + size;
        w.fixed_z = fixed + 2 * size;
      } else {
        w.floating_x = d;
        w.floating_y = d + size;
        w.floating_z = d + 2 * size;
      }
      return w;
    }
    KOKKOS_FUNCTION static bool track(record_type const& input, track_type* out,
                                      counter_type* counters, bool diagnostics) {
      return rr::recordTrack<atomic_type>(input, out, counters, diagnostics);
    }
    KOKKOS_FUNCTION static void coreas(track_type const& t, propagation_type const& p,
                                       observer_type const& o, waveform_type const& w,
                                       counter_type* c) {
      rr::accumulateCoREAS<atomic_type>(t, p, o, w, c);
    }
    KOKKOS_FUNCTION static void zhs(track_type const& t, propagation_type const& p,
                                    observer_type const& o, waveform_type const& w,
                                    counter_type* c) {
      rr::accumulateZHS<atomic_type>(t, p, o, w, c);
    }
    KOKKOS_FUNCTION static void offset(observer_type& o, std::size_t n) {
      o.waveform_offset += n;
    }
  };

  struct Current {
    using record_type = cp::LeptonTransportRecord;
    using counter_type = cr::RadioCounters;
    using track_type = cr::RadioTrackKinematics;
    using propagation_type = cr::RadioPropagationView;
    using observer_type = cr::RadioObserverView;
    using waveform_type = cr::RadioWaveformView;
    using atomic_type = cr::RadioAtomicOperations;
    static record_type convert(cp::LeptonTransportRecord const& input) {
      record_type out;
      out.start_.pid_ = input.start_.pid_;
      out.end_.pid_ = input.end_.pid_;
      out.start_.timeS_ = input.start_.timeS_;
      out.end_.timeS_ = input.end_.timeS_;
      out.start_.weight_ = input.start_.weight_;
      out.end_.weight_ = input.end_.weight_;
      out.start_.energyGeV_ = input.start_.energyGeV_;
      out.end_.energyGeV_ = input.end_.energyGeV_;
      for (unsigned j = 0; j < 3; ++j) {
        out.start_.positionM_[j] = input.start_.positionM_[j];
        out.end_.positionM_[j] = input.end_.positionM_[j];
        out.start_.direction_[j] = input.start_.direction_[j];
        out.end_.direction_[j] = input.end_.direction_[j];
      }
      return out;
    }
    static propagation_type propagation(cr::RadioConfig const& c, double const* n,
                                        double const* integral) {
      auto const& p = c.propagation_;
      return {p.homogeneousRefractiveIndex_,
              p.minimumHeightM_,
              p.maximumHeightM_,
              p.stepM_,
              p.inverseStepPerM_,
              p.slopeRefractivityLower_,
              p.slopeIntegratedRefractivityLower_,
              p.slopeRefractivityUpper_,
              p.slopeIntegratedRefractivityUpper_,
              n,
              integral,
              p.refractivity_.size(),
              c.zhsSubtrackRefinement_};
    }
    static observer_type observer(cr::RadioObserverSnapshot const& p, std::size_t offset,
                                  bool potential, bool integer, double limit) {
      observer_type v;
      for (unsigned j = 0; j < 3; ++j) v.positionM_[j] = p.positionM_[j];
      v.startTimeS_ = p.startTimeS_;
      v.durationS_ = p.durationS_;
      v.sampleRateHz_ = p.sampleRateHz_;
      v.numberOfBins_ = p.numberOfBins_;
      v.waveformOffset_ = offset;
      v.fixedPointScale_ =
          integer ? cr::FixedPointHeadroom * (potential ? p.sampleRateHz_ : 1.) / limit
                  : 1.;
      v.inverseFixedPointScale_ = 1. / v.fixedPointScale_;
      return v;
    }
    KOKKOS_FUNCTION static waveform_type waveform(double* d, long long* fixed,
                                                  std::size_t size, bool integer) {
      waveform_type w;
      if (integer) {
        w.fixedX_ = fixed;
        w.fixedY_ = fixed + size;
        w.fixedZ_ = fixed + 2 * size;
      } else {
        w.floatingX_ = d;
        w.floatingY_ = d + size;
        w.floatingZ_ = d + 2 * size;
      }
      return w;
    }
    KOKKOS_FUNCTION static bool track(record_type const& input, track_type* out,
                                      counter_type* counters, bool diagnostics) {
      return cr::record_track<atomic_type>(input, out, counters, diagnostics);
    }
    KOKKOS_FUNCTION static void coreas(track_type const& t, propagation_type const& p,
                                       observer_type const& o, waveform_type const& w,
                                       counter_type* c) {
      cr::accumulate_co_reas<atomic_type>(t, p, o, w, c);
    }
    KOKKOS_FUNCTION static void zhs(track_type const& t, propagation_type const& p,
                                    observer_type const& o, waveform_type const& w,
                                    counter_type* c) {
      cr::accumulate_zhs<atomic_type>(t, p, o, w, c);
    }
    KOKKOS_FUNCTION static void offset(observer_type& o, std::size_t n) {
      o.waveformOffset_ += n;
    }
  };

  template <class TDriver>
  struct Result {
    std::vector<typename TDriver::counter_type> counters_;
    std::vector<typename TDriver::track_type> tracks_;
    std::vector<long long> coreasFixed_, zhsFixed_;
    std::vector<double> coreas_, zhs_;
  };
  template <class TDriver>
  Result<TDriver> project(std::vector<cp::LeptonTransportRecord> const& inputs,
                          cr::RadioConfig const& c, bool integer, bool diagnostics) {
    ensure_runtime();
    using d = TDriver;
    std::vector<typename d::record_type> converted;
    for (auto const& input : inputs) converted.push_back(d::convert(input));
    auto records = upload(converted);
    auto n = upload(c.propagation_.refractivity_);
    auto integral = upload(c.propagation_.integratedRefractivity_);
    auto p = d::propagation(c, n.data(), integral.data());
    std::vector<typename d::observer_type> e, a;
    std::size_t eSize = 0, aSize = 0;
    for (auto const& o : c.coreasObservers_) {
      e.push_back(d::observer(o, eSize, false, integer, c.fixedPointFieldLimitVPerM_));
      eSize += o.numberOfBins_;
    }
    for (auto const& o : c.zhsObservers_) {
      a.push_back(d::observer(o, aSize, true, integer, c.fixedPointFieldLimitVPerM_));
      aSize += o.numberOfBins_;
    }
    auto eObs = upload(e), aObs = upload(a);
    auto const totalE = eSize * inputs.size(), totalA = aSize * inputs.size();
    Kokkos::View<double*, memory_type> ef("fields", 3 * totalE),
        af("potentials", 3 * totalA);
    Kokkos::View<long long*, memory_type> ei("fixed fields", 3 * totalE),
        ai("fixed potentials", 3 * totalA);
    Kokkos::View<typename d::counter_type*, memory_type> counters("radio counters",
                                                                  inputs.size());
    Kokkos::View<typename d::track_type*, memory_type> tracks("radio kinematics",
                                                              inputs.size());
    bool const coreas = c.coreasEnabled_, zhs = c.zhsEnabled_;
    // Each input owns a waveform row. This compares projection expressions, not
    // the scheduling/accumulator implementation reserved for the following slice.
    Kokkos::parallel_for(
        "frozen radio projection", Kokkos::RangePolicy<execution_type>(0, inputs.size()),
        KOKKOS_LAMBDA(std::size_t i) {
          if (!d::track(records(i), tracks.data() + i, counters.data() + i, diagnostics))
            return;
          auto const ew = d::waveform(ef.data(), ei.data(), totalE, integer);
          auto const aw = d::waveform(af.data(), ai.data(), totalA, integer);
          if (coreas)
            for (std::size_t j = 0; j < eObs.extent(0); ++j) {
              auto o = eObs(j);
              d::offset(o, i * eSize);
              d::coreas(tracks(i), p, o, ew, counters.data() + i);
            }
          if (zhs)
            for (std::size_t j = 0; j < aObs.extent(0); ++j) {
              auto o = aObs(j);
              d::offset(o, i * aSize);
              d::zhs(tracks(i), p, o, aw, counters.data() + i);
            }
        });
    return Result<d>{download(counters), download(tracks), download(ei),
                     download(ai),       download(ef),     download(af)};
  }
} // namespace radio_test
