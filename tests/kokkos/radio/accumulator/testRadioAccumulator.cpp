/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "RadioAccumulatorComparison.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>

namespace {
  using namespace radio_accumulator_test;

  cr::RadioConfig configuration(unsigned mask, std::size_t observers, bool separate,
                                bool integer, bool table = false) {
    cr::RadioConfig c;
    c.enabled_ = true;
    c.coreasEnabled_ = (mask & 1) != 0;
    c.zhsEnabled_ = (mask & 2) != 0;
    c.deterministic_ = integer;
    c.propagation_.homogeneousRefractiveIndex_ = table ? 0. : 1.0003;
    if (table) {
      auto& p = c.propagation_;
      p.minimumHeightM_ = 0.;
      p.maximumHeightM_ = 100.;
      p.stepM_ = 1.;
      p.inverseStepPerM_ = 1.;
      for (unsigned i = 0; i <= 100; ++i) {
        p.refractivity_.push_back(0.0003 * std::exp(-i / 100.));
        p.integratedRefractivity_.push_back(p.refractivity_.back() +
                                            (i ? p.integratedRefractivity_.back() : 0.));
      }
      p.slopeRefractivityLower_ = (p.refractivity_[10] - p.refractivity_[0]) / 10.;
      p.slopeRefractivityUpper_ = (p.refractivity_[100] - p.refractivity_[90]) / 10.;
      p.slopeIntegratedRefractivityLower_ =
          (p.integratedRefractivity_[10] - p.integratedRefractivity_[0]) / 10.;
      p.slopeIntegratedRefractivityUpper_ =
          (p.integratedRefractivity_[100] - p.integratedRefractivity_[90]) / 10.;
    }
    for (std::size_t i = 0; i < observers; ++i) {
      cr::RadioObserverSnapshot o;
      o.positionM_[0] = 30. + i;
      o.positionM_[1] = 40.;
      o.positionM_[2] = 3.;
      o.durationS_ = 4.e-6;
      o.sampleRateHz_ = 128.e6;
      o.numberOfBins_ = 512;
      if (c.coreasEnabled_) c.coreasObservers_.push_back(o);
      if (separate) o.positionM_[1] += 1.;
      if (c.zhsEnabled_) c.zhsObservers_.push_back(o);
    }
    return c;
  }

  record_type track(std::size_t i) {
    record_type r{};
    r.start_.pid_ = i % 2 ? -11 : 11;
    r.start_.energyGeV_ = 1. + .01 * (i % 7);
    r.start_.weight_ = 1.e-4 * (1 + i % 5);
    r.start_.direction_[0] = 1.;
    r.start_.positionM_[0] = .001 * (i % 37);
    r.start_.positionM_[1] = .001 * (i % 23);
    r.start_.positionM_[2] = 10. + .001 * (i % 11);
    r.start_.timeS_ = 1.e-9 * (i % 17);
    r.end_ = r.start_;
    if (i % 19 != 0) {
      auto length = .1 + .01 * (i % 7);
      r.end_.positionM_[0] += length;
      r.end_.timeS_ += length / (.99 * cr::SpeedOfLightMPerS);
    }
    return r;
  }

  template <class TLeft, class TRight>
  void check_components(TLeft const& a, TRight const& b, bool exact) {
    REQUIRE(a.size() == b.size());
    double peak{};
    for (auto v : b) peak = std::max(peak, std::abs(v));
    for (std::size_t i = 0; i < a.size(); ++i) {
      REQUIRE(std::isfinite(a[i]));
      REQUIRE(std::isfinite(b[i]));
      if (exact)
        CHECK(a[i] == b[i]);
      else
        CHECK(std::abs(a[i] - b[i]) <= 1.e-28 + 2.e-12 * peak);
    }
  }

  void check_waveforms(cr::RadioWaveforms const& a, rr::GpuRadioWaveforms const& b,
                       bool exact) {
    auto check = [&](auto const& x, auto const& y) {
      REQUIRE(x.size() == y.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
        check_components(x[i].x_, y[i].x, exact);
        check_components(x[i].y_, y[i].y, exact);
        check_components(x[i].z_, y[i].z, exact);
      }
    };
    check(a.coreas_, b.coreas);
    check(a.zhs_, b.zhs);
  }

  void compare(cr::RadioConfig const& c, std::size_t count,
               cp::TuningParameters tuning = {}, bool fixed_export = false) {
    execution_type ex;
    auto old_config = to_reference(c);
    auto old_tuning = to_reference(tuning);
    accumulator_type::track_input_view_type input("ported accumulator input", count);
    reference_type::TrackInputView original("frozen production input", count);
    auto a = Kokkos::create_mirror_view(input);
    auto b = Kokkos::create_mirror_view(original);
    for (std::size_t i = 0; i < count; ++i) {
      a(i) = track(i);
      b(i) = to_reference(a(i));
    }
    Kokkos::deep_copy(ex, input, a);
    Kokkos::deep_copy(ex, original, b);
    accumulator_type migrated;
    reference_type production;
    migrated.initialize(c, tuning, ex);
    production.initialize(old_config, old_tuning, ex);
    CHECK(migrated.deviceBytes() == production.deviceBytes());
    CHECK(migrated.staticHostToDeviceBytes() == production.staticHostToDeviceBytes());
    CHECK(accumulator_type::projectedDeviceBytes(c) ==
          reference_type::projectedDeviceBytes(old_config));
    auto new_projection = migrated.projectedTrackCapacity(count);
    auto old_projection = production.projectedTrackCapacity(count);
    CHECK(new_projection.retainedBytes_ == old_projection.retained_bytes);
    CHECK(new_projection.transientPeakBytes_ == old_projection.transient_peak_bytes);
    migrated.reserveTrackCapacity(count, ex);
    production.reserveTrackCapacity(count, ex);
    CHECK(migrated.deviceBytes() == new_projection.retainedBytes_);
    CHECK(migrated.deviceBytes() == production.deviceBytes());
    migrated.accumulateLeptonTracks(input, count, ex);
    production.accumulateLeptonTracks(original, count, ex);
    if (fixed_export) {
      auto actual = migrated.downloadFixed("production-equivalence",
                                           cp::CooperativeEndpoint::Cuda, ex);
      auto expected = production.downloadFixed("production-equivalence",
                                               ar::detail::CooperativeEndpoint::Cuda, ex);
      REQUIRE(actual.coreas_ == expected.coreas);
      REQUIRE(actual.zhs_ == expected.zhs);
      check_waveforms(cr::decode_fixed_radio(actual),
                      ar::detail::decodeFixedRadio(expected), true);
    } else
      check_waveforms(migrated.download(ex), production.download(ex), c.deterministic_);
    check_statistics(migrated.statistics(), production.statistics());
    if (c.trackDiagnostics_) {
      reference_type repeated;
      repeated.initialize(old_config, old_tuning, ex);
      repeated.reserveTrackCapacity(count, ex);
      repeated.accumulateLeptonTracks(original, count, ex);
      repeated.download(ex);
      double port_difference = 0., repeat_difference = 0.;
      auto const& baseline =
          production.statistics().weighted_track_length_by_kinetic_energy_m;
      auto const& twice = repeated.statistics().weighted_track_length_by_kinetic_energy_m;
      auto const& actual = migrated.statistics().weightedTrackLengthByKineticEnergyM_;
      for (std::size_t i = 0; i < baseline.size(); ++i) {
        CHECK(close(baseline[i], twice[i]));
        port_difference = std::max(port_difference, std::abs(actual[i] - baseline[i]));
        repeat_difference = std::max(repeat_difference, std::abs(twice[i] - baseline[i]));
      }
      std::cout << std::setprecision(17)
                << "floating diagnostic absolute differences: port=" << port_difference
                << " production-repeat=" << repeat_difference << '\n';
    }
    CHECK(migrated.statistics().leptonTracks_ > 0);
    CHECK(migrated.statistics().fixedPointOverflows_ == 0);
    CHECK_THROWS(migrated.download(ex));
    CHECK_THROWS(production.download(ex));
    CHECK_THROWS(migrated.initialize(c, tuning, ex));
    CHECK_THROWS(production.initialize(old_config, old_tuning, ex));
    ex.fence("complete frozen accumulator comparison");
  }
} // namespace

TEST_CASE("Radio accumulator preserves production fixed and floating output",
          "[kokkos][radio][accumulator]") {
  ensure_runtime();
  static_assert(sizeof(record_type) == sizeof(rr::gpu::em::LeptonTransportRecord));
  static_assert(sizeof(cr::RadioCounters) == sizeof(rr::DeviceRadioCounters));
  unsigned cases{};
  for (unsigned mask : {1u, 2u, 3u}) {
    for (auto count : {127u, 16385u}) {
      for (auto observers : {3u, 33u}) {
        INFO(mask);
        INFO(count);
        INFO(observers);
        compare(configuration(mask, observers, false, true), count, {}, true);
        ++cases;
      }
    }
  }
  auto c = configuration(3, 33, true, true, true);
  c.trackDiagnostics_ = true;
  c.zhsSubtrackRefinement_ = 3;
  compare(c, 137, {}, false);
  ++cases;
  for (bool integer : {false, true}) {
    auto conf = configuration(3, 5, true, integer);
    cp::TuningParameters tuning{};
    tuning.trackTileSize_ = 3;
    tuning.observerTileSize_ = 5;
    tuning.teamSize_ = 8;
    compare(conf, 16385, tuning, false);
    ++cases;
  }
  std::cout << "frozen production accumulator cases=" << cases << '\n';
  CHECK(cases == 15);
}

TEST_CASE("Radio accumulator capacity and rejection contracts remain unchanged",
          "[kokkos][radio][accumulator]") {
  ensure_runtime();
  execution_type ex;
  auto c = configuration(3, 3, false, true);
  accumulator_type a;
  reference_type b;
  a.initialize(c, {}, ex);
  b.initialize(to_reference(c), {}, ex);
  for (auto count : {0u, 1u, 16383u, 16384u, 16385u, 32771u, 7u}) {
    auto n = a.projectedTrackCapacity(count);
    auto o = b.projectedTrackCapacity(count);
    CHECK(n.retainedBytes_ == o.retained_bytes);
    CHECK(n.transientPeakBytes_ == o.transient_peak_bytes);
    a.reserveTrackCapacity(count, ex);
    b.reserveTrackCapacity(count, ex);
    CHECK(a.deviceBytes() == n.retainedBytes_);
    CHECK(a.deviceBytes() == b.deviceBytes());
  }
  accumulator_type::track_input_view_type empty("empty", 0);
  reference_type::TrackInputView old_empty("empty oracle", 0);
  CHECK_THROWS_AS(a.accumulateLeptonTracks(empty, 1, ex), std::out_of_range);
  CHECK_THROWS_AS(b.accumulateLeptonTracks(old_empty, 1, ex), std::out_of_range);
  CHECK_THROWS_AS(a.enqueueHostLeptonTracks(nullptr, 1, ex), std::invalid_argument);
  CHECK_THROWS_AS(b.enqueueHostLeptonTracks(nullptr, 1, ex), std::invalid_argument);
  CHECK_NOTHROW(a.enqueueHostLeptonTracks(nullptr, 0, ex));
  CHECK_NOTHROW(b.enqueueHostLeptonTracks(nullptr, 0, ex));
  auto r = track(1);
  auto old_r = to_reference(r);
  a.enqueueHostLeptonTracks(&r, 1, ex);
  b.enqueueHostLeptonTracks(&old_r, 1, ex);
  CHECK_THROWS_AS(a.reset(ex), std::logic_error);
  CHECK_THROWS_AS(b.reset(ex), std::logic_error);
  check_waveforms(a.download(ex), b.download(ex), true);
  CHECK_THROWS_AS(a.enqueueHostLeptonTracks(&r, 1, ex), std::logic_error);
  CHECK_THROWS_AS(b.enqueueHostLeptonTracks(&old_r, 1, ex), std::logic_error);
  a.reset(ex);
  b.reset(ex);
  check_waveforms(a.download(ex), b.download(ex), true);
  check_statistics(a.statistics(), b.statistics());
  for (unsigned variant = 0; variant < 7; ++variant) {
    auto invalid = c;
    switch (variant) {
      case 0:
        invalid.coreasEnabled_ = invalid.zhsEnabled_ = false;
        break;
      case 1:
        invalid.fixedPointFieldLimitVPerM_ = 0.;
        break;
      case 2:
        invalid.zhsSubtrackRefinement_ = 0;
        break;
      case 3:
        invalid.zhsSubtrackRefinement_ = 65;
        break;
      case 4:
        invalid.coreasObservers_[0].numberOfBins_ = 0;
        break;
      case 5:
        invalid.zhsObservers_[0].sampleRateHz_ = -1.;
        break;
      case 6:
        invalid.coreasObservers_[0].positionM_[0] =
            std::numeric_limits<double>::infinity();
        break;
    }
    CHECK_THROWS_AS(accumulator_type::projectedDeviceBytes(invalid),
                    std::invalid_argument);
    CHECK_THROWS_AS(reference_type::projectedDeviceBytes(to_reference(invalid)),
                    std::invalid_argument);
  }
  auto disabled = c;
  disabled.enabled_ = false;
  CHECK(accumulator_type::projectedDeviceBytes(disabled) == 0);
  accumulator_type off;
  off.initialize(disabled, {}, ex);
  CHECK_FALSE(off.enabled());
  CHECK(off.deviceBytes() == 0);
  CHECK(off.download(ex).coreas_.empty());
  cp::TuningParameters invalid_tuning;
  invalid_tuning.teamSize_ = 1025;
  accumulator_type bad;
  reference_type old_bad;
  CHECK_THROWS_AS(bad.initialize(c, invalid_tuning, ex), std::invalid_argument);
  CHECK_THROWS_AS(old_bad.initialize(to_reference(c), to_reference(invalid_tuning), ex),
                  std::invalid_argument);
}
