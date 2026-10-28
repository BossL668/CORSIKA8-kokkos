/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "RadioAccumulatorChecks.hpp"
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
  using Execution = Kokkos::DefaultExecutionSpace;
  using Memory = Execution::memory_space;
  namespace detail = corsika::kokkos::radio;
  using Accumulator = corsika::kokkos::radio::RadioAccumulator<Execution>;

  void require(bool const condition, char const* message) {
    INFO(message);
    REQUIRE(condition);
  }

  struct PropagationProbe {
    Kokkos::View<detail::RadioVector*, Memory> sources;
    Kokkos::View<double * [3], Memory> results;
    detail::RadioPropagationView propagation;
    detail::RadioObserverView observer;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const i) const {
      auto const path = detail::propagate(propagation, sources(i), observer);
      results(i, 0) = path.propagationTimeS_;
      results(i, 1) = path.refractiveIndexSource_;
      results(i, 2) = path.refractiveIndexDestination_;
    }
  };

  void checkPropagation() {
    std::array<detail::RadioVector, 5> const locations{{{0., 0., 0.},
                                                        {10., 20., 30.},
                                                        {-100., -200., -300.},
                                                        {0., 0., 0.001},
                                                        {1.e5, 2.e5, 3.e5}}};
    Kokkos::View<detail::RadioVector*, Memory> sources("mountain_radio_sources",
                                                       locations.size());
    auto host = Kokkos::create_mirror_view(sources);
    for (std::size_t i = 0; i < locations.size(); ++i) host(i) = locations[i];
    Kokkos::deep_copy(sources, host);
    Kokkos::View<double * [3], Memory> results("mountain_radio_times", locations.size());
    detail::RadioPropagationView propagation;
    propagation.homogeneousRefractiveIndex_ = 2.;
    // All table pointers stay null. The constant-n path must never index them.
    detail::RadioObserverView observer;
    observer.positionM_[0] = 30.;
    observer.positionM_[1] = 40.;
    Kokkos::parallel_for("mountain_homogeneous_radio", locations.size(),
                         PropagationProbe{sources, results, propagation, observer});
    auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    for (std::size_t i = 0; i < locations.size(); ++i) {
      auto const& p = locations[i];
      double const distance = std::hypot(std::hypot(p.x_ - 30., p.y_ - 40.), p.z_);
      double const expected = 2. * distance / detail::SpeedOfLightMPerS;
      require(std::abs(actual(i, 0) / expected - 1.) < 4.e-15,
              "homogeneous horizontal/inclined nR/c mismatch");
      require(actual(i, 1) == 2. && actual(i, 2) == 2.,
              "homogeneous endpoint index mismatch");
    }
  }

  void checkAirDefaultUnchanged() {
    std::array<double, 11> refractivity{};
    std::array<double, 11> integral{};
    for (std::size_t i = 0; i < 11; ++i) {
      refractivity[i] = 0.0003;
      integral[i] = (i + 1) * 0.0003;
    }
    detail::RadioPropagationView air;
    require(air.homogeneousRefractiveIndex_ == 0., "air no longer defaults to lookup");
    air.minimumHeightM_ = 0.;
    air.maximumHeightM_ = 10.;
    air.stepM_ = 1.;
    air.inverseStepPerM_ = 1.;
    air.refractivity_ = refractivity.data();
    air.integratedRefractivity_ = integral.data();
    air.tableSize_ = 11;
    detail::RadioObserverView observer;
    observer.positionM_[0] = 30.;
    observer.positionM_[1] = 40.;
    observer.positionM_[2] = 3.;
    auto const horizontal = detail::propagate(air, {0., 0., 3.}, observer);
    // Preserve the released same-bin air behavior; do not opportunistically fix it.
    require(horizontal.propagationTimeS_ == 50. / detail::SpeedOfLightMPerS,
            "released same-bin atmospheric behavior changed");
    auto const inclined = detail::propagate(air, {0., 0., 5.}, observer);
    double const legacy = (1. + (integral[5] - integral[3]) / 2.) * std::sqrt(2504.) /
                          detail::SpeedOfLightMPerS;
    require(inclined.propagationTimeS_ == legacy,
            "released inclined atmospheric lookup changed");
  }

  void checkAccumulatorGates() {
    corsika::kokkos::radio::RadioConfig config;
    config.enabled_ = true;
    config.propagation_.homogeneousRefractiveIndex_ = 2.;
    corsika::kokkos::radio::RadioObserverSnapshot observer;
    observer.positionM_[0] = 30.;
    observer.positionM_[1] = 40.;
    observer.durationS_ = 1.e-6;
    observer.sampleRateHz_ = 1.e9;
    observer.numberOfBins_ = 1000;
    config.coreasObservers_.push_back(observer);
    config.zhsObservers_.push_back(observer);
    Accumulator valid;
    valid.initialize(config);
    Accumulator::track_input_view_type records("mountain_radio_test_track", 1);
    auto track = Kokkos::create_mirror_view(records);
    track(0) = {};
    track(0).start_.pid_ = static_cast<std::int32_t>(corsika::kokkos::EmPid::Electron);
    track(0).start_.energyGeV_ = 1.;
    track(0).start_.weight_ = 1.;
    track(0).start_.direction_[0] = 1.;
    track(0).end_ = track(0).start_;
    track(0).end_.positionM_[0] = 0.1;
    track(0).end_.timeS_ = 0.1 / (0.99 * detail::SpeedOfLightMPerS);
    Kokkos::deep_copy(records, track);
    valid.accumulateLeptonTracks(records, 1);
    auto const waveforms = valid.download();
    require(waveforms.coreas_.at(0).x_.size() == 1000,
            "homogeneous waveform upload failed");
    bool nonzero = false;
    for (double const value : waveforms.zhs_.at(0).x_) {
      require(std::isfinite(value), "homogeneous GPU ZHS produced a nonfinite sample");
      nonzero = nonzero || value != 0.;
    }
    require(nonzero, "homogeneous device propagation produced no ZHS potential");
    for (double const invalid : {-1., std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN()}) {
      config.propagation_.homogeneousRefractiveIndex_ = invalid;
      bool threw = false;
      try {
        Accumulator rejected;
        rejected.initialize(config);
      } catch (std::invalid_argument const&) { threw = true; }
      require(threw, "invalid homogeneous refractive index accepted");
    }
    config.propagation_.homogeneousRefractiveIndex_ = 0.;
    bool threw = false;
    try {
      Accumulator rejected;
      rejected.initialize(config);
    } catch (std::invalid_argument const&) { threw = true; }
    require(threw, "default air path accepted missing altitude table");
  }
} // namespace

TEST_CASE("Production homogeneous radio and air regression",
          "[kokkos][radio][accumulator]") {
  radio_accumulator_test::ensure_runtime();
  checkPropagation();
  checkAirDefaultUnchanged();
  checkAccumulatorGates();
}
