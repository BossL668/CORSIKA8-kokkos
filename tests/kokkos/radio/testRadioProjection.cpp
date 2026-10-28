/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "RadioFixture.hpp"
#include "RadioProjectionDriver.hpp"
#include "RadioChecks.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <type_traits>

namespace rt = radio_test;
namespace cr = corsika::kokkos::radio;
namespace cp = corsika::kokkos;
namespace rr = radio_reference;

#if defined(CORSIKA_RADIO_EXPECT_CUDA)
static_assert(std::is_same_v<rt::execution_type, Kokkos::Cuda>);
#elif defined(CORSIKA_RADIO_EXPECT_OPENMP)
static_assert(std::is_same_v<rt::execution_type, Kokkos::OpenMP>);
#endif

TEST_CASE("Radio projection preserves production on the same execution space",
          "[radio][production]") {
  auto cases = rt::make_scalar_cases();
  REQUIRE(cases.size() == 14);
  std::size_t checked = 0;
  for (auto const& c : cases) {
    CAPTURE(c.name_);
    for (bool integer : {false, true}) {
      for (unsigned mode : {1, 2, 3}) {
        auto config = c.config_;
        config.coreasEnabled_ = mode & 1;
        config.zhsEnabled_ = mode & 2;
        auto current = rt::project<rt::Current>({c.track_}, config, integer, true);
        auto reference = rt::project<rt::Reference>({c.track_}, config, integer, true);
        rt::compare(current, reference);
        REQUIRE(current.counters_[0].fixedPointOverflows_ == 0);
        REQUIRE(current.counters_[0].validTracks_ == 1);
        if (!(mode & 1)) REQUIRE(current.counters_[0].coreasContributions_ == 0);
        if (!(mode & 2)) REQUIRE(current.counters_[0].zhsContributions_ == 0);
        ++checked;
      }
    }
  }
  REQUIRE(checked == 84);
  auto config = cases.front().config_;
  // Positive n selects uniform propagation; refinement remains an existing
  // validation control, not a newly introduced transport/ray approximation.
  for (double index : {1., 1.0003, 1.78}) {
    config.propagation_.homogeneousRefractiveIndex_ = index;
    config.zhsSubtrackRefinement_ = 3;
    rt::compare(rt::project<rt::Current>({cases.front().track_}, config, true, false),
                rt::project<rt::Reference>({cases.front().track_}, config, true, false));
  }
  std::vector<cp::LeptonTransportRecord> invalid(5, cases.front().track_);
  invalid[0].start_.pid_ = 22;
  invalid[1].start_.pid_ = 13;
  invalid[2].end_.timeS_ = invalid[2].start_.timeS_;
  invalid[3].end_.timeS_ = invalid[3].start_.timeS_ - 1.;
  for (unsigned i = 0; i < 3; ++i)
    invalid[4].end_.positionM_[i] = invalid[4].start_.positionM_[i];
  auto a = rt::project<rt::Current>(invalid, config, true, true);
  rt::compare(a, rt::project<rt::Reference>(invalid, config, true, true));
  for (auto const& counter : a.counters_) REQUIRE(counter.validTracks_ == 0);
}

TEST_CASE("Projected waveforms agree with the real C8 CoREAS and ZHS modules",
          "[radio][native]") {
  auto cases = rt::make_scalar_cases();
  std::size_t release_edge_differences = 0;
  for (auto const& c : cases) {
    CAPTURE(c.name_);
    for (bool integer : {false, true}) {
      auto const result = rt::project<rt::Current>({c.track_}, c.config_, integer, false);
      for (unsigned algorithm = 0; algorithm < 2; ++algorithm) {
        auto const& expected = algorithm ? c.expected_.zhs_ : c.expected_.coreas_;
        auto const& floating = algorithm ? result.zhs_ : result.coreas_;
        auto const& fixed = algorithm ? result.zhsFixed_ : result.coreasFixed_;
        auto const& release = algorithm ? c.release_.zhs_ : c.release_.coreas_;
        std::size_t size = floating.size() / 3, offset = 0, observer_index = 0;
        double const floor = algorithm ? 2.e-27 : 2.e-18;
        for (auto const& observer : expected) {
          std::array<std::vector<double> const*, 3> axes{&observer.x_, &observer.y_,
                                                         &observer.z_};
          for (unsigned axis = 0; axis < 3; ++axis) {
            auto const& scalar = *axes[axis];
            auto const& legacy_observer = release[observer_index];
            std::array<std::vector<double> const*, 3> legacy_axes{
                &legacy_observer.x_, &legacy_observer.y_, &legacy_observer.z_};
            auto const& legacy = *legacy_axes[axis];
            REQUIRE(scalar.size() == legacy.size());
            auto const& config_observer =
                algorithm ? c.config_.zhsObservers_[observer_index]
                          : c.config_.coreasObservers_[observer_index];
            double const inverse_scale =
                c.config_.fixedPointFieldLimitVPerM_ /
                (cr::FixedPointHeadroom *
                 (algorithm ? config_observer.sampleRateHz_ : 1.));
            double scale = 0., difference = 0.;
            for (std::size_t i = 0; i < scalar.size(); ++i) {
              auto const j = axis * size + offset + i;
              double const actual = integer ? fixed[j] * inverse_scale : floating[j];
              // The production observer fixes ZHS first/last potential bins.
              // Every other release scalar sample is still checked unchanged.
              if (algorithm && (i == 0 || i + 1 == scalar.size())) {
                if (scalar[i] != legacy[i]) ++release_edge_differences;
              } else {
                REQUIRE(scalar[i] == legacy[i]);
              }
              REQUIRE(std::isfinite(actual));
              scale = std::max(scale, std::abs(scalar[i]));
              difference = std::max(difference, std::abs(actual - scalar[i]));
            }
            CAPTURE(algorithm, integer, axis, scale, difference);
            // Same tolerance as the released scalar radio alignment test;
            // the separate production-vs-port test above is exact.
            REQUIRE(difference <= floor + 2.e-8 * scale);
          }
          offset += observer.x_.size();
          ++observer_index;
        }
        REQUIRE(offset == size);
      }
    }
  }
  REQUIRE(release_edge_differences > 0);
}

TEST_CASE("Radio time bins, overflow and Doppler recovery retain their contracts",
          "[radio][edge]") {
  rt::ensure_runtime();
  using mem = rt::memory_type;
  using exec = rt::execution_type;
  auto const inf = std::numeric_limits<double>::infinity();
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> times{-.75,
                            -.5,
                            std::nextafter(-.5, 0.),
                            -.25,
                            0.,
                            std::nextafter(.5, 0.),
                            .5,
                            std::nextafter(.5, 1.),
                            4.,
                            4.25,
                            std::nextafter(4.5, 0.),
                            4.5,
                            4.75,
                            -1.e100,
                            1.e100,
                            -inf,
                            inf,
                            nan};
  auto input = rt::upload(times);
  Kokkos::View<double*, mem> data("window", 4 * 3 * 5);
  Kokkos::View<cr::RadioCounters*, mem> counts("current counters", 2);
  Kokkos::View<rr::DeviceRadioCounters*, mem> oldCounts("reference counters", 2);
  Kokkos::View<double*, mem> values("checks", 6);
  Kokkos::View<long long*, mem> fixed("overflow", 3);
  Kokkos::View<unsigned*, mem> success("overflow status", 3);
  Kokkos::parallel_for(
      "radio edge cases", Kokkos::RangePolicy<exec>(0, 1), KOKKOS_LAMBDA(int) {
        cr::RadioObserverView current;
        current.durationS_ = 4.;
        current.sampleRateHz_ = 1.;
        current.numberOfBins_ = 5;
        rr::DeviceObserver reference;
        reference.duration_s = 4.;
        reference.sample_rate_Hz = 1.;
        reference.number_of_bins = 5;
        for (unsigned potential = 0; potential < 2; ++potential) {
          cr::RadioWaveformView w;
          w.floatingX_ = data.data() + 15 * potential;
          w.floatingY_ = w.floatingX_ + 5;
          w.floatingZ_ = w.floatingY_ + 5;
          rr::DeviceWaveforms old;
          old.floating_x = data.data() + 30 + 15 * potential;
          old.floating_y = old.floating_x + 5;
          old.floating_z = old.floating_y + 5;
          for (std::size_t i = 0; i < input.extent(0); ++i) {
            double const v = static_cast<double>(1 << i);
            if (potential) {
              cr::add_sample<cr::RadioAtomicOperations, true>(
                  current, w, input(i), {v, 0., 0.}, &counts(potential).zhsContributions_,
                  counts.data() + potential);
              rr::addSample<rr::KokkosRadioAtomicOperations, true>(
                  reference, old, input(i), {v, 0., 0.},
                  &oldCounts(potential).zhs_contributions, oldCounts.data() + potential);
            } else {
              cr::add_sample<cr::RadioAtomicOperations>(
                  current, w, input(i), {v, 0., 0.},
                  &counts(potential).coreasContributions_, counts.data() + potential);
              rr::addSample<rr::KokkosRadioAtomicOperations>(
                  reference, old, input(i), {v, 0., 0.},
                  &oldCounts(potential).coreas_contributions,
                  oldCounts.data() + potential);
            }
          }
        }
        values(0) = cr::co_reas_doppler(1., {1., 0x1p-30, 0.}, {1., 0x1p-30, 0.});
        values(1) = cr::co_reas_doppler(1., {1., 0., 0.}, {1., 0., 0.});
        values(2) = cr::observer_nearest_bin(0.5);
        values(3) = cr::observer_nearest_bin(0x1.fffffffffffffp-2);
        values(4) = cr::diagnostic_energy_bin(.001);
        values(5) = cr::diagnostic_energy_bin(101.);
        fixed(0) = cr::SignedIntegerMaximum;
        fixed(1) = cr::SignedIntegerMinimum;
        success(0) =
            cr::checked_radio_atomic_add<cr::RadioAtomicOperations>(fixed.data(), 1);
        success(1) =
            cr::checked_radio_atomic_add<cr::RadioAtomicOperations>(fixed.data() + 1, -1);
        success(2) = cr::add_radio_fixed_point<cr::RadioAtomicOperations>(
            fixed.data() + 2, inf, 1.);
      });
  auto bins = rt::download(data);
  for (unsigned i = 0; i < 30; ++i) REQUIRE(bins[i] == bins[30 + i]);
  REQUIRE((static_cast<unsigned>(bins[15]) & (1 << 3)) != 0);
  REQUIRE((static_cast<unsigned>(bins[19]) & (1 << 9)) != 0);
  auto v = rt::download(values);
  REQUIRE(v[0] == -0x1p-60);
  REQUIRE(v[1] == 0.);
  REQUIRE(v[2] == 1.);
  REQUIRE(v[3] == 0.);
  REQUIRE(v[4] == 0.);
  REQUIRE(v[5] == 14.);
  for (auto value : rt::download(success)) REQUIRE(value == 0);
}

TEST_CASE("Radio propagation preserves table edges and homogeneous mode",
          "[radio][propagation]") {
  rt::ensure_runtime();
  std::vector<double> n, integral;
  for (unsigned i = 0; i < 21; ++i) {
    n.push_back(.0003 * std::exp(-double(i) / 8.));
    integral.push_back((i ? integral.back() : 0.) + n.back());
  }
  auto dn = rt::upload(n), di = rt::upload(integral);
  auto heights = rt::upload(
      std::vector<double>{-12., -10., -9.9, -9.5, -9., 0., 9., 9.5, 10., 11., 100.});
  using mem = rt::memory_type;
  Kokkos::View<double**, mem> got("current paths", heights.extent(0), 8);
  Kokkos::View<double**, mem> expected("production paths", heights.extent(0), 8);
  for (double index : {0., 1., 1.78}) {
    Kokkos::parallel_for(
        "radio propagation edges",
        Kokkos::RangePolicy<rt::execution_type>(0, heights.extent(0)),
        KOKKOS_LAMBDA(std::size_t i) {
          cr::RadioPropagationView p{index,
                                     -10.,
                                     10.,
                                     1.,
                                     1.,
                                     (dn(10) - dn(0)) / 10.,
                                     (di(10) - di(0)) / 10.,
                                     (dn(20) - dn(10)) / 10.,
                                     (di(20) - di(10)) / 10.,
                                     dn.data(),
                                     di.data(),
                                     21,
                                     1};
          rr::DevicePropagation old{index,
                                    -10.,
                                    10.,
                                    1.,
                                    1.,
                                    (dn(10) - dn(0)) / 10.,
                                    (di(10) - di(0)) / 10.,
                                    (dn(20) - dn(10)) / 10.,
                                    (di(20) - di(10)) / 10.,
                                    dn.data(),
                                    di.data(),
                                    21,
                                    1};
          cr::RadioObserverView o;
          o.positionM_[0] = 100.;
          rr::DeviceObserver oo;
          oo.position_m[0] = 100.;
          auto a = cr::propagate(p, {3., 4., heights(i)}, o);
          auto b = rr::propagate(old, {3., 4., heights(i)}, oo);
          got(i, 0) = a.propagationTimeS_;
          expected(i, 0) = b.propagation_time_s;
          got(i, 1) = a.refractiveIndexSource_;
          expected(i, 1) = b.refractive_index_source;
          got(i, 2) = a.refractiveIndexDestination_;
          expected(i, 2) = b.refractive_index_destination;
          got(i, 3) = a.distanceM_;
          expected(i, 3) = b.distance_m;
          got(i, 4) = a.emit_.x_;
          expected(i, 4) = b.emit.x;
          got(i, 5) = a.emit_.y_;
          expected(i, 5) = b.emit.y;
          got(i, 6) = a.emit_.z_;
          expected(i, 6) = b.emit.z;
          got(i, 7) = index * a.distanceM_ / cr::SpeedOfLightMPerS;
        });
    auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, got);
    auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, expected);
    for (std::size_t i = 0; i < heights.extent(0); ++i) {
      for (unsigned j = 0; j < 7; ++j) REQUIRE(a(i, j) == b(i, j));
      if (index > 0.) REQUIRE(a(i, 0) == a(i, 7));
    }
  }
}
