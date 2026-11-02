/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/radio/algorithms/TimeDomainEmission.hpp>
#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>
#include <corsika/kokkos/radio/firn/ExponentialProfile.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <Kokkos_Complex.hpp>
#include <catch2/catch_test_macros.hpp>
#include "NativeEmissionReference.hpp"
#include <corsika/kokkos/radio/firn/Fft.hpp>
#include <array>
#include <complex>
#include <iostream>

using namespace corsika::kokkos::radio::firn;

namespace {
  using TimeDomainRay = ::corsika::kokkos::radio::time_domain_detail::TimeDomainRay<
      Kokkos::complex<double>>;
  void ensure_runtime() { static corsika::kokkos::Runtime lifetime; }
  struct Capture {
    std::vector<std::array<std::complex<double>, 3>>& values_;
    void operator()(std::size_t bin, Kokkos::complex<double> const* values, int) const {
      for (int component = 0; component < 3; ++component)
        values_[bin][component] +=
            std::complex<double>(values[component].real(), values[component].imag());
    }
  };

  double band_error(GeometricalOpticsSpectra::components_type const& actual,
                    GeometricalOpticsSpectra::components_type const& expected,
                    FrequencyGrid const& grid) {
    double error = 0., scale = 0.;
    for (int component = 0; component < 3; ++component)
      for (std::size_t frequency = 1; frequency < grid.frequencyCount(); ++frequency) {
        if (grid.frequencyHz(frequency) < 150.e6 || grid.frequencyHz(frequency) > 720.e6)
          continue;
        error += std::norm(actual[component][frequency] - expected[component][frequency]);
        scale += std::norm(expected[component][frequency]);
      }
    REQUIRE(scale > 0.);
    return std::sqrt(error / scale);
  }
} // namespace

TEST_CASE("ZHS bin integrals retain the exact cone limit and clip window edges",
          "[firn][time-domain]") {
  double const interval = 1.e-9;
  double const area =
      corsika::kokkos::scalar_constants::ElementaryChargeC /
      (4. * pi * corsika::kokkos::scalar_constants::VacuumPermittivityFPerM * c_m_per_s *
       c_m_per_s);
  for (double duration : {0., 1.e-30, -1.e-30, .2e-9, -.2e-9, 3.e-9, -3.e-9}) {
    std::vector<std::array<std::complex<double>, 3>> samples(17);
    TimeDomainRay ray;
    ray.projectedDisplacement_[0] = Kokkos::complex<double>(1., .3);
    ray.arrivalTimeS_ = 8.1e-9;
    ray.retardedDurationS_ = duration;
    ::corsika::kokkos::radio::time_domain_detail::deposit_zhs(ray, 1., interval, 16,
                                                              Capture{samples});
    std::complex<double> integral{};
    for (auto const& sample : samples) integral += sample[0] * interval;
    CHECK(std::abs(integral / (area * std::complex<double>(1., .3)) - 1.) < 1.e-13);
  }
  std::vector<std::array<std::complex<double>, 3>> clipped(17);
  TimeDomainRay edge;
  edge.projectedDisplacement_[0] = Kokkos::complex<double>(1., 0.);
  edge.retardedDurationS_ = 2.e-9;
  edge.arrivalTimeS_ = -1.e-9;
  ::corsika::kokkos::radio::time_domain_detail::deposit_zhs(edge, 1., interval, 16,
                                                            Capture{clipped});
  CHECK(std::abs(clipped[0][0].real() * interval / area - .25) < 1.e-13);
}

TEST_CASE("CoREAS exact-cone endpoints remain finite", "[firn][time-domain]") {
  TimeDomainRay ray;
  ray.projectedDisplacement_[0] = Kokkos::complex<double>(1., .3);
  ray.arrivalTimeS_ = 8.1e-9;
  ray.retardedDurationS_ = 0.;
  std::vector<std::array<std::complex<double>, 3>> samples(16);
  ::corsika::kokkos::radio::time_domain_detail::deposit_coreas(
      ray, ray, ray, 1.e-9, 1., 1.e-9, 16, Capture{samples});
  std::complex<double> sum{};
  double norm = 0.;
  for (auto const& sample : samples) {
    CHECK(std::isfinite(sample[0].real()));
    sum += sample[0];
    norm += std::norm(sample[0]);
  }
  CHECK(std::abs(sum) == 0.);
  CHECK(norm > 0.);
}

TEST_CASE("CoREAS endpoint deposition preserves time moments on both sides of the cone",
          "[firn][time-domain]") {
  double const interval = 1.e-9;
  double const area =
      corsika::kokkos::scalar_constants::ElementaryChargeC /
      (4. * pi * corsika::kokkos::scalar_constants::VacuumPermittivityFPerM * c_m_per_s *
       c_m_per_s);
  for (double doppler : {-1.7, -.2, -.0011, -1.e-5, 0., 1.e-5, .0011, .2, 1.7})
    for (double midpoint : {7.1e-9, 7.95e-9, 8.e-9, 8.5e-9}) {
      TimeDomainRay middle;
      middle.projectedDisplacement_[0] = Kokkos::complex<double>(1., .3);
      middle.arrivalTimeS_ = midpoint;
      middle.retardedDurationS_ = doppler * 1.e-9;
      auto start = middle, end = middle;
      start.arrivalTimeS_ -= .5 * middle.retardedDurationS_;
      end.arrivalTimeS_ += .5 * middle.retardedDurationS_;
      std::vector<std::array<std::complex<double>, 3>> values(16);
      ::corsika::kokkos::radio::time_domain_detail::deposit_coreas(
          start, end, middle, 1.e-9, 1., interval, 16, Capture{values});
      std::complex<double> integral{}, moment{};
      for (std::size_t sample = 0; sample < values.size(); ++sample) {
        integral += values[sample][0] * interval;
        moment += values[sample][0] * interval * (sample * interval - midpoint);
      }
      CHECK(std::abs(integral * interval / area) < 1.e-11);
      CHECK(std::abs(moment / (area * std::complex<double>(1., .3)) - 1.) < 1.e-9);
    }
}

TEST_CASE("Dense-medium endpoint fallback agrees with a finite current near the cone",
          "[firn][ice][time-domain]") {
  double const index = 1.78, length = .01, duration = length / c_m_per_s;
  double const interval = 5.e-12, centerTime = 4.e-9;
  double const area =
      corsika::kokkos::scalar_constants::ElementaryChargeC /
      (4. * pi * corsika::kokkos::scalar_constants::VacuumPermittivityFPerM * c_m_per_s *
       c_m_per_s);
  for (double distance : {10., 1000.})
    for (double doppler : {-.05, -.005, 0., .005, .05}) {
      double const cosine = (1. - doppler) / index;
      double const x = distance * std::sqrt(1. - cosine * cosine);
      double const z = distance * cosine;
      auto ray = [&](double position) {
        double const r = std::hypot(x, z - position);
        double const kz = (z - position) / r, kx = x / r;
        TimeDomainRay result;
        result.projectedDisplacement_[0] = -length * kx * kz / r;
        result.projectedDisplacement_[2] = length * (1. - kz * kz) / r;
        result.arrivalTimeS_ =
            centerTime + position / c_m_per_s + index * (r - distance) / c_m_per_s;
        result.retardedDurationS_ = duration * (1. - index * kz);
        return result;
      };
      auto const middle = ray(0.);
      std::vector<std::array<std::complex<double>, 3>> samples(2048);
      ::corsika::kokkos::radio::time_domain_detail::deposit_coreas(
          ray(-.5 * length), ray(.5 * length), middle, duration, -1., interval,
          samples.size(), Capture{samples});
      for (double frequency : {100.e6, 300.e6, 500.e6}) {
        std::complex<double> actual{};
        for (std::size_t sample = 0; sample < samples.size(); ++sample)
          actual += samples[sample][0] * interval *
                    std::exp(std::complex<double>(
                        0., -2. * pi * frequency * (sample * interval - centerTime)));
        double const argument = pi * frequency * duration * doppler;
        double const sinc =
            std::abs(argument) < 1.e-12 ? 1. : std::sin(argument) / argument;
        auto const expected = std::complex<double>(0., 2. * pi * frequency * area) *
                              middle.projectedDisplacement_[0].real() * sinc;
        INFO("distance=" << distance << " D=" << doppler << " f=" << frequency);
        CHECK(std::abs(actual / expected - 1.) < .009);
      }
    }
}

TEST_CASE("Time-domain firn emission converges to the analytic finite-track spectrum",
          "[firn][time-domain]") {
  ensure_runtime();
  GeometricalOpticsConfig config;
  config.radio_.grid_ = {1.e-12, 262144};
  config.radio_.timeOriginS_ = 450.e-9;
  config.radio_.maximumLinearizationLengthM_ = .25;
  config.radio_.writeBranchComponents_ = true;
  config.batchSegments_ = 8;
  config.includeTurning_ = false;
  // Equation (1) in Windischhofer and Coleman, PoS(ICRC2025)1214.
  ExponentialProfile profile(1.78, .454, 1. / .0202, 0.);
  std::vector<vector_type> const observers{{70., 0., -100.}, {5., 3., -100.}};
  GeometricalOpticsTrack const track{
      {0., 0., -2.}, {0., 0., -2.01}, 0., .01 / c_m_per_s, -1.};
  config.radio_.emissionFormalism_ = EmissionFormalism::FiniteTrackSpectrum;
  auto spectrum =
      make_geometrical_optics_accumulator(profile.parameters(), observers, config);
  spectrum->append(track);
  auto const reference = spectrum->download();
  for (auto formalism : {EmissionFormalism::ZHS}) {
    config.radio_.emissionFormalism_ = formalism;
    auto device =
        make_geometrical_optics_accumulator(profile.parameters(), observers, config);
    device->append(track);
    auto const actual = device->download();
    for (std::size_t observer = 0; observer < observers.size(); ++observer)
      for (int branch : {0, 1}) {
        auto const error =
            band_error(actual.branches_[observer][branch],
                       reference.branches_[observer][branch], config.radio_.grid_);
        INFO("formalism=" << int(formalism) << " observer=" << observer
                          << " branch=" << branch << " L2=" << error);
        std::cout << "FIRN_BAND_L2 " << int(formalism) << ' ' << observer << ' ' << branch
                  << ' ' << error << '\n';
        CHECK(error < .035);
      }
    CHECK(device->statistics().inputSegments_ == 1);
    auto const repeat = device->download();
    CHECK(repeat.total_ == actual.total_);
    device->reset();
    double resetNorm = 0.;
    for (auto const& observer : device->download().total_)
      for (auto const& component : observer)
        for (auto const value : component) resetNorm += std::norm(value);
    CHECK(resetNorm == 0.);
    device->append(track);
    CHECK(device->download().total_ == actual.total_);
  }
  config.radio_.emissionFormalism_ = EmissionFormalism::CoREAS;
  auto firnDevice =
      make_geometrical_optics_accumulator(profile.parameters(), observers, config);
  auto firnHost = make_geometrical_optics_accumulator(profile.parameters(), observers,
                                                      config, 0, -1, true);
  firnDevice->append(track);
  firnHost->append(track);
  auto const deviceFirn = firnDevice->download();
  auto const hostFirn = firnHost->download();
  for (std::size_t observer = 0; observer < observers.size(); ++observer)
    for (int branch : {0, 1}) {
      auto const error =
          band_error(deviceFirn.branches_[observer][branch],
                     hostFirn.branches_[observer][branch], config.radio_.grid_);
      std::cout << "FIRN_COREAS_CPU_GPU " << observer << ' ' << branch << ' ' << error
                << '\n';
      CHECK(error < 2.e-7);
    }
  firnDevice.reset();
  firnHost.reset();
  // Original C8 modules are a separate oracle, not another copy of these kernels.
  config.radio_.grid_ = {1.e-11, 262144};
  config.radio_.timeOriginS_ = 0.;
  config.radio_.writeBranchComponents_ = false;
  config.includeSurfaceReflection_ = false;
  ExponentialProfile uniform(1.78, 0., 50., 0.);
  config.radio_.maximumLinearizationLengthM_ = 2.;
  std::vector<vector_type> const uniformObservers{{200., 0., -100.}, {200., 10., -110.}};
  // Use resolved pulses for the unmodified C8 nearest-bin oracle. Unresolved
  // pairs are tested independently above: matching their old binning is not a
  // correctness criterion for moment-preserving deposition.
  GeometricalOpticsTrack const resolvedTrack{
      {0., 0., -2.}, {0., 0., -3.}, 0., 1. / c_m_per_s, -1.};
  for (auto formalism : {EmissionFormalism::ZHS, EmissionFormalism::CoREAS}) {
    config.radio_.emissionFormalism_ = formalism;
    auto const& nativeTrack = formalism == EmissionFormalism::ZHS ? track : resolvedTrack;
    auto const native = firn_test::native_uniform_emission(nativeTrack, uniformObservers,
                                                           config.radio_, 1.78);
    auto device = make_geometrical_optics_accumulator(uniform.parameters(),
                                                      uniformObservers, config);
    device->append(nativeTrack);
    auto const actual = device->download();
    auto host = make_geometrical_optics_accumulator(
        uniform.parameters(), uniformObservers, config, 0, -1, true);
    host->append(nativeTrack);
    auto const cpu = host->download();
    for (std::size_t observer = 0; observer < uniformObservers.size(); ++observer) {
      double const nativeError = band_error(actual.total_[observer],
                                            native.total_[observer], config.radio_.grid_);
      double const deviceError =
          band_error(actual.total_[observer], cpu.total_[observer], config.radio_.grid_);
      std::cout << "FIRN_NATIVE_L2 " << int(formalism) << ' ' << observer << ' '
                << nativeError << ' ' << deviceError << '\n';
      CHECK(nativeError < (formalism == EmissionFormalism::ZHS ? 2.e-7 : .02));
      CHECK(deviceError < 2.e-7);
    }
  }
}
