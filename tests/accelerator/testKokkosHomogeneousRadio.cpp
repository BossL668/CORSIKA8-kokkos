/* Exact homogeneous propagation is opt-in; the atmospheric path is unchanged. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionStep.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Execution = Kokkos::DefaultExecutionSpace;
using Memory = Execution::memory_space;
namespace detail = corsika::accelerator::radio::detail;
using Accumulator =
    corsika::accelerator::radio::kokkos_detail::KokkosRadioAccumulator<Execution>;

void require(bool const condition, char const* message) {
  if (!condition) throw std::runtime_error(message);
}

struct PropagationProbe {
  Kokkos::View<detail::Vec3*, Memory> sources;
  Kokkos::View<double*[3], Memory> results;
  detail::DevicePropagation propagation;
  detail::DeviceObserver observer;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t const i) const {
    auto const path = detail::propagate(propagation, sources(i), observer);
    results(i, 0) = path.propagation_time_s;
    results(i, 1) = path.refractive_index_source;
    results(i, 2) = path.refractive_index_destination;
  }
};

void checkPropagation() {
  std::array<detail::Vec3, 5> const locations{{
      {0., 0., 0.}, {10., 20., 30.}, {-100., -200., -300.},
      {0., 0., 0.001}, {1.e5, 2.e5, 3.e5}}};
  Kokkos::View<detail::Vec3*, Memory> sources("mountain_radio_sources", locations.size());
  auto host = Kokkos::create_mirror_view(sources);
  for (std::size_t i = 0; i < locations.size(); ++i) host(i) = locations[i];
  Kokkos::deep_copy(sources, host);
  Kokkos::View<double*[3], Memory> results("mountain_radio_times", locations.size());
  detail::DevicePropagation propagation;
  propagation.homogeneous_refractive_index = 2.;
  // All table pointers stay null. The constant-n path must never index them.
  detail::DeviceObserver observer;
  observer.position_m[0] = 30.;
  observer.position_m[1] = 40.;
  Kokkos::parallel_for("mountain_homogeneous_radio", locations.size(),
                      PropagationProbe{sources, results, propagation, observer});
  auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
  for (std::size_t i = 0; i < locations.size(); ++i) {
    auto const& p = locations[i];
    double const distance = std::hypot(std::hypot(p.x - 30., p.y - 40.), p.z);
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
  detail::DevicePropagation air;
  require(air.homogeneous_refractive_index == 0., "air no longer defaults to lookup");
  air.minimum_height_m = 0.;
  air.maximum_height_m = 10.;
  air.step_m = 1.;
  air.inverse_step_per_m = 1.;
  air.refractivity = refractivity.data();
  air.integrated_refractivity = integral.data();
  air.table_size = 11;
  detail::DeviceObserver observer;
  observer.position_m[0] = 30.;
  observer.position_m[1] = 40.;
  observer.position_m[2] = 3.;
  auto const horizontal = detail::propagate(air, {0., 0., 3.}, observer);
  // Preserve the released same-bin air behavior; do not opportunistically fix it.
  require(horizontal.propagation_time_s == 50. / detail::SpeedOfLightMPerS,
          "released same-bin atmospheric behavior changed");
  auto const inclined = detail::propagate(air, {0., 0., 5.}, observer);
  double const legacy = (1. + (integral[5] - integral[3]) / 2.) *
                         std::sqrt(2504.) / detail::SpeedOfLightMPerS;
  require(inclined.propagation_time_s == legacy,
          "released inclined atmospheric lookup changed");
}

void checkAccumulatorGates() {
  corsika::gpu::radio::GpuRadioConfig config;
  config.enabled = true;
  config.propagation.homogeneous_refractive_index = 2.;
  corsika::gpu::radio::RadioObserverSnapshot observer;
  observer.position_m[0] = 30.;
  observer.position_m[1] = 40.;
  observer.duration_s = 1.e-6;
  observer.sample_rate_Hz = 1.e9;
  observer.number_of_bins = 1000;
  config.coreas_observers.push_back(observer);
  config.zhs_observers.push_back(observer);
  Accumulator valid;
  valid.initialize(config);
  Accumulator::TrackInputView records("mountain_radio_test_track", 1);
  auto track = Kokkos::create_mirror_view(records);
  track(0) = {};
  track(0).start.pid =
      static_cast<std::int32_t>(corsika::gpu::em::EmPid::Electron);
  track(0).start.energy_GeV = 1.;
  track(0).start.weight = 1.;
  track(0).start.direction[0] = 1.;
  track(0).end = track(0).start;
  track(0).end.position_m[0] = 0.1;
  track(0).end.time_s = 0.1 / (0.99 * detail::SpeedOfLightMPerS);
  Kokkos::deep_copy(records, track);
  valid.accumulateLeptonTracks(records, 1);
  auto const waveforms = valid.download();
  require(waveforms.coreas.at(0).x.size() == 1000, "homogeneous waveform upload failed");
  bool nonzero = false;
  for (double const value : waveforms.zhs.at(0).x) {
    require(std::isfinite(value), "homogeneous GPU ZHS produced a nonfinite sample");
    nonzero = nonzero || value != 0.;
  }
  require(nonzero, "homogeneous device propagation produced no ZHS potential");
  for (double const invalid : {-1., std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
    config.propagation.homogeneous_refractive_index = invalid;
    bool threw = false;
    try { Accumulator rejected; rejected.initialize(config); }
    catch (std::invalid_argument const&) { threw = true; }
    require(threw, "invalid homogeneous refractive index accepted");
  }
  config.propagation.homogeneous_refractive_index = 0.;
  bool threw = false;
  try { Accumulator rejected; rejected.initialize(config); }
  catch (std::invalid_argument const&) { threw = true; }
  require(threw, "default air path accepted missing altitude table");
}
}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  try {
    checkPropagation();
    checkAirDefaultUnchanged();
    checkAccumulatorGates();
    std::cout << "homogeneous nR/c, air-default regression and initialization gates passed\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
