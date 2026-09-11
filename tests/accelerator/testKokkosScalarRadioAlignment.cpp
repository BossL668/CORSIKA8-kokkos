/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Independent oracle: call the real scalar CoREAS/ZHS implementation and
// TimeDomainObserver, not a host invocation of RadioProjectionStep.hpp.
#include "ScalarRadioAlignmentDriver.hpp"

#include <corsika/accelerator/radio/common/RadioSnapshotBuilder.hpp>
#include <corsika/accelerator/radio/detail/RadioProjectionStep.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/propagators/TabulatedFlatAtmospherePropagator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using namespace corsika;
  namespace radio = corsika::accelerator::radio;
  namespace detail = radio::detail;
  using Interface =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using Model = UniformRefractiveIndex<
      MediumPropertyModel<UniformMagneticField<HomogeneousMedium<Interface>>>>;
  using Env = Environment<Model>;
  using Detector = ObserverCollection<TimeDomainObserver>;

  struct Particle {
    Point position;
    DirectionVector direction;
    TimeType time;
    Code pid{Code::Electron};
    double weight{1.};
    Point const& getPosition() const { return position; }
    DirectionVector const& getDirection() const { return direction; }
    TimeType getTime() const { return time; }
    HEPEnergyType getKineticEnergy() const { return 1_GeV; }
    Code getPID() const { return pid; }
    double getWeight() const { return weight; }
  };

  void require(bool value, std::string const& message) {
    if (!value) throw std::runtime_error(message);
  }

  double peak(std::vector<double> const& values) {
    double result{};
    for (auto value : values) result = std::max(result, std::abs(value));
    return result;
  }

  double compare(std::vector<double> const& scalar,
                 std::vector<double> const& accelerated,
                 double floor, std::string const& name) {
    require(scalar.size() == accelerated.size(), name + " sample count");
    double difference{};
    for (std::size_t i = 0; i < scalar.size(); ++i) {
      require(std::isfinite(scalar[i]) && std::isfinite(accelerated[i]),
              name + " nonfinite waveform");
      difference = std::max(difference, std::abs(scalar[i] - accelerated[i]));
    }
    auto const scale = peak(scalar);
    require(difference <= floor + 2.e-8 * scale,
            name + " differs from actual scalar radio implementation");
    return difference / std::max(scale, floor);
  }

  detail::DevicePropagation hostPropagation(gpu::radio::GpuRadioConfig const& c) {
    auto const& p = c.propagation;
    detail::DevicePropagation result{};
    result.minimum_height_m = p.minimum_height_m;
    result.maximum_height_m = p.maximum_height_m;
    result.step_m = p.step_m;
    result.inverse_step_per_m = p.inverse_step_per_m;
    result.slope_refractivity_lower = p.slope_refractivity_lower;
    result.slope_refractivity_upper = p.slope_refractivity_upper;
    result.slope_integrated_refractivity_lower = p.slope_integrated_refractivity_lower;
    result.slope_integrated_refractivity_upper = p.slope_integrated_refractivity_upper;
    result.refractivity = p.refractivity.data();
    result.integrated_refractivity = p.integrated_refractivity.data();
    result.table_size = p.refractivity.size();
    return result;
  }

  void testDoppler() {
    // The ordinary double sum loses 2^-60.  Scalar CoREAS's long-double
    // rescue and the device compensated arithmetic must both retain it.
    auto const result = scalar_radio_test::doppler();
    auto const oracle = static_cast<double>(1.L - (1.L + 0x1p-60L));
    require(result[0] == oracle && result[0] != 0., "zero Doppler rescue");
    require(result[1] == 0., "exact zero must not be epsilon-clamped");
    require(result[2] == 1. - 1.0003 * (0.7 * 0.6 + 0.2 * 0.8),
            "ordinary Doppler arithmetic changed");
  }

  void testObserverWindow() {
    Env environment;
    auto const cs = environment.getCoordinateSystem();
    Point const position(cs, 0_m, 0_m, 0_m);
    TimeDomainObserver observer("window", position, cs, 0_s, 4_ns, 1_GHz, 0_s);
    DirectionVector const emit(cs, {0., 0., 1.});
    corsika::SignalPath path(0_s, 1., 1., 1., emit, -emit, 1_m, {position});
    std::vector<double> const times{
        std::nextafter(0., -1.), 0., std::nextafter(0., 1.),
        std::nextafter(.5e-9, 0.), .5e-9, std::nextafter(.5e-9, 1.),
        std::nextafter(4.e-9, 0.), 4.e-9, std::nextafter(4.e-9, 1.)};
    for (std::size_t i = 0; i < times.size(); ++i) {
      observer.receive(times[i] * 1_s, path,
                       ElectricFieldVector(cs, static_cast<double>(1 << i) * 1_V / 1_m,
                                            0_V / 1_m, 0_V / 1_m));
    }
    auto const bins = observer.getWaveformX().size();
    detail::DeviceObserver snapshot{};
    snapshot.start_time_s = 0.; snapshot.duration_s = 4.e-9;
    snapshot.sample_rate_Hz = 1.e9; snapshot.number_of_bins = bins;
    auto const host = scalar_radio_test::observerWindow(times, snapshot);
    for (std::size_t i = 0; i < bins; ++i)
      require(host[i] == observer.getWaveformX()[i],
              "actual TimeDomainObserver boundary/rounding mismatch at bin " +
              std::to_string(i) + ": scalar=" + std::to_string(observer.getWaveformX()[i]) +
              " accelerated=" + std::to_string(host[i]));
  }

  void testPotentialWindow() {
    Env environment;
    auto const cs = environment.getCoordinateSystem();
    Point const position(cs, 0_m, 0_m, 0_m);
    TimeDomainObserver observer("potential_window", position, cs, 0_s, 4_ns, 1_GHz, 0_s);
    DirectionVector const emit(cs, {0., 0., 1.});
    SignalPath path(0_s, 1., 1., 1., emit, -emit, 1_m, {position});
    std::vector<double> const times{
        -.75e-9, std::nextafter(-.5e-9, -1.), -.5e-9,
        std::nextafter(-.5e-9, 0.), -.25e-9, 0.,
        std::nextafter(.5e-9, 0.), .5e-9, std::nextafter(.5e-9, 1.),
        4.e-9, 4.25e-9, std::nextafter(4.5e-9, 0.), 4.5e-9,
        std::nextafter(4.5e-9, 1.), 4.75e-9,
        -1.e100, 1.e100, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    for (std::size_t i = 0; i < times.size(); ++i)
      observer.receive(times[i] * 1_s, path,
          VectorPotential(cs, static_cast<double>(1 << i) * 1_V * 1_s / 1_m,
                          0_V * 1_s / 1_m, 0_V * 1_s / 1_m));
    detail::DeviceObserver snapshot{};
    snapshot.duration_s = 4.e-9;
    snapshot.sample_rate_Hz = 1.e9;
    snapshot.number_of_bins = observer.getWaveformX().size();
    auto const result = scalar_radio_test::observerWindow(times, snapshot, true);
    require(result == observer.getWaveformX(), "potential half-bin/invalid-time alignment");
    require((static_cast<unsigned>(result.front()) & (1 << 4)) != 0,
            "first potential bin must accept -0.25 ns");
    require((static_cast<unsigned>(result.back()) & (1 << 10)) != 0,
            "last potential bin must accept 4.25 ns");
    auto const axis = observer.getAxis();
    observer.reset();
    require(observer.getAxis() == axis && peak(observer.getWaveformX()) == 0.,
            "potential reset must preserve the output axis and clear every bin");
  }

  struct Case {
    char const* name;
    std::array<double, 3> start;
    std::array<double, 3> end;
    double beta;
    double start_time_s;
    double duration_s;
    double refractive_index;
    Code pid{Code::Electron};
    double weight{1.};
  };

  void runCase(Case const& test, bool deterministic, double table_step_m,
               bool tiled = false) {
    Env environment;
    auto const cs = environment.getCoordinateSystem();
    auto node = Env::createNode<Sphere>(
        Point(cs, 0_m, 0_m, 0_m), std::numeric_limits<double>::infinity() * 1_m);
    NuclearComposition composition({Code::Nitrogen}, {1.});
    node->setModelProperties<Model>(
        test.refractive_index, Medium::AirDry1Atm,
        MagneticFieldVector(cs, 0_T, 0_T, 0_T), 1.2_kg / cube(1_m), composition);
    environment.getUniverse()->addChild(std::move(node));

    Point const start(cs, test.start[0] * 1_m, test.start[1] * 1_m, test.start[2] * 1_m);
    Point const end(cs, test.end[0] * 1_m, test.end[1] * 1_m, test.end[2] * 1_m);
    auto const direction = (end - start).normalized();
    auto const dt = (end - start).getNorm() / (constants::c * test.beta);
    Particle particle{start, direction, test.start_time_s * 1_s, test.pid, test.weight};
    StraightTrajectory trajectory(Line(start, (end - start) / dt), dt);
    Step<Particle> step(particle, trajectory);

    Detector coreas_detector, zhs_detector;
    // Signed components, an off-axis observer, and a cone-near observer.
    for (int i = 0; i < 3; ++i) {
      Point const position(cs, (i == 0 ? 100. : i == 1 ? 17. : 1.) * 1_m,
                            (i == 0 ? 2. : i == 1 ? 31. : 0.) * 1_m, 0_m);
      TimeDomainObserver observer("observer_" + std::to_string(i), position, cs,
                                  0_s, test.duration_s * 1_s, 1_GHz, 0_s);
      coreas_detector.addObserver(observer);
      zhs_detector.addObserver(observer);
    }
    Point const upper(cs, 0_m, 0_m, 2000_m), lower(cs, 0_m, 0_m, 0_m);
    auto propagator = make_tabulated_flat_atmosphere_radio_propagator(
        environment, upper, lower, table_step_m * 1_m);
    CoREAS<Detector, decltype(propagator)> coreas(coreas_detector, propagator);
    ZHS<Detector, decltype(propagator)> zhs(zhs_detector, propagator);
    coreas.doContinuous(step, true);
    zhs.doContinuous(step, true);

    auto config = gpu::radio::makeGpuRadioConfig(
        environment, upper, lower, table_step_m * 1_m, coreas_detector, zhs_detector);
    config.deterministic = deterministic;
    using Record = gpu::em::LeptonTransportRecord;
    Record record{};
    record.start.pid = test.pid == Code::Electron
                           ? static_cast<int>(gpu::em::EmPid::Electron)
                           : static_cast<int>(gpu::em::EmPid::Positron);
    record.start.energy_GeV = 1. + gpu::em::TransportElectronMassGeV;
    record.start.weight = test.weight;
    record.start.time_s = step.getTimePre() / 1_s;
    record.end = record.start;
    record.end.time_s = step.getTimePost() / 1_s;
    auto const p0 = step.getPositionPre().getCoordinates(cs);
    auto const p1 = step.getPositionPost().getCoordinates(cs);
    auto const u = direction.getComponents(cs);
    for (int axis = 0; axis < 3; ++axis) {
      record.start.position_m[axis] = p0[axis] / 1_m;
      record.end.position_m[axis] = p1[axis] / 1_m;
      record.start.direction[axis] = record.end.direction[axis] = u[axis];
    }
    auto const waveforms = scalar_radio_test::project(config, record, tiled);
    if (std::string(test.name) == "ordinary" && table_step_m == 1.) {
      // Exercise project<true,false> and project<false,true>, not only the
      // fused project<true,true> entry. Compare to the fused result, which is
      // independently checked against the real scalar modules below.
      for (bool coreas_only : {true, false}) {
        auto single = config;
        single.coreas_enabled = coreas_only;
        single.zhs_enabled = !coreas_only;
        if (coreas_only) single.zhs_observers.clear();
        else single.coreas_observers.clear();
        auto const selected = scalar_radio_test::project(single, record, tiled);
        auto const& actual = coreas_only ? selected.coreas : selected.zhs;
        auto const& expected = coreas_only ? waveforms.coreas : waveforms.zhs;
        require((coreas_only ? selected.zhs : selected.coreas).empty(),
                "disabled radio algorithm produced observers");
        require(actual.size() == expected.size(), "single-algorithm observer count");
        auto const floor = coreas_only ? 2.e-18 : 2.e-27;
        for (std::size_t i = 0; i < actual.size(); ++i) {
          compare(expected[i].x, actual[i].x, floor, "single/fused x");
          compare(expected[i].y, actual[i].y, floor, "single/fused y");
          compare(expected[i].z, actual[i].z, floor, "single/fused z");
        }
        std::cout << (coreas_only ? "coreas_only" : "zhs_only")
                  << " deterministic=" << deterministic << " tiled=" << tiled
                  << " matches_fused=true\n";
      }
    }
    double coreas_error{}, zhs_error{};
    auto const& coreas_observers = coreas_detector.getObservers();
    auto const& zhs_observers = zhs_detector.getObservers();
    for (std::size_t i = 0; i < waveforms.coreas.size(); ++i) {
      auto const& ecpu = coreas_observers[i];
      auto const& acpu = zhs_observers[i];
      auto const& egpu = waveforms.coreas[i];
      auto const& agpu = waveforms.zhs[i];
      std::array<std::vector<double> const*, 3> c{&ecpu.getWaveformX(),
                                                &ecpu.getWaveformY(), &ecpu.getWaveformZ()};
      std::array<std::vector<double> const*, 3> g{&egpu.x, &egpu.y, &egpu.z};
      std::array<std::vector<double> const*, 3> a{&acpu.getWaveformX(),
                                                &acpu.getWaveformY(), &acpu.getWaveformZ()};
      std::array<std::vector<double> const*, 3> b{&agpu.x, &agpu.y, &agpu.z};
      for (int axis = 0; axis < 3; ++axis) {
        auto const label = std::string(test.name) + "/observer" + std::to_string(i);
        coreas_error = std::max(coreas_error, compare(*c[axis], *g[axis], 2.e-18, label));
        zhs_error = std::max(zhs_error, compare(*a[axis], *b[axis], 2.e-27, label));
        // Use exactly the output writer's adjacent-potential difference,
        // including the final bin. No window padding or endpoint smoothing.
        std::vector<double> cpu_field(a[axis]->size() - 1), gpu_field(cpu_field.size());
        for (std::size_t bin = 0; bin < cpu_field.size(); ++bin) {
          cpu_field[bin] = -((*a[axis])[bin + 1] - (*a[axis])[bin]) * 1.e9;
          gpu_field[bin] = -((*b[axis])[bin + 1] - (*b[axis])[bin]) * 1.e9;
        }
        compare(cpu_field, gpu_field, 4.e-18, label + " ZHS differentiated");
        if ((std::string(test.name) == "edge_first" ||
             std::string(test.name) == "edge_last") && i == 0) {
          require(peak(cpu_field) == 0. && peak(gpu_field) == 0.,
                  "constant-potential track must have zero E at every narrow-window bin");
        }
        if (std::string(test.name) == "window_end" && i == 0 && axis == 1) {
          auto const last = cpu_field.size() - 1;
          require(std::abs(cpu_field[last]) < 1.e-24,
                  "window-end fixture must not create a terminal pulse from missing A");
          require(std::abs(cpu_field[last - 1]) < 1.e-24,
                  "window-end fixture must be flat before the terminal pulse");
          std::cout << "window_end actual_scalar_ZHS_Ey_previous="
                    << cpu_field[last - 1] << " last=" << cpu_field[last]
                    << " Kokkos_last=" << gpu_field[last] << '\n';
        }
      }
    }

    // A genuine signal stays at the same time when the recording window is
    // extended. Compare against the actual scalar/GPU calculation in a wider
    // window, including both edge A bins and the final differentiated E bin.
    // The old time-label gate fails this check even though CPU and GPU agreed.
    for (int padding : {2, 17}) {
      Detector wide_coreas, wide_zhs;
      for (auto const& observer : zhs_observers) {
        TimeDomainObserver wide(observer.getName(), observer.getLocation(), cs,
            -padding * 1_ns, test.duration_s * 1_s + 2 * padding * 1_ns, 1_GHz, 0_s);
        wide_coreas.addObserver(wide);
        wide_zhs.addObserver(wide);
      }
      ZHS<Detector, decltype(propagator)> wide_process(wide_zhs, propagator);
      wide_process.doContinuous(step, true);
      auto wide_config = gpu::radio::makeGpuRadioConfig(
          environment, upper, lower, table_step_m * 1_m, wide_coreas, wide_zhs);
      wide_config.deterministic = deterministic;
      auto const wide_gpu = scalar_radio_test::project(wide_config, record, tiled);
      for (std::size_t i = 0; i < zhs_observers.size(); ++i) {
        auto const& narrow_cpu = zhs_observers[i];
        auto const& full_cpu = wide_zhs.getObservers()[i];
        auto const& narrow_gpu = waveforms.zhs[i];
        auto const& full_gpu = wide_gpu.zhs[i];
        std::array<std::vector<double> const*, 3> a{
            &narrow_cpu.getWaveformX(), &narrow_cpu.getWaveformY(), &narrow_cpu.getWaveformZ()};
        std::array<std::vector<double> const*, 3> b{
            &full_cpu.getWaveformX(), &full_cpu.getWaveformY(), &full_cpu.getWaveformZ()};
        std::array<std::vector<double> const*, 3> c{&narrow_gpu.x, &narrow_gpu.y, &narrow_gpu.z};
        std::array<std::vector<double> const*, 3> d{&full_gpu.x, &full_gpu.y, &full_gpu.z};
        for (int axis = 0; axis < 3; ++axis) {
          require(b[axis]->size() == a[axis]->size() + 2 * padding,
                  "window padding changed sample-grid shape");
          auto const count = a[axis]->size();
          std::vector<double> crop_cpu(b[axis]->begin() + padding,
                                      b[axis]->begin() + padding + count);
          std::vector<double> crop_gpu(d[axis]->begin() + padding,
                                      d[axis]->begin() + padding + count);
          auto const label = std::string(test.name) + " window-invariance";
          compare(*a[axis], crop_cpu, 2.e-27, label + " scalar A");
          compare(*c[axis], crop_gpu, 2.e-27, label + " Kokkos A");
          std::vector<double> e0(count - 1), e1(count - 1);
          for (std::size_t j = 0; j + 1 < count; ++j) {
            e0[j] = -((*a[axis])[j + 1] - (*a[axis])[j]) * 1.e9;
            e1[j] = -(crop_cpu[j + 1] - crop_cpu[j]) * 1.e9;
          }
          compare(e0, e1, 4.e-18, label + " scalar E");
          for (std::size_t j = 0; j + 1 < count; ++j) {
            e0[j] = -((*c[axis])[j + 1] - (*c[axis])[j]) * 1.e9;
            e1[j] = -(crop_gpu[j + 1] - crop_gpu[j]) * 1.e9;
          }
          compare(e0, e1, 4.e-18, label + " Kokkos E");
        }
      }
    }

    // Compare actual CPU tabulated propagation at all four branches,
    // including its legacy shadow-variable return values at/below min.
    auto const hp = hostPropagation(config);
    detail::DeviceObserver observer{};
    observer.position_m[0] = 100.; observer.position_m[1] = 2.;
    auto const destination = coreas_observers[0].getLocation();
    for (double z : {-1001., -1000., -999.25, 0., 2000., 2001.}) {
      Point const point(cs, 3_m, 4_m, z * 1_m);
      auto const expected = propagator.propagate(particle, point, destination).front();
      auto const got = detail::propagate(hp, {3., 4., z}, observer);
      require(got.refractive_index_source == expected.refractive_index_source_ &&
              got.refractive_index_destination == expected.refractive_index_destination_,
              "tabulated scalar index mismatch at source height " + std::to_string(z));
      require(std::abs(got.propagation_time_s - expected.propagation_time_ / 1_s) < 1.e-16,
              "tabulated propagation time mismatch");
    }
    std::cout << test.name << " deterministic=" << deterministic
              << " tiled=" << tiled << " step_m=" << table_step_m
              << " coreas_peak_relative=" << coreas_error
              << " zhs_potential_peak_relative=" << zhs_error << '\n';
  }
} // namespace

int main(int argc, char** argv) {
  scalar_radio_test::initialize(argc, argv);
  int status = 0;
  try {
    logging::set_level(logging::level::err);
    testDoppler();
    testObserverWindow();
    testPotentialWindow();
    double const light_speed = constants::c / (1_m / 1_s);
    double const half_length = .99 * light_speed * 20.e-9 / 2.;
    std::vector<Case> const cases{
        {"ordinary", {3., 4., 100.}, {4., 6., 99.}, .8, 0., 2.e-6, 1.0003},
        {"positron_weight", {3., 4., 100.}, {4., 6., 99.}, .8, 0., 2.e-6, 1.0003,
         Code::Positron, 37.},
        {"fraunhofer", {3., 4., 100.}, {4., 6., 10.}, .99, 0., 2.e-6, 1.0003},
        {"near_cone", {0., 0., 1000.}, {0., 0., 990.}, .999999, -3.e-6, 2.e-6, 1.0003},
        {"lower_edge", {3., 4., -1000.}, {4., 6., -1001.}, .8, 0., 8.e-6, 1.0003},
        {"late_track", {3., 4., 100.}, {4., 6., 99.}, .8, 1.e-3, 4.e-7, 1.0003},
        {"window_end", {3., 4., 10.}, {8., 4., 10.}, .8, 7.181319171927605e-8, 4.e-7, 1.0003},
        {"edge_first", {100. - half_length, 2., 1000.}, {100. + half_length, 2., 1000.},
         .99, -1000. / light_speed - 5.25e-9, 10.e-9, 1.},
        {"edge_last", {100. - half_length, 2., 1000.}, {100. + half_length, 2., 1000.},
         .99, -1000. / light_speed - 5.75e-9, 10.e-9, 1.},
        {"subbin_first", {100. - half_length / 100., 2., 1000.},
         {100. + half_length / 100., 2., 1000.}, .99,
         -1000. / light_speed - .35e-9, 10.e-9, 1.},
        {"subbin_last", {100. - half_length / 100., 2., 1000.},
         {100. + half_length / 100., 2., 1000.}, .99,
         -1000. / light_speed + 10.15e-9, 10.e-9, 1.},
        {"vacuum", {3., 4., 100.}, {4., 6., 99.}, .8, 0., 2.e-6, 1.}};
    for (auto const& test : cases)
      for (bool deterministic : {false, true}) runCase(test, deterministic, 1.);
    runCase(cases.front(), false, .5);
    runCase(cases.front(), false, 2.);
    runCase(cases.front(), true, 1., true);
    runCase(cases.front(), false, 1., true);
    runCase(cases[7], true, 1., true);
    runCase(cases[8], false, 1., true);
    std::cout << "PASS: real scalar CoREAS/ZHS, complete potential bins, widened-window "
                 "invariance, reset/reuse, and Doppler rescue\n";
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  scalar_radio_test::finalize();
  return status;
}
