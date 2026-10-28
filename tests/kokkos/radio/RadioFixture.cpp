/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "RadioFixture.hpp"
#include "RadioSnapshotReference.hpp"
#include "ProductionObserverReference.hpp"
#include <corsika/kokkos/radio/RadioSnapshotBuilder.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/framework/geometry/SeparationPlane.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/modules/radio/CoREAS.hpp>
#include <corsika/modules/radio/ZHS.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <corsika/modules/radio/propagators/TabulatedFlatAtmospherePropagator.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <limits>
namespace radio_test {
  using namespace corsika;
  namespace cp = ::corsika::kokkos;
  namespace cr = cp::radio;
  using interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using Model = UniformRefractiveIndex<
      MediumPropertyModel<UniformMagneticField<HomogeneousMedium<interface_type>>>>;
  using Env = Environment<Model>;
  using Detector = ObserverCollection<TimeDomainObserver>;
  struct TestParticle {
    Point position_;
    DirectionVector direction_;
    TimeType time_;
    Code pid_{Code::Electron};
    double weight_{1.};
    Point const& getPosition() const { return position_; }
    DirectionVector const& getDirection() const { return direction_; }
    TimeType getTime() const { return time_; }
    HEPEnergyType getKineticEnergy() const { return 1_GeV; }
    Code getPID() const { return pid_; }
    double getWeight() const { return weight_; }
  };
  struct Case {
    char const* name;
    std::array<double, 3> start_;
    std::array<double, 3> end_;
    double beta_;
    double startTimeS_;
    double durationS_;
    double refractive_index;
    Code pid_{Code::Electron};
    double weight_{1.};
  };
  template <class TObserver>
  ScalarCase make_case(Case const& test, double table_step_m) {
    using TimeDomainObserver = TObserver;
    using Detector = ObserverCollection<TimeDomainObserver>;
    Env environment;
    auto const cs = environment.getCoordinateSystem();
    auto node = Env::createNode<Sphere>(Point(cs, 0_m, 0_m, 0_m),
                                        std::numeric_limits<double>::infinity() * 1_m);
    NuclearComposition composition({Code::Nitrogen}, {1.});
    node->setModelProperties<Model>(test.refractive_index, Medium::AirDry1Atm,
                                    MagneticFieldVector(cs, 0_T, 0_T, 0_T),
                                    1.2_kg / cube(1_m), composition);
    environment.getUniverse()->addChild(std::move(node));

    Point const start(cs, test.start_[0] * 1_m, test.start_[1] * 1_m,
                      test.start_[2] * 1_m);
    Point const end(cs, test.end_[0] * 1_m, test.end_[1] * 1_m, test.end_[2] * 1_m);
    auto const direction = (end - start).normalized();
    auto const dt = (end - start).getNorm() / (constants::c * test.beta_);
    TestParticle particle{start, direction, test.startTimeS_ * 1_s, test.pid_,
                          test.weight_};
    StraightTrajectory trajectory(Line(start, (end - start) / dt), dt);
    Step<TestParticle> step(particle, trajectory);

    Detector coreas_detector, zhs_detector;
    // Signed components, an off-axis observer, and a cone-near observer.
    for (int i = 0; i < 3; ++i) {
      Point const position(cs, (i == 0 ? 100. : i == 1 ? 17. : 1.) * 1_m,
                           (i == 0 ? 2. : i == 1 ? 31. : 0.) * 1_m, 0_m);
      TimeDomainObserver observer("observer_" + std::to_string(i), position, cs, 0_s,
                                  test.durationS_ * 1_s, 1_GHz, 0_s);
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

    auto config = cr::make_radio_config(environment, upper, lower, table_step_m * 1_m,
                                        coreas_detector, zhs_detector);
    config.deterministic_ = true;
    using Record = ::corsika::kokkos::LeptonTransportRecord;
    Record record{};
    record.start_.pid_ = test.pid_ == Code::Electron
                             ? static_cast<int>(::corsika::kokkos::EmPid::Electron)
                             : static_cast<int>(::corsika::kokkos::EmPid::Positron);
    record.start_.energyGeV_ = 1. + ::corsika::kokkos::TransportElectronMassGeV;
    record.start_.weight_ = test.weight_;
    record.start_.timeS_ = step.getTimePre() / 1_s;
    record.end_ = record.start_;
    record.end_.timeS_ = step.getTimePost() / 1_s;
    auto const p0 = step.getPositionPre().getCoordinates(cs);
    auto const p1 = step.getPositionPost().getCoordinates(cs);
    auto const u = direction.getComponents(cs);
    for (int axis = 0; axis < 3; ++axis) {
      record.start_.positionM_[axis] = p0[axis] / 1_m;
      record.end_.positionM_[axis] = p1[axis] / 1_m;
      record.start_.direction_[axis] = record.end_.direction_[axis] = u[axis];
    }

    ScalarCase out;
    out.name_ = std::string(test.name) + "/step=" + std::to_string(table_step_m);
    out.config_ = config;
    out.track_ = record;
    for (auto const& o : coreas_detector.getObservers())
      out.expected_.coreas_.push_back(
          {o.getWaveformX(), o.getWaveformY(), o.getWaveformZ()});
    for (auto const& o : zhs_detector.getObservers())
      out.expected_.zhs_.push_back(
          {o.getWaveformX(), o.getWaveformY(), o.getWaveformZ()});
    return out;
  }
  std::vector<ScalarCase> make_scalar_cases() {
    logging::set_level(logging::level::err);
    double const light_speed = constants::c / (1_m / 1_s);
    double const half_length = .99 * light_speed * 20.e-9 / 2.;
    std::vector<Case> const cases{
        {"ordinary", {3., 4., 100.}, {4., 6., 99.}, .8, 0., 2.e-6, 1.0003},
        {"positron_weight",
         {3., 4., 100.},
         {4., 6., 99.},
         .8,
         0.,
         2.e-6,
         1.0003,
         Code::Positron,
         37.},
        {"fraunhofer", {3., 4., 100.}, {4., 6., 10.}, .99, 0., 2.e-6, 1.0003},
        {"near_cone", {0., 0., 1000.}, {0., 0., 990.}, .999999, -3.e-6, 2.e-6, 1.0003},
        {"lower_edge", {3., 4., -1000.}, {4., 6., -1001.}, .8, 0., 8.e-6, 1.0003},
        {"late_track", {3., 4., 100.}, {4., 6., 99.}, .8, 1.e-3, 4.e-7, 1.0003},
        {"window_end",
         {3., 4., 10.},
         {8., 4., 10.},
         .8,
         7.181319171927605e-8,
         4.e-7,
         1.0003},
        {"edge_first",
         {100. - half_length, 2., 1000.},
         {100. + half_length, 2., 1000.},
         .99,
         -1000. / light_speed - 5.25e-9,
         10.e-9,
         1.},
        {"edge_last",
         {100. - half_length, 2., 1000.},
         {100. + half_length, 2., 1000.},
         .99,
         -1000. / light_speed - 5.75e-9,
         10.e-9,
         1.},
        {"subbin_first",
         {100. - half_length / 100., 2., 1000.},
         {100. + half_length / 100., 2., 1000.},
         .99,
         -1000. / light_speed - .35e-9,
         10.e-9,
         1.},
        {"subbin_last",
         {100. - half_length / 100., 2., 1000.},
         {100. + half_length / 100., 2., 1000.},
         .99,
         -1000. / light_speed + 10.15e-9,
         10.e-9,
         1.},
        {"vacuum", {3., 4., 100.}, {4., 6., 99.}, .8, 0., 2.e-6, 1.}};

    std::vector<ScalarCase> out;
    auto add = [&](Case const& c, double step) {
      auto current = make_case<radio_production_observer::TimeDomainObserver>(c, step);
      auto release = make_case<corsika::TimeDomainObserver>(c, step);
      current.release_ = release.expected_;
      out.push_back(std::move(current));
    };
    for (auto const& c : cases) add(c, 1.);
    add(cases.front(), .5);
    add(cases.front(), 2.);
    return out;
  }
  template <class TEnvironment>
  void check_snapshot(TEnvironment const& environment, Point const& upper,
                      Point const& lower, Detector const& d, double step) {
    namespace old = radio_snapshot_reference;
    for (int mode = 0; mode < 4; ++mode) {
      auto const a = cr::make_radio_config(environment, upper, lower, step * 1_m, d, d,
                                           mode & 1, mode & 2);
      auto const b = old::makeGpuRadioConfig(environment, upper, lower, step * 1_m, d, d,
                                             mode & 1, mode & 2);
      REQUIRE(a.enabled_ == b.enabled);
      REQUIRE(a.coreasEnabled_ == b.coreas_enabled);
      REQUIRE(a.zhsEnabled_ == b.zhs_enabled);
      REQUIRE(a.deterministic_ == b.deterministic);
      REQUIRE(a.fixedPointFieldLimitVPerM_ == b.fixed_point_field_limit_V_per_m);
      auto const& x = a.propagation_;
      auto const& y = b.propagation;
      REQUIRE(x.homogeneousRefractiveIndex_ == y.homogeneous_refractive_index);
      REQUIRE(x.minimumHeightM_ == y.minimum_height_m);
      REQUIRE(x.maximumHeightM_ == y.maximum_height_m);
      REQUIRE(x.stepM_ == y.step_m);
      REQUIRE(x.inverseStepPerM_ == y.inverse_step_per_m);
      REQUIRE(x.slopeRefractivityLower_ == y.slope_refractivity_lower);
      REQUIRE(x.slopeRefractivityUpper_ == y.slope_refractivity_upper);
      REQUIRE(x.slopeIntegratedRefractivityLower_ ==
              y.slope_integrated_refractivity_lower);
      REQUIRE(x.slopeIntegratedRefractivityUpper_ ==
              y.slope_integrated_refractivity_upper);
      REQUIRE(x.refractivity_ == y.refractivity);
      REQUIRE(x.integratedRefractivity_ == y.integrated_refractivity);

      REQUIRE(a.coreasObservers_.size() == b.coreas_observers.size());
      REQUIRE(a.zhsObservers_.size() == b.zhs_observers.size());
      if (mode & 1) {
        auto const& p = a.coreasObservers_.front();
        auto const& q = b.coreas_observers.front();
        for (unsigned i = 0; i < 3; ++i) REQUIRE(p.positionM_[i] == q.position_m[i]);
        REQUIRE(p.startTimeS_ == q.start_time_s);
        REQUIRE(p.durationS_ == q.duration_s);
        REQUIRE(p.sampleRateHz_ == q.sample_rate_Hz);
        REQUIRE(p.numberOfBins_ == q.number_of_bins);
      }
    }
  }
  TEST_CASE("Radio snapshots retain production atmosphere and independent firn exports",
            "[radio][snapshot]") {
    Env environment;
    auto const cs = environment.getCoordinateSystem();
    auto node = Env::createNode<Sphere>(Point(cs, 0_m, 0_m, 0_m), 1.e8_m);
    node->setModelProperties<Model>(
        1.0003, Medium::AirDry1Atm, MagneticFieldVector(cs, {0_T, 0_T, 0_T}),
        1.2_kg / cube(1_m), NuclearComposition({Code::Nitrogen}, {1.}));
    environment.getUniverse()->addChild(std::move(node));
    Detector d;
    d.addObserver(
        TimeDomainObserver("one", Point(cs, 1_m, 2_m, 3_m), cs, 0_s, 10_ns, 1_GHz, 0_s));
    TimeDomainObserver fractional("fractional", Point(cs, 1_m, 2_m, 3_m), cs, 7_ns,
                                  -10.25_ns, 1_GHz, 0_s);
    REQUIRE(fractional.getDuration() == 10.25_ns);
    Detector fractional_detector;
    fractional_detector.addObserver(fractional);
    auto fractional_snapshot = cr::make_radio_config(
        environment, Point(cs, 0_m, 0_m, 100_m), Point(cs, 0_m, 0_m, 0_m), 1_m,
        fractional_detector, fractional_detector);
    REQUIRE(fractional_snapshot.coreasObservers_.front().durationS_ ==
            fractional.getDuration() / 1_s);
    REQUIRE(fractional_snapshot.zhsObservers_.front().durationS_ ==
            fractional.getDuration() / 1_s);
    for (double step : {.5, 1., 2.})
      check_snapshot(environment, Point(cs, 0_m, 0_m, 100_m), Point(cs, 0_m, 0_m, 0_m), d,
                     step);
    REQUIRE_THROWS_AS(cr::make_radio_config(environment, Point(cs, 0_m, 0_m, 100_m),
                                            Point(cs, 0_m, 0_m, 0_m), 0_m, d, d),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(cr::make_radio_config(environment, Point(cs, 0_m, 0_m, -995_m),
                                            Point(cs, 0_m, 0_m, 0_m), 1_m, d, d),
                      std::invalid_argument);
    using firn_model_type =
        UniformMagneticField<MediumPropertyModel<ExponentialFirn<interface_type>>>;
    using firn_environment_type = Environment<interface_type>;
    firn_environment_type firn;
    auto const fc = firn.getCoordinateSystem();
    auto ice = firn_environment_type::createNode<SeparationPlane>(
        Plane(Point(fc, 0_m, 0_m, 0_m), DirectionVector(fc, {0., 0., 1.})));
    ice->setModelProperties<firn_model_type>(
        MagneticFieldVector(fc, {0_T, 0_T, 0_T}), Medium::WaterIce, fc,
        NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.}));
    firn.getUniverse()->addChild(std::move(ice));
    Detector fd;
    fd.addObserver(TimeDomainObserver("ice", Point(fc, 1_m, 2_m, -10_m), fc, 0_s, 10_ns,
                                      1_GHz, 0_s));
    check_snapshot(firn, Point(fc, 0_m, 0_m, -1_m), Point(fc, 0_m, 0_m, -10_m), fd, 1.);
    auto const exported = cr::make_radio_config(firn, Point(fc, 0_m, 0_m, -1_m),
                                                Point(fc, 0_m, 0_m, -10_m), 1_m, fd, fd);
    REQUIRE(exported.propagation_.refractivity_.front() !=
            exported.propagation_.refractivity_.back());
    // This checks the generic export, not the validity of a straight-ray model in firn.
  }
} // namespace radio_test
