/* Application adapter contracts: convex in-rock paths and raw E/A merging. */
#include <applications/detail/mountain/MountainRadio.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <catch2/catch_all.hpp>

using namespace corsika;
using namespace corsika::units::si;
using Catch::Approx;

TEST_CASE("Mountain homogeneous scalar radio and merge contract", "[mountain][radio]") {
  using namespace corsika::applications::mountain;
  using Interface = IRefractiveIndexModel<IMediumPropertyModel<IMediumModel>>;
  Environment<Interface> environment;
  auto const cs = environment.getCoordinateSystem();
  auto inside = [cs](Point const& point) {
    auto const values = point.getCoordinates(cs);
    return std::abs(values[0] / 1_m) < 100. &&
           std::abs(values[1] / 1_m) < 100. &&
           std::abs(values[2] / 1_m) < 100.;
  };
  HomogeneousMountainRadioPropagator<decltype(environment)> propagator(
      environment, 2., inside);
  Point const source(cs, 0_m, 0_m, 0_m);
  Point const horizontal(cs, 3_m, 4_m, 0_m);
  Point const inclined(cs, 3_m, 4_m, 12_m);
  CHECK(propagator.propagate(0, source, horizontal).at(0).propagation_time_ / 1_s ==
        Approx(10. / 299792458.).epsilon(1.e-14));
  CHECK(propagator.propagate(0, source, inclined).at(0).propagation_time_ / 1_s ==
        Approx(26. / 299792458.).epsilon(1.e-14));
  CHECK_THROWS(propagator.propagate(0, source, source));
  CHECK_THROWS(propagator.propagate(0, source, Point(cs, 200_m, 0_m, 0_m)));

  ObserverCollection<TimeDomainObserver> coreas, zhs;
  TimeDomainObserver observer("in_rock", horizontal, cs, 0_s, 1_us, 1_GHz, 0_s);
  coreas.addObserver(observer);
  zhs.addObserver(observer);
  auto const config = makeHomogeneousMountainRadioConfig(2., cs, coreas, zhs, inside);
  CHECK(config.enabled);
  CHECK(config.propagation.homogeneous_refractive_index == 2.);
  CHECK(config.propagation.refractivity.empty());
  CHECK(config.propagation.integrated_refractivity.empty());

  auto const size = coreas.getObservers().at(0).getWaveformX().size();
  gpu::radio::RadioWaveform field, potential;
  field.x.assign(size, 1.); field.y.assign(size, 2.); field.z.assign(size, 3.);
  potential.x.assign(size, 4.); potential.y.assign(size, 5.); potential.z.assign(size, 6.);
  gpu::radio::GpuRadioWaveforms waveforms{{field}, {potential}};
  mergeMountainGpuRadioWaveforms(waveforms, coreas, zhs);
  CHECK(coreas.getObservers().at(0).getWaveformX().at(0) == 1.);
  CHECK(zhs.getObservers().at(0).getWaveformX().at(0) == 4.);
  CHECK(zhs.getObservers().at(0).getWaveformZ().back() == 6.);
  waveforms.zhs.at(0).x.pop_back();
  CHECK_THROWS(mergeMountainGpuRadioWaveforms(waveforms, coreas, zhs));
  CHECK(coreas.getObservers().at(0).getWaveformX().at(0) == 1.);

  ObserverCollection<TimeDomainObserver> outside;
  TimeDomainObserver external("outside", Point(cs, 200_m, 0_m, 0_m), cs,
                              0_s, 1_us, 1_GHz, 0_s);
  outside.addObserver(external);
  CHECK_THROWS(makeHomogeneousMountainRadioConfig(2., cs, coreas, outside, inside));
}
