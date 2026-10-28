/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentPreparation.hpp>
#include <corsika/kokkos/environment/EnvironmentSnapshotData.hpp>
#include <corsika/kokkos/application/cascade/atmosphere/AtmospherePolicy.hpp>
#include <corsika/kokkos/radio/BorrowedRadioBackend.hpp>
#include <corsika/kokkos/radio/RadioSnapshotBuilder.hpp>
#include <corsika/kokkos/radio/ice/HomogeneousRadioConfig.hpp>
#include <corsika/kokkos/application/cascade/ice/HomogeneousIcePolicy.hpp>
#include <corsika/kokkos/application/cascade/firn/FirnPolicy.hpp>
#include <corsika/kokkos/application/cascade/SouthPoleSite.hpp>
#include <corsika/modules/radio/propagators/HomogeneousRadioPropagator.hpp>
#include <corsika/framework/geometry/Sphere.hpp>
#include <corsika/framework/geometry/Box.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/ExponentialFirn.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "RadioBuilderReference.hpp"

namespace {
  using namespace corsika;
  namespace cp = corsika::kokkos;
  using interface_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using environment_type = Environment<interface_type>;
  using uniform_type = UniformRefractiveIndex<
      MediumPropertyModel<UniformMagneticField<HomogeneousMedium<interface_type>>>>;
  using firn_type =
      UniformMagneticField<MediumPropertyModel<ExponentialFirn<interface_type>>>;
  unsigned checks = 0;
  void check(bool value) {
    ++checks;
    if (!value)
      throw std::runtime_error("scene contract check " + std::to_string(checks));
  }

  // Interface test double: checks lifetime/dispatch, does not simulate radiation.
  class BufferedProbe {
  public:
    std::vector<cp::LeptonTransportRecord> copied_;
    bool gpuRadioEnabled() const noexcept { return true; }
    void enqueueRadioTracks(boost::span<cp::LeptonTransportRecord const> records) {
      if (!records.empty()) copied_.insert(copied_.end(), records.begin(), records.end());
    }
    cp::radio::RadioWaveforms downloadRadioWaveforms() {
      cp::radio::RadioWaveforms result;
      result.coreas_.resize(1);
      for (auto const& record : copied_)
        result.coreas_[0].x_.push_back(record.distanceM_);
      copied_.clear();
      return result;
    }
  };
} // namespace

int main() {
  auto const frame = get_root_CoordinateSystem();
  {
    cp::application::SouthPoleSite site;
    auto const polar = site.coordinates(frame);
    auto const surface = site.surface(polar);
    auto const global = surface.getCoordinates(frame);
    check(std::abs(global.getX() / 1_m) < 1.e-8);
    check(std::abs(global.getY() / 1_m) < 1.e-8);
    check(std::abs(global.getZ() / 1_m + constants::EarthRadius::Mean / 1_m + 2835.) <
          1.e-8);
    auto const up = DirectionVector(polar, {0., 0., 1.}).getComponents(frame);
    check(std::abs(up[2].magnitude() + 1.) < 1.e-15);
    environment_type ice;
    auto volume = environment_type::createNode<Sphere>(
        Point(frame, 0_m, 0_m, 0_m), constants::EarthRadius::Mean + 8000_m);
    volume->setModelProperties<uniform_type>(
        1.78, Medium::WaterIce, MagneticFieldVector(polar, 0_T, 0_T, 0_T),
        .919_g / cube(1_cm),
        NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.}));
    ice.getUniverse()->addChild(std::move(volume));
    auto const snapshot = cp::prepare_environment_snapshot(ice, polar);
    check(cp::application::ice::require_homogeneous_ice(*snapshot) == 1.78);
    bool rejectedAsFirn = false;
    try {
      cp::application::firn::require_firn(*snapshot);
    } catch (std::invalid_argument const&) { rejectedAsFirn = true; }
    check(rejectedAsFirn);
    HomogeneousRadioPropagator<environment_type> propagator(ice);
    Point const source = surface + LengthVector(polar, {0_m, 0_m, -200_m});
    Point const observer = source + LengthVector(polar, {100_m, 0_m, -10_m});
    auto paths = propagator.propagate(0, source, observer);
    check(paths.size() == 1);
    check(paths[0].average_refractive_index_ == 1.78);
    auto const expected = 1.78 * (observer - source).getNorm() / constants::c;
    check(paths[0].propagation_time_ == expected);
    // Equivalent geometry expressed in Earth axes must produce the same delay.
    Point const earthSource(frame, source.getCoordinates(frame));
    Point const earthObserver(frame, observer.getCoordinates(frame));
    auto earthPaths = propagator.propagate(0, earthSource, earthObserver);
    check(std::abs((earthPaths[0].propagation_time_ - expected) / expected) < 1.e-10);
    ObserverCollection<TimeDomainObserver> detector;
    detector.addObserver(
        TimeDomainObserver("ice", observer, polar, 0_ns, 100_ns, 1_GHz, 0_ns));
    auto const radio = cp::radio::make_homogeneous_radio_config(1.78, polar, detector,
                                                                detector, true, true);
    check(radio.propagation_.homogeneousRefractiveIndex_ == 1.78);
    check(radio.propagation_.refractivity_.empty());
    auto const position = observer.getCoordinates(polar);
    for (int axis = 0; axis < 3; ++axis) {
      check(radio.coreasObservers_[0].positionM_[axis] == position[axis] / 1_m);
      check(snapshot->getModels()[0].magneticFieldT_[axis] == 0.);
    }
    bool refused = false;
    try {
      cp::application::atmosphere::require_air_materials(*snapshot);
    } catch (std::invalid_argument const&) { refused = true; }
    check(refused);
  }
  std::vector<Medium> const media{Medium::AirDry1Atm, Medium::WaterIce,
                                  Medium::StandardRock};
  std::vector<NuclearComposition> const composition{
      NuclearComposition({Code::Nitrogen, Code::Oxygen}, {.8, .2}),
      NuclearComposition({Code::Hydrogen, Code::Oxygen}, {2. / 3., 1. / 3.}),
      NuclearComposition({get_nucleus_code(28, 14), Code::Oxygen}, {1. / 3., 2. / 3.})};
  for (std::size_t i = 0; i < media.size(); ++i) {
    environment_type env;
    auto child = environment_type::createNode<Box>(frame, 10_m, 10_m, 10_m);
    child->setModelProperties<uniform_type>(
        1. + .2 * i, media[i], MagneticFieldVector(frame, 0_T, 0_T, 0_T),
        (i + 1.) * (1_g / cube(1_cm)), composition[i]);
    auto const* native = child.get();
    env.getUniverse()->addChild(std::move(child));
    auto snapshot = cp::prepare_environment_snapshot(env, frame);
    check(snapshot->getModels().size() == 1);
    check(snapshot->getModels()[0].mediumId_ == static_cast<int>(media[i]));
    check(snapshot->getModels()[0].referenceDensityGPerCm3_ == i + 1.);
    check(snapshot->getTopology().getNodeIndex(*(native)) == 1);
    check(snapshot->getComposition().size() == composition[i].getSize());
    bool refused = false;
    try {
      cp::application::atmosphere::require_air_materials(*snapshot);
    } catch (std::invalid_argument const&) { refused = true; }
    check(refused == (i != 0));
  }
  environment_type firn;
  auto ice = environment_type::createNode<Box>(frame, 10_m, 10_m, 10_m);
  ice->setModelProperties<firn_type>(MagneticFieldVector(frame, 0_T, 0_T, 0_T),
                                     Medium::WaterIce, frame, composition[1],
                                     ExponentialFirnProfile{});
  firn.getUniverse()->addChild(std::move(ice));
  auto snapshot = cp::prepare_environment_snapshot(firn, frame);
  check(snapshot->getModels()[0].densityKind_ ==
        cp::EnvironmentModel::Density::ExponentialFirn);
  check(snapshot->getModels()[0].mediumId_ == static_cast<int>(Medium::WaterIce));
  cp::application::firn::require_firn(*snapshot);

  for (auto const& coordinates :
       {frame, make_rotation(frame, QuantityVector<dimensionless_d>{1., 2., 3.}, .7)}) {
    Plane const native(Point(coordinates, 3_m, 4_m, 5_m),
                       DirectionVector(coordinates, {0., 0., 1.}));
    auto plane = cp::export_observation_plane(native, frame);
    auto const point = native.getCenter().getCoordinates(frame);
    auto const normal = native.getNormal().getComponents(frame);
    for (int j = 0; j < 3; ++j) {
      check(plane.pointM_[j] == point[j] / 1_m);
      check(plane.normal_[j] == normal[j].magnitude());
    }
    double start[3], direction[3];
    for (int j = 0; j < 3; ++j) {
      start[j] = plane.pointM_[j] + 2 * plane.normal_[j];
      direction[j] = -plane.normal_[j];
    }
    auto hit = cp::intersect_observation_plane_straight(plane, start, direction, 3.);
    check(hit.status_ == cp::ObservationPlaneStatus::Success);
    check(std::abs(hit.distanceM_ - 2.) < 1.e-14);
  }
  environment_type air;
  air.getUniverse()->setModelProperties<uniform_type>(
      1.000327, Medium::AirDry1Atm, MagneticFieldVector(frame, 0_T, 0_T, 0_T),
      1.2_kg / cube(1_m), composition[0]);
  ObserverCollection<TimeDomainObserver> detector;
  for (double step : {1., 2.5}) {
    auto original = scene_reference::make_radio_config(
        air, Point(frame, 0_m, 0_m, 100_m), Point(frame, 0_m, 0_m, 0_m), step * 1_m,
        detector, detector, true, true);
    auto current = cp::radio::atmosphere::make_radio_config(
        air, Point(frame, 0_m, 0_m, 100_m), Point(frame, 0_m, 0_m, 0_m), step * 1_m,
        detector, detector, true, true);
    check(current.propagation_.refractivity_ == original.propagation_.refractivity_);
    check(current.propagation_.integratedRefractivity_ ==
          original.propagation_.integratedRefractivity_);
    check(current.propagation_.slopeRefractivityUpper_ ==
          original.propagation_.slopeRefractivityUpper_);
    check(current.propagation_.slopeIntegratedRefractivityLower_ ==
          original.propagation_.slopeIntegratedRefractivityLower_);
  }
  BufferedProbe probe;
  cp::radio::BorrowedRadioBackend<BufferedProbe> adapter(probe);
  cp::radio::IRadioBackend& radio = adapter;
  check(radio.gpuRadioEnabled());
  cp::LeptonTransportRecord record;
  record.distanceM_ = 17.;
  radio.enqueueRadioTracks({&record, 1});
  record.distanceM_ = -1.;
  radio.enqueueRadioTracks({});
  auto result = radio.downloadRadioWaveforms();
  check(result.coreas_[0].x_ == std::vector<double>{17.});
  check(probe.copied_.empty());
  std::cout
      << "{\"passed\":true,\"checks\":" << checks
      << ",\"scope\":\"scene conversion and radio interface; no ray tracing claim\"}\n";
}
