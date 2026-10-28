/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

#include <boost/filesystem.hpp>

#include <corsika/modules/writers/EnergyLossWriter.hpp>

#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/ShowerAxis.hpp>

#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/CoordinateSystem.hpp>

#include <corsika/framework/core/Logging.hpp>

using namespace corsika;
using Catch::Approx;

const auto density = 1_kg / (1_m * 1_m * 1_m);

auto setupEnvironment(Code vTargetCode) {
  // setup environment, geometry
  auto env = std::make_unique<Environment<IMediumModel>>();
  auto& universe = *(env->getUniverse());
  const CoordinateSystemPtr& cs = env->getCoordinateSystem();

  auto theMedium = Environment<IMediumModel>::createNode<Sphere>(
      Point{cs, 0_m, 0_m, 0_m}, 1_km * std::numeric_limits<double>::infinity());

  using MyHomogeneousModel = HomogeneousMedium<IMediumModel>;
  theMedium->setModelProperties<MyHomogeneousModel>(
      density, NuclearComposition({vTargetCode}, {1.}));

  auto const* nodePtr = theMedium.get();
  universe.addChild(std::move(theMedium));

  return std::make_tuple(std::move(env), &cs, nodePtr);
}

class TestEnergyLoss : public corsika::EnergyLossWriter<> {
public:
  TestEnergyLoss(corsika::ShowerAxis const& axis)
      : EnergyLossWriter(axis) {}
};

TEST_CASE("EnergyLossWriter") {

  // best to run unit-tests with TRACE output
  logging::set_level(logging::level::trace);

  auto [env, csPtr, nodePtr] = setupEnvironment(Code::Nitrogen);
  auto const& cs = *csPtr;
  [[maybe_unused]] auto const& env_dummy = env;
  [[maybe_unused]] auto const& node_dummy = nodePtr;

  auto const observationHeight = 0_km;
  auto const injectionHeight = 20_km;
  auto const t = -observationHeight + injectionHeight;
  Point const showerCore{cs, 0_m, 0_m, observationHeight};
  Point const injectionPos = showerCore + DirectionVector{cs, {0, 0, 1}} * t;

  ShowerAxis const showerAxis{injectionPos, (showerCore - injectionPos), *env,
                              false, // -> throw exceptions
                              1000}; // -> number of bins

  std::string const outputDir = "./output_dir_eloss";

  // preparation
  if (boost::filesystem::exists(outputDir)) { boost::filesystem::remove_all(outputDir); }
  boost::filesystem::create_directory(outputDir);

  TestEnergyLoss test(showerAxis);
  test.startOfLibrary(outputDir);
  test.startOfShower(0);

  CHECK(test.getEnergyLost() / 1_GeV == Approx(0));
  test.write(100_g / square(1_cm), 110_g / square(1_cm), Code::Photon, 100_GeV);
  CHECK(test.getEnergyLost() / 1_GeV == Approx(100));
  test.write(Point(cs, {0_m, 0_m, observationHeight + 10_km}), Code::Proton, 100_GeV);
  CHECK(test.getEnergyLost() / 1_GeV == Approx(200));

  // generate straight simple track
  CoordinateSystemPtr rootCS = get_root_CoordinateSystem();
  Point r0(rootCS, {0_m, 0_m, 6.555_km});
  SpeedType const V0 = constants::c;
  VelocityVector v0(rootCS, {0_m / second, 0_m / second, -V0});
  Line const line(r0, v0);
  auto const time = 1000_ns;
  StraightTrajectory track(line, time);
  StraightTrajectory trackShort(line, time / 5e3);     // short
  StraightTrajectory trackPointLike(line, time / 1e7); // ultra short
  StraightTrajectory trackInverse({track.getPosition(1), -v0}, time);
  // test write
  test.write(track.getPosition(0), track.getPosition(1), Code::Proton, 100_GeV);
  test.write(trackInverse.getPosition(0), trackInverse.getPosition(1), Code::Proton,
             100_GeV); // equivalent
  test.write(trackShort.getPosition(0), trackShort.getPosition(1), Code::Proton,
             100_GeV); // this is in a single bin
  test.write(trackPointLike.getPosition(0), trackPointLike.getPosition(1), Code::Proton,
             100_GeV); // this is just a located point-like dE

  // incompatible binning
  CHECK_THROWS(test.write(100_g / square(1_cm), // extra line break by purpose
                          120_g / square(1_cm), Code::Photon, 100_GeV));
  test.write(100000_g / square(1_cm), 100010_g / square(1_cm), Code::Photon,
             100_GeV); // this doesn't throw, but it skips

  test.endOfShower(0);
  test.endOfLibrary();

  CHECK(boost::filesystem::exists(outputDir + "/dEdX.parquet"));

  auto const config = test.getConfig();
  CHECK(config["type"].as<std::string>() == "EnergyLoss");
  CHECK(config["units"]["energy"].as<std::string>() == "GeV");
  CHECK(config["units"]["grammage"].as<std::string>() == "g/cm^2");
  CHECK(config["bin-size"].as<double>() == 10.);
  CHECK(config["nbins"].as<int>() == 200);
  CHECK(config["grammage_threshold"].as<double>() == Approx(0.0001));

  auto const summary = test.getSummary();
  CHECK(summary["shower_0"]["sum_dEdX"].as<double>() == 600);

  // clean up
  if (boost::filesystem::exists(outputDir)) { boost::filesystem::remove_all(outputDir); }
}
