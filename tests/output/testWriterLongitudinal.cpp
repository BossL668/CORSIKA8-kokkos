/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

#include <boost/filesystem.hpp>

#include <corsika/modules/writers/LongitudinalWriter.hpp>

#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/ShowerAxis.hpp>

#include <corsika/framework/geometry/StraightTrajectory.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/CoordinateSystem.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Logging.hpp>

#include <string>

using namespace corsika;
using Catch::Approx;

const auto density = 1_kg / (1_m * 1_m * 1_m);

auto setupEnvironment2(Code vTargetCode) {
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

class TestLongitudinal : public corsika::LongitudinalWriter<> {
public:
  TestLongitudinal(corsika::ShowerAxis const& axis)
      : LongitudinalWriter(axis) {}
};

TEST_CASE("LongitudinalWriter") {

  logging::set_level(logging::level::info);

  auto [env, csPtr, nodePtr] = setupEnvironment2(Code::Nitrogen);
  auto const& cs = *csPtr;
  [[maybe_unused]] auto const& env_dummy = env;
  [[maybe_unused]] auto const& node_dummy = nodePtr;

  auto const observationHeight = 0_km;
  auto const injectionHeight = 10_km;
  auto const t = -observationHeight + injectionHeight;
  Point const showerCore{cs, 0_m, 0_m, observationHeight};
  Point const injectionPos = showerCore + DirectionVector{cs, {0, 0, 1}} * t;

  ShowerAxis const showerAxis{injectionPos, (showerCore - injectionPos), *env,
                              false, // -> throw exceptions
                              1000}; // -> number of bins

  // preparation
  if (boost::filesystem::exists("./output_dir_long")) {
    boost::filesystem::remove_all("./output_dir_long");
  }
  boost::filesystem::create_directory("./output_dir_long");

  TestLongitudinal test(showerAxis);
  test.startOfLibrary("./output_dir_long");
  test.startOfShower(0);

  // generate straight simple track
  CoordinateSystemPtr rootCS = get_root_CoordinateSystem();
  Point r0(rootCS, {0_km, 0_m, 8_km});
  SpeedType const V0 = constants::c;
  VelocityVector v0(rootCS, {0_m / second, 0_m / second, -V0});
  Line const line(r0, v0);
  auto const time = 1000_ns;
  StraightTrajectory track(line, time);
  // test write
  test.write(track.getPosition(0), track.getPosition(1), Code::Proton, 1.0);
  test.write(track.getPosition(0), track.getPosition(1), Code::Photon, 1.0);
  test.write(track.getPosition(0), track.getPosition(1), Code::Electron, 1.0);
  test.write(track.getPosition(0), track.getPosition(1), Code::Positron, 1.0);
  test.write(track.getPosition(0), track.getPosition(1), Code::MuPlus, 1.0);
  test.write(track.getPosition(0), track.getPosition(1), Code::MuMinus, 1.0);

  test.write(10_g / square(1_cm), 20_g / square(1_cm), Code::PiPlus, 1.0);
  test.write(10_g / square(1_cm), 20_g / square(1_cm), Code::Electron, 1.0);
  test.write(10_g / square(1_cm), 20_g / square(1_cm), Code::Positron, 1.0);
  test.write(10_g / square(1_cm), 20_g / square(1_cm), Code::Photon, 1.0);
  test.write(10_g / square(1_cm), 20_g / square(1_cm), Code::MuPlus, 1.0);
  test.write(10_g / square(1_cm), 20_g / square(1_cm), Code::MuMinus, 1.0);

  // wrong binning
  CHECK_THROWS(test.write(10_g / square(1_cm), 10.1_g / square(1_cm), Code::PiPlus, 1.0));
  test.write(100000_g / square(1_cm), 100010_g / square(1_cm), Code::PiPlus,
             1.0); // this doesn't throw, it just skips

  test.endOfShower(0);
  test.endOfLibrary();

  CHECK(boost::filesystem::exists("./output_dir_long/profile.parquet"));

  auto const config = test.getConfig();
  CHECK(config["type"].as<std::string>() == "LongitudinalProfile");
  CHECK(config["units"]["grammage"].as<std::string>() == "g/cm^2");
  CHECK(config["bin-size"].as<double>() == 10.);
  CHECK(config["nbins"].as<int>() ==
        static_cast<int>(showerAxis.getMaximumX() / (10_g / 1_cm / 1_cm)) + 1);

  auto const summary = test.getSummary(); // nothing to check yet
}
