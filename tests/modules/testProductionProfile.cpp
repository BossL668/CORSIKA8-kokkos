/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/ProductionProfile.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/utility/CorsikaFenv.hpp>
#include <corsika/media/Environment.hpp>

#include <SetupTestStack.hpp>
#include <SetupTestTrajectory.hpp>
#include <SetupTestEnvironment.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

#include <catch2/catch_all.hpp>

using namespace corsika;

using DummyEnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using DummyEnvironment = Environment<DummyEnvironmentInterface>;

TEST_CASE("ProductionProfile", "process,secondary") {

  logging::set_level(logging::level::info);

  feenableexcept(FE_INVALID);
  using EnvType = DummyEnvironment;

  EnvType env;
  CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();

  // setup empty particle stack
  test::Stack stack;
  stack.clear();
  // two energies

  // list of arbitrary particles
  std::vector<Code> const particleList = {Code::PiPlus,   Code::PiMinus, Code::KPlus,
                                          Code::KMinus,   Code::K0Long,  Code::K0Short,
                                          Code::Electron, Code::MuPlus,  Code::NuE,
                                          Code::MuMinus,  Code::NuMu};

  // common stating point
  const Point point0(rootCS, 0_m, 0_m, 0_m);

  SECTION("muon production profile") {

    // production profile
    ProductionProfile prof;

    // add primary particle to stack
    auto particle = stack.addParticle(
        std::make_tuple(Code::Proton, 1_TeV, DirectionVector(rootCS, {1, 0, 0}),
                        Point(rootCS, 0_m, 0_m, 0_m), 0_ns));
    // view on secondary particles
    test::StackView view(particle);
    // ref. to primary particle through the secondary view.
    // only this way the secondary view is populated
    auto projectile = view.getProjectile();
    // add secondaries,
    // only cut is by species
    for (auto const proType : particleList)
      projectile.addSecondary(
          std::make_tuple(proType, 100_GeV, DirectionVector(rootCS, {1, 0, 0})));
    CHECK(view.getEntries() == 11);
    CHECK(stack.getEntries() == 12);

    prof.doSecondaries(view);

    // no change to view induced
    CHECK(view.getEntries() == 11);
    // only two muons were produced
    CHECK(prof.getCount() == 2);
  }
}

#include <corsika/modules/writers/ProductionWriter.hpp>

#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/ShowerAxis.hpp>

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/CoordinateSystem.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/core/Logging.hpp>

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

class TestProduction : public corsika::ProductionWriter<> {
public:
  TestProduction(corsika::ShowerAxis const& axis)
      : ProductionWriter(axis) {}
};

TEST_CASE("ProductionWriter") {

  logging::set_level(logging::level::trace);

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
  if (boost::filesystem::exists("./output_dir_prod")) {
    boost::filesystem::remove_all("./output_dir_prod");
  }
  boost::filesystem::create_directory("./output_dir_prod");

  TestProduction test(showerAxis);
  test.startOfLibrary("./output_dir_prod");
  test.startOfShower(0);

  // generate straight simple track
  CoordinateSystemPtr rootCS = get_root_CoordinateSystem();
  Point r0(rootCS, {0_km, 0_m, 8_km});

  test.write(r0, Code::Proton, 1.0);
  test.write(r0, Code::Photon, 1.0);
  test.write(r0, Code::PiPlus, 1.0);
  test.write(r0, Code::MuPlus, 1.0);
  test.write(r0, Code::Electron, 1.0);
  test.write(r0, Code::Positron, 1.0);

  test.endOfShower(0);
  test.endOfLibrary();

  CHECK(boost::filesystem::exists("./output_dir_prod/profile.parquet"));

  auto const config = test.getConfig();
  CHECK(config["type"].as<std::string>() == "ProductionProfile");
  CHECK(config["units"]["grammage"].as<std::string>() == "g/cm^2");
  CHECK(config["bin-size"].as<double>() == 10.);
  CHECK(config["nbins"].as<int>() ==
        static_cast<int>(showerAxis.getMaximumX() / (10_g / 1_cm / 1_cm)) + 1);

  auto const summary = test.getSummary(); // nothing to check yet
}
