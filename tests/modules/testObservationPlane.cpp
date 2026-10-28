/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <catch2/catch_all.hpp>

#include <corsika/modules/ObservationPlane.hpp>

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>

#include <SetupTestEnvironment.hpp>
#include <SetupTestStack.hpp>
#include <SetupTestTrajectory.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

using namespace corsika;
using Catch::Approx;

TEST_CASE("ObservationPlane", "interface") {

  logging::set_level(logging::level::info);

  auto [env, csPtr, nodePtr] = setup::testing::setup_environment(Code::Oxygen);
  auto const& cs = *csPtr;
  [[maybe_unused]] auto const& env_dummy = env;
  [[maybe_unused]] auto const& node_dummy = nodePtr;

  /*
    Test with 1_GeV neutrino, starting at 0,0,0 travelling into +x direction (see
    setup_stack).

    ObservationPlane has origin at 10,0,0 and a normal in x-direction
   */
  auto [stack, viewPtr] = setup::testing::setup_stack(Code::NuE, 1_GeV, nodePtr, cs);
  [[maybe_unused]] test::StackView& view = *viewPtr;
  auto particle = stack->getNextParticle();

  // dummy track. Not used for calculation!
  Point const start(cs, {0_m, 1_m, 10_m});
  VelocityVector vec(cs, constants::c, 0_m / second, 0_m / second);
  Line line(start, vec);
  setup::Trajectory no_used_track =
      setup::testing::make_track<setup::Trajectory>(line, 12_m / constants::c);

  SECTION("horizontal plane") {

    Plane const obsPlane(Point(cs, {10_m, 0_m, 0_m}), DirectionVector(cs, {1., 0., 0.}));
    ObservationPlane<setup::Tracking, WriterOff> obs(obsPlane,
                                                     DirectionVector(cs, {0., 1., 0.}));

    Step step(particle, no_used_track);
    LengthType const length = obs.getMaxStepLength(particle, no_used_track);
    ProcessReturn const ret = obs.doContinuous(step, true);

    CHECK(length / 10_m == Approx(1).margin(1e-4));
    CHECK(ret == ProcessReturn::ParticleAbsorbed);

    // particle does not reach plane:
    {
      setup::Trajectory no_hit_track =
          setup::testing::make_track<setup::Trajectory>(line, 1_nm / constants::c);
      LengthType const no_hit = obs.getMaxStepLength(particle, no_hit_track);
      CHECK(no_hit == std::numeric_limits<double>::infinity() * 1_m);
    }

    // particle past plane:
    {
      particle.setPosition({cs, {11_m, 0_m, -1_m}});
      setup::Trajectory no_hit_track =
          setup::testing::make_track<setup::Trajectory>(line, 1_nm / constants::c);
      LengthType const no_hit = obs.getMaxStepLength(particle, no_hit_track);
      CHECK(no_hit == std::numeric_limits<double>::infinity() * 1_m);
    }
  }

  SECTION("transparent plane") {
    Plane const obsPlane(Point(cs, {1_m, 0_m, 0_m}), DirectionVector(cs, {1., 0., 0.}));
    ObservationPlane<setup::Tracking, WriterOff> obs(
        obsPlane, DirectionVector(cs, {0., 0., 1.}), false);

    Step step(particle, no_used_track);
    LengthType const length = obs.getMaxStepLength(particle, no_used_track);
    ProcessReturn const ret = obs.doContinuous(step, false);
    ProcessReturn const ret2 = obs.doContinuous(step, true);

    CHECK(length / 1_m == Approx(1).margin(1e-4));
    CHECK(ret == ProcessReturn::Ok);
    CHECK(ret2 == ProcessReturn::Ok);
  }

  SECTION("inclined plane, inclined particle") {

    MomentumVector const pnew = MomentumVector(cs, {1_GeV, 0.5_GeV, -0.4_GeV});
    HEPEnergyType const enew = sqrt(pnew.dot(pnew));
    particle.setDirection(pnew.normalized());
    particle.setEnergy(enew);

    Plane const obsPlane(Point(cs, {10_m, 5_m, 5_m}),
                         DirectionVector(cs, {1, 0.1, -0.05}));
    ObservationPlane<setup::Tracking, WriterOff> obs(obsPlane,
                                                     DirectionVector(cs, {0., 1., 0.}));

    Step step(particle, no_used_track);
    LengthType const length = obs.getMaxStepLength(particle, no_used_track);
    ProcessReturn const ret = obs.doContinuous(step, true);

    CHECK(length / 10_m == Approx(1.1375).margin(1e-4));
    CHECK(ret == ProcessReturn::ParticleAbsorbed);
  }

  SECTION("output") {
    Plane const obsPlane(Point(cs, {1_m, 0_m, 0_m}), DirectionVector(cs, {1., 0., 0.}));
    ObservationPlane<setup::Tracking, WriterOff> obs(
        obsPlane, DirectionVector(cs, {0., 0., 1.}), false);
    auto const cfg = obs.getConfig();
    CHECK(cfg["type"]);
  }
}
