/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/ParticleCut.hpp>

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/utility/CorsikaFenv.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/framework/process/ContinuousProcessIndex.hpp>

#include <SetupTestStack.hpp>
#include <SetupTestTrajectory.hpp>
#include <SetupTestEnvironment.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

#include <catch2/catch_all.hpp>

using namespace corsika;

using DummyEnvironmentInterface = IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>;
using DummyEnvironment = Environment<DummyEnvironmentInterface>;

TEST_CASE("ParticleCut", "process,continuous,secondary") {

  logging::set_level(logging::level::info);

  feenableexcept(FE_INVALID);
  using EnvType = DummyEnvironment;

  EnvType env;
  CoordinateSystemPtr const& rootCS = env.getCoordinateSystem();

  // setup empty particle stack
  test::Stack stack;
  stack.clear();
  // two energies
  HEPEnergyType const Eabove = 1_TeV;
  HEPEnergyType const Ebelow = 10_GeV;
  // list of arbitrary particles
  std::vector<Code> const particleList = {Code::PiPlus,   Code::PiMinus, Code::KPlus,
                                          Code::KMinus,   Code::K0Long,  Code::K0Short,
                                          Code::Electron, Code::MuPlus,  Code::NuE,
                                          Code::Neutron,  Code::NuMu};

  // common stating point
  const Point point0(rootCS, 0_m, 0_m, 0_m);

  SECTION("cut on particle type: inv") {

    // particle cut with 20GeV threshold for all, also cut invisible
    ParticleCut cut(20_GeV, 20_GeV, 20_GeV, 20_GeV, 20_GeV, true);
    CHECK(cut.getHadronKineticECut() == 20_GeV);

    // add primary particle to stack
    auto particle = stack.addParticle(
        std::make_tuple(Code::Proton, Eabove, DirectionVector(rootCS, {1, 0, 0}),
                        Point(rootCS, 0_m, 0_m, 0_m), 0_ns));
    // view on secondary particles
    test::StackView view(particle);
    // ref. to primary particle through the secondary view.
    // only this way the secondary view is populated
    auto projectile = view.getProjectile();
    // add secondaries, all with energies above the threshold
    // only cut is by species
    for (auto const proType : particleList)
      projectile.addSecondary(
          std::make_tuple(proType, Eabove, DirectionVector(rootCS, {1, 0, 0})));
    CHECK(view.getEntries() == 11);
    CHECK(stack.getEntries() == 12);

    cut.doSecondaries(view);

    CHECK(view.getEntries() == 9);
  }

  SECTION("cut on particle type: em") {

    ParticleCut cut(1_EeV, 1_EeV, 1_GeV, 1_GeV, 1_GeV, false);

    // add primary particle to stack
    auto particle = stack.addParticle(std::make_tuple(
        Code::Proton, Eabove, DirectionVector(rootCS, {1, 0, 0}), point0, 0_ns));
    // view on secondary particles
    test::StackView view(particle);
    // ref. to primary particle through the secondary view.
    // only this way the secondary view is populated
    auto projectile = view.getProjectile();
    // add secondaries, all with energies above the threshold
    // only cut is by species
    for (auto const proType : particleList) {
      projectile.addSecondary(
          std::make_tuple(proType, Eabove, DirectionVector(rootCS, {1, 0, 0})));
    }
    cut.doSecondaries(view);

    CHECK(view.getEntries() == 10);
  }

  SECTION("cut low energy") {
    ParticleCut cut(20_GeV, 20_GeV, 20_GeV, 20_GeV, 20_GeV, true);

    // add primary particle to stack
    auto particle = stack.addParticle(std::make_tuple(
        Code::Proton, Eabove, DirectionVector(rootCS, {1, 0, 0}), point0, 0_ns));
    // view on secondary particles
    test::StackView view(particle);
    // ref. to primary particle through the secondary view.
    // only this way the secondary view is populated
    auto projectile = view.getProjectile();
    // add secondaries, all with energies below the threshold
    // only cut is by species
    for (auto const proType : particleList)
      projectile.addSecondary(
          std::make_tuple(proType, Ebelow, DirectionVector(rootCS, {1, 0, 0})));
    unsigned short A = 18;
    unsigned short Z = 8;
    projectile.addSecondary(std::make_tuple(get_nucleus_code(A, Z), Eabove * A,
                                            DirectionVector(rootCS, {1, 0, 0})));
    projectile.addSecondary(std::make_tuple(get_nucleus_code(A, Z), Ebelow * A,
                                            DirectionVector(rootCS, {1, 0, 0})));

    cut.doSecondaries(view);

    CHECK(view.getEntries() == 1);
    CHECK(view.getSize() == 13);
  }

  SECTION("cut low energy: electrons, photons, hadrons, muons and tau") {
    ParticleCut cut(5_MeV, 5_MeV, 5_GeV, 5_GeV, 5_MeV, true);

    // add primary particle to stack
    auto particle = stack.addParticle(std::make_tuple(Code::Proton, Eabove - Proton::mass,
                                                      DirectionVector(rootCS, {1, 0, 0}),
                                                      point0, 0_ns));
    // view on secondary particles
    test::StackView view(particle);
    // ref. to primary particle through the secondary view.
    // only this way the secondary view is populated
    auto projectile = view.getProjectile();
    // add secondaries
    projectile.addSecondary(
        std::make_tuple(Code::Photon, 3_MeV, DirectionVector(rootCS, {1, 0, 0})));
    projectile.addSecondary(
        std::make_tuple(Code::Electron, 3_MeV, DirectionVector(rootCS, {1, 0, 0})));
    projectile.addSecondary(
        std::make_tuple(Code::PiPlus, 4_GeV, DirectionVector(rootCS, {1, 0, 0})));

    unsigned short A = 18;
    unsigned short Z = 8;
    projectile.addSecondary(std::make_tuple(get_nucleus_code(A, Z), 4_GeV * A,
                                            DirectionVector(rootCS, {1, 0, 0})));
    projectile.addSecondary(std::make_tuple(get_nucleus_code(A, Z), 6_GeV * A,
                                            DirectionVector(rootCS, {1, 0, 0})));

    cut.doSecondaries(view);

    CHECK(view.getEntries() == 1);
    CHECK(view.getSize() == 5);
  }

  SECTION("cut low energy:  reset thresholds of arbitrary set of particles") {
    ParticleCut cut({{Code::Electron, 5_MeV}, {Code::Positron, 50_MeV}}, false);
    CHECK(get_kinetic_energy_propagation_threshold(Code::Electron) !=
          get_kinetic_energy_propagation_threshold(Code::Positron));
    CHECK_FALSE(get_kinetic_energy_propagation_threshold(Code::Electron) ==
                Electron::mass);
    // test default values still correct
    CHECK(get_kinetic_energy_propagation_threshold(Code::Proton) == 5_GeV);
  }

  SECTION("cut on time") {
    ParticleCut cut(20_GeV, 20_GeV, 20_GeV, 20_GeV, 20_GeV, false);
    const TimeType too_late = 1_s;

    // add primary particle to stack
    auto particle = stack.addParticle(std::make_tuple(
        Code::Proton, Eabove, DirectionVector(rootCS, {1, 0, 0}), point0, too_late));
    // view on secondary particles
    test::StackView view(particle);
    // ref. to primary particle through the secondary view.
    // only this way the secondary view is populated
    auto projectile = view.getProjectile();
    // add secondaries, all with energies above the threshold
    // only cut is by time
    for (auto const proType : particleList) {
      projectile.addSecondary(
          std::make_tuple(proType, Eabove, DirectionVector(rootCS, {1, 0, 0})));
    }
    cut.doSecondaries(view);

    CHECK(view.getEntries() == 0);
  }

  setup::Trajectory const track = setup::testing::make_track<setup::Trajectory>(
      Line{point0, VelocityVector{rootCS, {0_m / second, 0_m / second, -constants::c}}},
      12_m / constants::c);

  SECTION("cut on doContinous, just invisibles") {

    ParticleCut cut(20_GeV, 20_GeV, 20_GeV, 20_GeV, 20_GeV, true);

    // add particles, all with energies above the threshold
    // only cut is by species
    for (auto const proType : particleList) {
      auto particle = stack.addParticle(
          std::make_tuple(proType, Eabove - get_mass(proType),
                          DirectionVector(rootCS, {1, 0, 0}), point0, 0_ns));
      Step step(particle, track);
      if (cut.doContinuous(step) == ProcessReturn::ParticleAbsorbed) { particle.erase(); }
    }

    CHECK(stack.getEntries() == 9);
  }
}
