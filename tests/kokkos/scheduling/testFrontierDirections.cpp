/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/CpuFrontier.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/media/IMediumPropertyModel.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <tests/common/TransportTestStack.hpp>

#include <boost/filesystem.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace {
  using namespace corsika;
  namespace frontier = kokkos::frontier;
  using model_type =
      IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
  using environment_type = Environment<model_type>;
  using stack_type = test::TransportStack<environment_type>::type;

  struct Fixture {
    CoordinateSystemPtr cs_ = get_root_CoordinateSystem();
    environment_type environment_;
    kokkos::EnvironmentSnapshot<model_type> snapshot_{*environment_.getUniverse(), cs_};
    stack_type stack_;

    auto particle(double normSquared = 1., Code pid = Code::Electron) {
      auto p = stack_.addParticle(std::make_tuple(
          pid, 1_GeV, DirectionVector(cs_, {0., 0., std::sqrt(normSquared)}),
          Point(cs_, 1_m, 2_m, 3_m), 4_ns));
      p.setTransportIdentity(transport::TransportIdentity{22, 20, 16, 7});
      p.setWeight(2.);
      p.setNode(environment_.getUniverse().get());
      return p;
    }

    template <typename TParticle>
    auto exportState(TParticle const& p) const {
      return kokkos::particle_transfer::to_device_state(
          p, cs_, p.getHistoryId(), p.getParentHistoryId(), p.getGeneration(), 7,
          snapshot_.getTopology());
    }
  };

  struct TemporaryDirectory {
    boost::filesystem::path path_ =
        boost::filesystem::temp_directory_path() /
        boost::filesystem::unique_path("c8-frontier-%%%%-%%%%");
    TemporaryDirectory() { boost::filesystem::create_directory(path_); }
    ~TemporaryDirectory() {
      boost::system::error_code ignored;
      boost::filesystem::remove_all(path_, ignored);
    }
    std::string file(std::string const& name) const { return (path_ / name).string(); }
  };

  template <typename TRouter>
  void check_native_step_then_export(TRouter& router, Fixture& fixture,
                                     std::string const& path, bool cpu) {
    // Reproduce the air and firn norm errors, and a deviation below unity.
    std::uint64_t history = 22;
    for (double delta : {1.551738959548743e-9, 1.0308985665e-8, -5.e-8}) {
      CAPTURE(delta, cpu);
      auto p = fixture.particle(1. + delta);
      p.setTransportIdentity(transport::TransportIdentity{history++, 20, 16, 7});
      auto const before = fixture.exportState(p);
      REQUIRE_FALSE(router.canRoute(p, 7));
      REQUIRE_FALSE(router.canRoute(p, 7));
      auto const retained = fixture.exportState(p);
      CHECK(retained.energyGeV_ == before.energyGeV_);
      CHECK(retained.weight_ == before.weight_);
      CHECK(retained.timeS_ == before.timeS_);
      CHECK(retained.historyId_ == before.historyId_);
      CHECK(retained.parentHistoryId_ == before.parentHistoryId_);
      CHECK(retained.generation_ == before.generation_);
      CHECK(retained.environmentNode_ == before.environmentNode_);
      for (int axis = 0; axis < 3; ++axis) {
        CHECK(retained.positionM_[axis] == before.positionM_[axis]);
        CHECK(retained.direction_[axis] == before.direction_[axis]);
      }
      REQUIRE_THROWS_AS(frontier::validate_frontier_state(retained), std::runtime_error);

      // Use the native Step endpoint operation; the router must not normalize.
      StraightTrajectory track(Line(p.getPosition(), p.getVelocity()), 1_ns);
      Step step(p, track);
      p.setDirection(step.getDirectionPost());
      REQUIRE(router.canRoute(p, 8));
      auto const ready = fixture.exportState(p);
      REQUIRE_NOTHROW(frontier::validate_frontier_state(ready));
      REQUIRE_NOTHROW(router.stage(p, p.getHistoryId(), p.getParentHistoryId(),
                                   p.getGeneration(), 8));
      CHECK(p.getEnergy() / 1_GeV == before.energyGeV_);
      CHECK(p.getWeight() == before.weight_);
      fixture.stack_.clear();
    }
    router.endOfShower();
    std::ifstream input(path);
    std::string line;
    REQUIRE(static_cast<bool>(std::getline(input, line)));
    CHECK(line.find(cpu ? "C8_CPU_FRONTIER_V1" : "C8_STATIC_FRONTIER_V2") == 0);
    unsigned int count = 0;
    while (std::getline(input, line)) {
      std::istringstream row(line);
      int pid, node;
      std::uint64_t history, parent, generation, step;
      double energy, weight, time, x, y, z, nx, ny, nz;
      row >> pid >> history >> parent >> generation >> step >> energy >> weight >> time >>
          x >> y >> z >> nx >> ny >> nz >> node;
      REQUIRE(static_cast<bool>(row));
      CHECK(pid == 11);
      CHECK(history == 22 + count);
      CHECK(parent == 20);
      CHECK(generation == 16);
      CHECK(step == 8);
      CHECK(weight == 2.);
      CHECK(node == 0);
      CHECK(std::abs(nx * nx + ny * ny + nz * nz - 1.) <=
            frontier::DirectionNormSquaredTolerance);
      ++count;
    }
    CHECK(count == 3);
  }
} // namespace

TEST_CASE("Frontier routers retain direction drift for native stepping",
          "[kokkos][frontier]") {
  Fixture fixture;
  TemporaryDirectory output;
  SECTION("CPU bootstrap") {
    auto const path = output.file("cpu.txt");
    frontier::CpuBootstrapRouter<model_type> router(path, fixture.cs_, fixture.snapshot_);
    check_native_step_then_export(router, fixture, path, true);
  }
  SECTION("EM capture for device workers") {
    auto const path = output.file("em.txt");
    frontier::CaptureRouter<model_type> router(path, fixture.cs_, 100.,
                                               fixture.snapshot_);
    check_native_step_then_export(router, fixture, path, false);
  }
}

TEST_CASE("Frontier direction checks retain valid states and reject invalid directions",
          "[kokkos][frontier]") {
  Fixture fixture;
  std::size_t retained = 0;
  double maximumError = 0.;
  for (double norm : {1., 1. + 0.5e-9, 1. - 0.5e-9}) {
    auto p = fixture.particle(norm);
    CHECK(frontier::frontier_unit_direction(p, fixture.cs_, retained, maximumError));
    CHECK_NOTHROW(frontier::validate_frontier_state(fixture.exportState(p)));
  }
  CHECK(retained == 0);
  CHECK(maximumError == 0.);
  for (double norm : {0., std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
    auto p = fixture.particle(norm);
    CHECK_THROWS_AS(
        frontier::frontier_unit_direction(p, fixture.cs_, retained, maximumError),
        std::runtime_error);
    CHECK_THROWS_AS(frontier::validate_frontier_state(fixture.exportState(p)),
                    std::runtime_error);
  }
  auto state = fixture.exportState(fixture.particle());
  SECTION("Invalid energy remains an error") { state.energyGeV_ = -1.; }
  SECTION("Invalid weight remains an error") { state.weight_ = -1.; }
  SECTION("Invalid history remains an error") { state.historyId_ = 0; }
  CHECK_THROWS_AS(frontier::validate_frontier_state(state), std::runtime_error);
}

TEST_CASE("Frontier eligibility retains species generation and energy limits",
          "[kokkos][frontier]") {
  Fixture fixture;
  TemporaryDirectory output;
  frontier::CpuBootstrapRouter<model_type> cpu(output.file("cpu.txt"), fixture.cs_,
                                               fixture.snapshot_, 2.);
  frontier::CaptureRouter<model_type> em(output.file("em.txt"), fixture.cs_, 2.,
                                         fixture.snapshot_);
  for (Code pid : {Code::Electron, Code::Positron, Code::Photon, Code::MuPlus,
                   Code::MuMinus, Code::Proton}) {
    CAPTURE(pid);
    auto p = fixture.particle(1., pid);
    CHECK(cpu.canRoute(p, 7));
    CHECK(em.canRoute(p, 7) == (pid != Code::Proton));
    p.setEnergy(3_GeV);
    CHECK_FALSE(cpu.canRoute(p, 7));
    CHECK_FALSE(em.canRoute(p, 7));
  }
  auto primary = fixture.particle();
  primary.setTransportIdentity(transport::TransportIdentity{1, 0, 0, 0});
  CHECK_FALSE(cpu.canRoute(primary, 0));
  CHECK(em.canRoute(primary, 0));
}
