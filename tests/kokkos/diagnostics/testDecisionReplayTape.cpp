/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
#include "ProductionCudaDecisionReplayTape.hpp"
#endif
#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <tests/common/SetupTestTrajectory.hpp>
#include <corsika/framework/core/Step.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <corsika/kokkos/diagnostics/DecisionReplayTape.hpp>
#include <catch2/catch_test_macros.hpp>
#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <cstdlib>
#include <iterator>
using namespace corsika;
namespace cp = corsika::kokkos::diagnostics;
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
namespace old = corsika::production_stepper_reference::validation;
#endif
namespace fs = std::filesystem;
namespace {
  struct Directory {
    fs::path path_;
    Directory() {
      char name[] = "/tmp/c8-decision-replay-XXXXXX";
      auto p = mkdtemp(name);
      if (!p) throw std::runtime_error("Cannot make replay test directory");
      path_ = p;
    }
    ~Directory() {
      std::error_code error;
      fs::remove_all(path_, error);
    }
  };
  template <typename Tape, typename Snapshot, typename Limit>
  void write_tape(fs::path const& path, Snapshot const& snapshot, Limit limit) {
    auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    test::TransportStack<DummyEnvironment>::type stack;
    auto p = stack.addParticle(std::make_tuple(Code::Electron, 10_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 1_m, 2_m, 3_m), 4_ns));
    p.setNode(node);
    p.setWeight(3.5);
    auto track = setup::testing::make_track<setup::Trajectory>(
        Line(p.getPosition(), p.getMomentum() / p.getEnergy() * constants::c), 1_ns);
    Step step(p, track);
    step.add_dEkin(-.125_GeV);
    auto& tape = Tape::instance();
    tape.open(path, snapshot, 123, 2);
    CHECK(tape.enabled());
    CHECK_THROWS_AS(tape.open(path.string() + "-second", snapshot, 123, 2),
                    std::logic_error);
    tape.beginShower(7);
    tape.recordTransportStep(step, limit, ProcessReturn::Ok);
    tape.recordTransportStep(step, limit, ProcessReturn::ParticleAbsorbed);
    tape.beginShower(8);
    tape.recordTransportStep(step, limit, ProcessReturn::Ok);
    tape.close();
    CHECK_FALSE(tape.enabled());
    tape.close();
    CHECK_THROWS_AS(tape.open(path, snapshot, 123, 2), std::runtime_error);
  }
  struct Observer {
    Point location_;
    Point const& getLocation() const { return location_; }
    auto getStartTime() const { return 4_ns; }
    auto getSampleRate() const { return 1.e9_Hz; }
    std::vector<double> getWaveformX() const { return {0., 0., 0.}; }
    std::vector<double> getAxis() const { return {4., 5., 6.}; }
  };
  struct Detector {
    std::vector<Observer> observers_;
    auto size() const { return observers_.size(); }
    auto const& getObservers() const { return observers_; }
  };
} // namespace

TEST_CASE("Decision tapes round trip physical step records", "[decision_replay]") {
  Directory directory;
  cp::ReplayRadioSnapshot radio{};
  radio.propagation_.refractivity_ = {.1, .2, .3};
  radio.propagation_.integratedRefractivity_ = {.1, .3, .6};
  auto path = directory.path_ / "current.bin";
  write_tape<cp::DecisionReplayTape>(path, radio, cp::ReplayTransportLimit::Continuous);
  auto data = cp::DecisionReplayTape::read(path);
  CHECK(data.header_.randomSeed_ == 123);
  CHECK(data.header_.configuredShowers_ == 2);
  CHECK(data.header_.recordCount_ == 3);
  CHECK(data.radio_.propagation_.refractivity_ == radio.propagation_.refractivity_);
  CHECK(data.radio_.propagation_.integratedRefractivity_ ==
        radio.propagation_.integratedRefractivity_);
  REQUIRE(data.steps_.size() == 3);
  CHECK(data.steps_[0].shower_ == 7);
  CHECK(data.steps_[0].ordinal_ == 0);
  CHECK(data.steps_[1].shower_ == 7);
  CHECK(data.steps_[1].ordinal_ == 1);
  CHECK(data.steps_[2].shower_ == 8);
  CHECK(data.steps_[2].ordinal_ == 0);
  for (auto const& step : data.steps_) {
    CHECK(step.pdg_ == 11);
    CHECK(step.stackIndex_ == 0);
    CHECK(step.weight_ == 3.5);
    CHECK(step.startTotalEnergyGeV_ == (10_GeV + get_mass(Code::Electron)) / 1_GeV);
    CHECK(step.endTotalEnergyGeV_ == (9.875_GeV + get_mass(Code::Electron)) / 1_GeV);
    CHECK(step.startTimeS_ == 4_ns / 1_s);
    CHECK(step.endTimeS_ == 5_ns / 1_s);
    CHECK(step.transportLimit_ == cp::ReplayTransportLimit::Continuous);
  }
  std::ifstream input(path, std::ios::binary);
  std::vector<char> bytes{std::istreambuf_iterator<char>(input), {}};
  auto invalid = directory.path_ / "invalid.bin";
  auto rejects = [&](std::vector<char> const& value) {
    {
      std::ofstream file(invalid, std::ios::binary | std::ios::trunc);
      file.write(value.data(), value.size());
    }
    CHECK_THROWS_AS(cp::DecisionReplayTape::read(invalid), std::runtime_error);
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
    CHECK_THROWS_AS(old::CudaDecisionReplayTape::read(invalid), std::runtime_error);
#endif
  };
  auto bad = bytes;
  bad.pop_back();
  rejects(bad);
  bad = bytes;
  bad.push_back('x');
  rejects(bad);
  for (auto offset : {offsetof(cp::ReplayTapeHeader, magic_),
                      offsetof(cp::ReplayTapeHeader, formatVersion_),
                      offsetof(cp::ReplayTapeHeader, headerBytes_),
                      offsetof(cp::ReplayTapeHeader, recordBytes_),
                      offsetof(cp::ReplayTapeHeader, endianMarker_)}) {
    bad = bytes;
    bad[offset] ^= 1;
    rejects(bad);
  }
  bad = bytes;
  std::fill_n(bad.begin() + offsetof(cp::ReplayTapeHeader, refractivityCount_),
              sizeof(std::uint64_t), char(0xff));
  rejects(bad);
  CHECK_THROWS_AS(cp::DecisionReplayTape::read(directory.path_ / "absent"),
                  std::runtime_error);
  CHECK_THROWS_AS(cp::DecisionReplayTape::instance().open(
                      directory.path_ / "missing/dir/file", radio, 1, 1),
                  std::runtime_error);
  cp::DecisionReplayTape::instance().open({}, radio, 1, 1);
  CHECK_FALSE(cp::DecisionReplayTape::instance().enabled());
}

#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
TEST_CASE("Decision tapes preserve pinned production wire data", "[decision_replay]") {
  Directory directory;
  cp::ReplayRadioSnapshot radio{};
  old::ReplayRadioSnapshot previous{};
  radio.propagation_.refractivity_ = {.1, .2, .3};
  radio.propagation_.integratedRefractivity_ = {.1, .3, .6};
  previous.propagation.refractivity = {.1, .2, .3};
  previous.propagation.integrated_refractivity = {.1, .3, .6};
  auto current = directory.path_ / "current.bin",
       original = directory.path_ / "original.bin";
  write_tape<cp::DecisionReplayTape>(current, radio,
                                     cp::ReplayTransportLimit::Continuous);
  write_tape<old::CudaDecisionReplayTape>(original, previous,
                                          old::ReplayTransportLimit::Continuous);
  for (auto const& path : {current, original}) {
    auto a = cp::DecisionReplayTape::read(path);
    auto b = old::CudaDecisionReplayTape::read(path);
    CHECK(a.header_.formatVersion_ == b.header.format_version);
    CHECK(a.header_.recordBytes_ == b.header.record_bytes);
    REQUIRE(a.steps_.size() == 3);
    REQUIRE(b.steps.size() == 3);
    CHECK(a.radio_.propagation_.refractivity_ == b.radio.propagation.refractivity);
    CHECK(a.radio_.propagation_.integratedRefractivity_ ==
          b.radio.propagation.integrated_refractivity);
    for (std::size_t i = 0; i < 3; ++i) {
      auto const& x = a.steps_[i];
      auto const& y = b.steps[i];
      CHECK(x.shower_ == y.shower);
      CHECK(x.ordinal_ == y.ordinal);
      CHECK(x.stackIndex_ == y.stack_index);
      CHECK(x.pdg_ == y.pdg);
      CHECK(x.weight_ == y.weight);
      CHECK(x.startTimeS_ == y.start_time_s);
      CHECK(x.endTimeS_ == y.end_time_s);
      CHECK(x.startTotalEnergyGeV_ == y.start_total_energy_GeV);
      CHECK(x.endTotalEnergyGeV_ == y.end_total_energy_GeV);
      CHECK(static_cast<int>(x.transportLimit_) == static_cast<int>(y.transport_limit));
      CHECK(x.processReturn_ == y.process_return);
      for (int axis = 0; axis < 3; ++axis) {
        CHECK(x.startPositionM_[axis] == y.start_position_m[axis]);
        CHECK(x.endPositionM_[axis] == y.end_position_m[axis]);
        CHECK(x.startDirection_[axis] == y.start_direction[axis]);
        CHECK(x.endDirection_[axis] == y.end_direction[axis]);
      }
    }
  }
  auto a = cp::DecisionReplayTape::read(current);
  auto b = cp::DecisionReplayTape::read(original);
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(a.steps_[i].endTotalEnergyGeV_ == b.steps_[i].endTotalEnergyGeV_);
    CHECK(a.steps_[i].endTimeS_ == b.steps_[i].endTimeS_);
    for (int axis = 0; axis < 3; ++axis)
      CHECK(a.steps_[i].endPositionM_[axis] == b.steps_[i].endPositionM_[axis]);
  }
}
#endif

TEST_CASE("Diagnostic column snapshot samples the configured medium",
          "[decision_replay]") {
  using Interface = IRefractiveIndexModel<IMediumModel>;
  Environment<Interface> env;
  auto const& cs = env.getCoordinateSystem();
  auto world =
      Environment<Interface>::createNode<Sphere>(Point(cs, 0_m, 0_m, 0_m), 100_km);
  world->setModelProperties<UniformRefractiveIndex<HomogeneousMedium<Interface>>>(
      1.0003, 1_kg / cube(1_m), NuclearComposition({Code::Oxygen}, {1.}));
  env.getUniverse()->addChild(std::move(world));
  Detector detector{{Observer{Point(cs, 1_m, 2_m, 0_m)}}};
  auto a =
      cp::make_replay_radio_snapshot(env, Point(cs, 0_m, 0_m, 100_m),
                                     Point(cs, 0_m, 0_m, 0_m), 10_m, detector, detector);
  REQUIRE_FALSE(a.propagation_.refractivity_.empty());
  for (double value : a.propagation_.refractivity_) CHECK(value == 1.0003 - 1.);
  REQUIRE(a.coreasObservers_.size() == 1);
  REQUIRE(a.zhsObservers_.size() == 1);
  CHECK(a.coreasObservers_[0].sampleRateHz_ == 1.e9);
  CHECK(a.coreasObservers_[0].numberOfBins_ == 3);
  CHECK(a.coreasObservers_[0].axisStartNs_ == 4.);
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
  auto b =
      old::makeReplayRadioSnapshot(env, Point(cs, 0_m, 0_m, 100_m),
                                   Point(cs, 0_m, 0_m, 0_m), 10_m, detector, detector);
  CHECK(a.propagation_.refractivity_ == b.propagation.refractivity);
  CHECK(a.propagation_.integratedRefractivity_ == b.propagation.integrated_refractivity);
  CHECK(a.propagation_.minimumHeightM_ == b.propagation.minimum_height_m);
  CHECK(a.propagation_.slopeIntegratedRefractivityUpper_ ==
        b.propagation.slope_integrated_refractivity_upper);
  CHECK(a.coreasObservers_[0].durationS_ == b.coreas_observers[0].duration_s);
#endif
  CHECK_THROWS_AS(
      cp::make_replay_radio_snapshot(env, Point(cs, 0_m, 0_m, 100_m),
                                     Point(cs, 0_m, 0_m, 0_m), 0_m, detector, detector),
      std::invalid_argument);
}
cp::DecisionReplayTape* replay_tape_address_a();
cp::DecisionReplayTape* replay_tape_address_b();
TEST_CASE("Replay singleton has one identity across translation units",
          "[decision_replay]") {
  CHECK(replay_tape_address_a() == replay_tape_address_b());
  CHECK(replay_tape_address_a() == &cp::DecisionReplayTape::instance());
}
