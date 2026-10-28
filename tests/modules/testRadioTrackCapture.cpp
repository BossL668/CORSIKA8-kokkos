/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>
#include <catch2/catch_all.hpp>
#include <vector>
namespace {
  using namespace corsika;
  using Detector = ObserverCollection<TimeDomainObserver>;
  struct Propagator {};

  class CountingRadio final : public RadioProcess<Detector, CountingRadio, Propagator> {
  public:
    static constexpr auto algorithm = "CoREAS";
    using RadioProcess::RadioProcess;
    std::size_t calls{};
    template <typename Particle>
    ProcessReturn simulate(Step<Particle> const&) {
      ++calls;
      return ProcessReturn::Ok;
    }
  };

  struct Particle {
    Point position;
    DirectionVector direction;
    Code pid{Code::Electron};
    Point const& getPosition() const { return position; }
    DirectionVector const& getDirection() const { return direction; }
    TimeType getTime() const { return 3_ns; }
    HEPEnergyType getKineticEnergy() const { return 2_GeV; }
    Code getPID() const { return pid; }
    double getWeight() const { return 2.5; }
  };

  struct Sink : radio::RadioTrackSink {
    std::vector<radio::RadioTrackSegment> tracks_;
    void append(radio::RadioTrackSegment const& track) override {
      tracks_.push_back(track);
    }
  };
} // namespace
TEST_CASE("Optional native radio capture preserves default path and common frame",
          "[radio][capture]") {
  using namespace corsika;
  auto root = get_root_CoordinateSystem();
  auto local = make_translation(root, QuantityVector<length_d>{10_m, 20_m, 30_m});
  local = make_rotation(local, QuantityVector<dimensionless_d>{0., 0., 1.}, 0.3);
  Detector detector, empty;
  TimeDomainObserver observer("test", Point(root, 0_m, 0_m, 0_m), root, 0_s, 100_ns,
                              1_GHz, 0_s);
  detector.addObserver(observer);
  Propagator prop;
  CountingRadio process(detector, prop), disabled(empty, prop);
  Particle particle{Point(local, 1_m, 2_m, 10_m),
                    DirectionVector(local, {0.6, 0., -0.8})};
  Line line(particle.position, particle.direction * (0.9 * constants::c));
  StraightTrajectory trajectory(line, 1_ns);
  Step step(particle, trajectory);
  step.add_dEkin(-0.1_GeV);
  process.doContinuous(step, false);
  REQUIRE(process.calls == 1);
  Sink sink;
  REQUIRE_THROWS_AS(process.setTrackSink(&sink, {}), std::logic_error);
  REQUIRE_FALSE(process.hasTrackSink());
  process.setTrackSink(&sink, root);
  REQUIRE_THROWS_AS(process.setTrackSink(&sink, root), std::logic_error);
  process.doContinuous(step, false);
  disabled.setTrackSink(&sink, root);
  disabled.doContinuous(step, false);
  REQUIRE(sink.tracks_.size() == 1);
  REQUIRE(process.calls == 1);
  REQUIRE(process.offloadedSegmentCount() == 1);
  REQUIRE(disabled.offloadedSegmentCount() == 0);
  auto const& t = sink.tracks_[0];
  auto a = step.getPositionPre().getCoordinates(root);
  auto b = step.getPositionPost().getCoordinates(root);
  auto c = step.getDirectionPre().getComponents(root);
  auto d = step.getDirectionPost().getComponents(root);
  for (int i = 0; i < 3; ++i) {
    REQUIRE(t.startPositionM_[i] == a[i] / 1_m);
    REQUIRE(t.endPositionM_[i] == b[i] / 1_m);
    REQUIRE(t.startDirection_[i] == c[i].magnitude());
    REQUIRE(t.endDirection_[i] == d[i].magnitude());
  }
  REQUIRE(t.startTimeS_ == step.getTimePre() / 1_s);
  REQUIRE(t.endTimeS_ == step.getTimePost() / 1_s);
  REQUIRE(t.startKineticEnergyGeV_ == step.getEkinPre() / 1_GeV);
  REQUIRE(t.endKineticEnergyGeV_ == step.getEkinPost() / 1_GeV);
  REQUIRE(t.weight_ == 2.5);
  REQUIRE(t.pdg_ == 11);
  process.setTrackSink(nullptr, {});
  process.doContinuous(step, false);
  REQUIRE(process.calls == 2);
  // Capture-disabled routing still suppresses this model's native calculation.
  process.setTrackSink(&sink, root, false);
  process.doContinuous(step, false);
  REQUIRE(sink.tracks_.size() == 1);
  REQUIRE(process.calls == 2);
  REQUIRE(process.offloadedSegmentCount() == 2);
  process.setTrackSink(nullptr, {});
  disabled.setTrackSink(nullptr, {});
}
