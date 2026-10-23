/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/radio/common/HostRadioCollector.hpp>
#include <corsika/modules/radio/RadioProcess.hpp>
#include <corsika/modules/radio/detectors/ObserverCollection.hpp>
#include <corsika/modules/radio/observers/TimeDomainObserver.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
  using namespace corsika;
  namespace accelerated_radio = corsika::accelerator::radio;
  using Detector = ObserverCollection<TimeDomainObserver>;
  struct Propagator {};

  class CountingRadio final
      : public RadioProcess<Detector, CountingRadio, Propagator> {
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

  struct Backend {
    bool enabled{true};
    bool fail{};
    std::vector<gpu::em::LeptonTransportRecord> tracks;
    bool gpuRadioEnabled() const { return enabled; }
    void enqueueRadioTracks(gpu::em::LeptonTransportRecord const* records, std::size_t n) {
      if (fail) throw std::runtime_error("test sink failure");
      tracks.insert(tracks.end(), records, records + n);
    }
  };

  void require(bool value, char const* message) {
    if (!value) throw std::runtime_error(message);
  }
}

int main() {
  using namespace corsika;
  auto const root = get_root_CoordinateSystem();
  auto const local = make_translation(root, QuantityVector<length_d>{10_m, 20_m, 30_m});
  Detector detector;
  TimeDomainObserver observer("test", Point(root, 0_m, 0_m, 0_m), root,
                              0_s, 100_ns, 1_GHz, 0_s);
  detector.addObserver(observer);
  Propagator propagator;
  CountingRadio coreas(detector, propagator), zhs(detector, propagator);
  Particle particle{Point(local, 0_m, 0_m, 10_m), DirectionVector(local, {0., 0., -1.})};
  Line const line(particle.position, particle.direction * (0.9 * constants::c));
  StraightTrajectory const trajectory(line, 1_ns);
  Step step(particle, trajectory);
  step.add_dEkin(-0.1_GeV);

  coreas.doContinuous(step, false);
  zhs.doContinuous(step, false);
  require(coreas.calls == 1 && zhs.calls == 1, "default scalar projection changed");

  Backend backend;
  accelerated_radio::HostRadioCollector collector(backend);
  {
    accelerated_radio::ScopedRadioTrackSink scope(coreas, zhs, collector, root);
    coreas.doContinuous(step, false);
    zhs.doContinuous(step, false);
    require(backend.tracks.size() == 1, "CoREAS/ZHS duplicated captured track");
    require(coreas.calls == 1 && zhs.calls == 1, "native simulation still executed");
    require(coreas.offloadedSegmentCount() == 1 && zhs.offloadedSegmentCount() == 1,
            "offload counters incorrect");
    auto const& record = backend.tracks.front();
    auto const expected = step.getPositionPre().getCoordinates(root);
    for (int axis = 0; axis < 3; ++axis)
      require(record.start.position_m[axis] == expected[axis] / 1_m,
              "track was not converted to the shared coordinate system");
    require(record.start.pid == 11 && record.start.weight == 2.5, "PID or weight lost");
    require(record.start.energy_GeV == 2. + gpu::em::TransportElectronMassGeV,
            "kinetic/total energy conversion mismatch");
    require(record.end.energy_GeV == step.getEkinPost() / 1_GeV +
                                         gpu::em::TransportElectronMassGeV,
            "post-step energy lost");
    require(record.start.time_s == step.getTimePre() / 1_s &&
                record.end.time_s == step.getTimePost() / 1_s, "step time lost");
    bool nested_rejected = false;
    try {
      accelerated_radio::ScopedRadioTrackSink nested(coreas, zhs, collector, root);
    } catch (std::logic_error const&) { nested_rejected = true; }
    require(nested_rejected && coreas.hasTrackSink(), "nested scope disturbed existing sink");

    particle.pid = Code::MuMinus;
    coreas.doContinuous(step, false);
    zhs.doContinuous(step, false);
    require(backend.tracks.size() == 1, "muon incorrectly added to radio");
    particle.pid = Code::Positron;
    coreas.doContinuous(step, false);
    zhs.doContinuous(step, false);
    require(backend.tracks.size() == 2 && backend.tracks.back().start.pid == -11,
            "positron track lost");
  }
  require(!coreas.hasTrackSink() && !zhs.hasTrackSink(), "scope did not detach sinks");
  coreas.doContinuous(step, false);
  zhs.doContinuous(step, false);
  require(coreas.calls == 2 && zhs.calls == 2, "scalar path not restored");

  bool failed = false;
  backend.fail = true;
  try {
    accelerated_radio::ScopedRadioTrackSink scope(coreas, zhs, collector, root);
    coreas.doContinuous(step, false);
  } catch (std::runtime_error const&) { failed = true; }
  require(failed && !coreas.hasTrackSink() && !zhs.hasTrackSink(),
          "sink failure was swallowed or left dangling routing");
  backend.enabled = false;
  failed = false;
  try { accelerated_radio::HostRadioCollector disabled(backend); }
  catch (std::logic_error const&) { failed = true; }
  require(failed, "disabled GPU radio accepted a redirected track");
  std::cout << "Host radio routing, coordinates, single capture and scope restoration: PASS\n";
}
