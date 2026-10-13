/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include "AtmosphereBoundaryNativeReference.hpp"
#include <corsika/accelerator/em/common/EnvironmentSnapshotBuilder.hpp>
#include <corsika/modules/tracking/TrackingStraight.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <tests/common/SetupTestStack.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace corsika;
namespace em = corsika::gpu::em;

em::EnvironmentSnapshot makeAtmosphereBoundaryTestEnvironment() {
  return em::makeCorsika7AtmosphereSnapshot(AtmosphereId::USStdBK,
      {0., 0., 0.}, 0, 6371000. + 2680.444195);
}

void compareNativeAtmosphereBoundary(
    std::vector<em::EmParticleState> const& inputs,
    std::vector<em::PhotonTransportRecord> const& records,
    em::EnvironmentSnapshot const& snapshot) {
  auto require = [](bool ok, char const* message) {
    if (!ok) throw std::runtime_error(message);
  };
  logging::set_level(logging::level::warn);
  // Independent C8 volume-tree and TrackingStraight oracle for the four
  // recorded cases. All near crossings are outside the 0.1 mm path guard.
  DummyEnvironment env;
  auto const& cs = env.getCoordinateSystem();
  auto* parent = env.getUniverse().get();
  DummyEnvironment::BaseNodeType* nodes[5]{};
  for (int i = 4; i >= 0; --i) {
    auto node = DummyEnvironment::createNode<Sphere>(
        Point(cs, 0_m, 0_m, 0_m), snapshot.atmosphere_layers[i].outer_radius_m * 1_m);
    nodes[i] = node.get();
    parent->addChild(std::move(node));
    parent = nodes[i];
  }
  tracking_line::Tracking tracking;
  for (int i = 0; i < 4; ++i) {
    auto const& p = inputs[i];
    Point const position(cs, p.position_m[0] * 1_m, p.position_m[1] * 1_m,
                         p.position_m[2] * 1_m);
    int layer = 0;
    while (!nodes[layer]->getVolume().contains(position)) ++layer;
    test::Stack stack;
    auto particle = stack.addParticle(std::make_tuple(Code::Photon, 1_GeV,
        DirectionVector(cs, {p.direction[0], p.direction[1], p.direction[2]}),
        position, 0_ns));
    particle.setNode(nodes[layer]);
    auto const [trajectory, next] = tracking.getTrack(particle);
    auto const& step = records[i];
    auto const distance = trajectory.getLength(1) / 1_m;
    std::cout << "native case=" << i << " C8_m=" << distance
              << " Kokkos_m=" << step.distance_m << '\n';
    require(step.start_layer_index == layer, "native start layer differs");
    require(step.limit == em::PhotonTransportLimit::LayerBoundary, "not a layer crossing");
    require(next == nodes[step.end_layer_index], "native next volume differs");
    // Allow cancellation differences on an Earth-sized sphere.
    require(std::abs(distance - step.distance_m) < 5.e-8,
            "native boundary distance differs");
  }
}
