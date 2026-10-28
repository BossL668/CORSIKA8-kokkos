/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/modules/ParticleCut.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <iostream>
#include <stdexcept>
using namespace corsika;
// Genuine read-only adapter: there is deliberately no addSecondary method.
struct ReadOnlyParticle {
  Point point = Point(get_root_CoordinateSystem(), 0_m, 0_m, 0_m);
  Code getPID() const { return Code::Positron; }
  HEPEnergyType getKineticEnergy() const { return .1_MeV; }
  double getWeight() const { return 3.; }
  TimeType getTime() const { return 0_s; }
  Point const& getPosition() const { return point; }
  DirectionVector getDirection() const {
    return DirectionVector(point.getCoordinateSystem(), {0., 0., 1.});
  }
};
int main() {
  ParticleCut<> cut(1_MeV, 1_MeV, 1_GeV, 1_GeV, 1_GeV, false);
  ReadOnlyParticle p;
  StraightTrajectory track(
      Line(p.getPosition(), VelocityVector(p.point.getCoordinateSystem(),
                                           {0_m / 1_s, 0_m / 1_s, 1_m / 1_s})),
      1_ns);
  Step step(p, track);
  if (cut.doContinuous(step) != ProcessReturn::ParticleAbsorbed ||
      cut.statistics().particles_ != 1 ||
      cut.statistics().stoppedPositronAnnihilations_ != 0 ||
      cut.statistics().weightedMediumRestMassInputGeV_ != 0.)
    throw std::runtime_error("disabled positron cut changed");
  std::cout << "CHECKS=4\n";
}
