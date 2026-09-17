// Independent RadioPropa bridge for the mountain-radio propagation audit.
//
// This file is intentionally not linked into CORSIKA.  It is compiled against
// an unmodified RadioPropa checkout so that the comparison does not share the
// production ray integrator.  The validated revision is recorded in the
// generated YAML, not assumed from this source file.

#include <radiopropa/Candidate.h>
#include <radiopropa/Geometry.h>
#include <radiopropa/ScalarField.h>
#include <radiopropa/module/Discontinuity.h>
#include <radiopropa/module/PropagationCK.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace radiopropa;

namespace {

constexpr double kInvariant = 0.9761499481235465;

class LinearRockProfile final : public ScalarField {
 public:
  double getValue(Vector3d const& position) const override {
    return 2.0 - 2.0e-4 * position.z;
  }

  Vector3d getGradient(Vector3d const&) const override {
    return Vector3d(0.0, 0.0, -2.0e-4);
  }
};

struct Crossing {
  double xM{};
  double timeS{};
  double sinTheta{};
  int steps{};
};

Crossing traceToHeight(double const maximumStepM,
                       std::vector<std::array<double, 3>>* sampledPath) {
  double const sinTheta0 = kInvariant / 2.0;
  ParticleState state(0, 250.e6, Vector3d(0.0, 0.0, 0.0),
                      Vector3d(sinTheta0, 0.0,
                               std::sqrt(1.0 - sinTheta0 * sinTheta0)),
                      Vector3d(0.0, 1.0, 0.0));
  Candidate candidate(state);
  candidate.setNextStep(maximumStepM);
  ref_ptr<ScalarField> field = new LinearRockProfile();
  PropagationCK propagator(field, 1.e-11, maximumStepM * 1.e-4,
                           maximumStepM);
  int steps = 0;
  double nextSampleZ = 0.0;
  double lastPreviousTime = 0.0;
  if (sampledPath != nullptr) sampledPath->push_back({0.0, 0.0, 0.0});

  while (candidate.current.getPosition().z < 1000.0 && steps < 10000000) {
    double const previousTime = candidate.getPropagationTime();
    lastPreviousTime = previousTime;
    propagator.process(&candidate);
    ++steps;
    Vector3d const a = candidate.previous.getPosition();
    Vector3d const b = candidate.current.getPosition();
    double const currentTime = candidate.getPropagationTime();
    if (sampledPath != nullptr) {
      while (nextSampleZ + 10.0 <= std::min(1000.0, b.z)) {
        nextSampleZ += 10.0;
        double const q = (nextSampleZ - a.z) / (b.z - a.z);
        sampledPath->push_back(
            {nextSampleZ, a.x + q * (b.x - a.x),
             previousTime + q * (currentTime - previousTime)});
      }
    }
  }
  Vector3d const a = candidate.previous.getPosition();
  Vector3d const b = candidate.current.getPosition();
  double const q = (1000.0 - a.z) / (b.z - a.z);
  double const timeCurrent = candidate.getPropagationTime();
  // `previous` and lastPreviousTime refer to the same pre-step state; do not
  // infer the accepted Cash-Karp step from the proposed next step.
  double const timePrevious = lastPreviousTime;
  Vector3d const directionA = candidate.previous.getDirection();
  Vector3d const directionB = candidate.current.getDirection();
  double const sinTheta =
      directionA.x + q * (directionB.x - directionA.x);
  return {a.x + q * (b.x - a.x),
          timePrevious + q * (timeCurrent - timePrevious), sinTheta, steps};
}

void emitInterfaceCase(double const incidenceDeg) {
  double const incidence = incidenceDeg * M_PI / 180.0;
  ParticleState before, after;
  before.setPosition(Vector3d(0.0, 0.0, -1.e-8));
  after.setPosition(Vector3d(0.0, 0.0, 1.e-8));
  before.setDirection(
      Vector3d(std::sin(incidence), 0.0, std::cos(incidence)));
  after.setDirection(before.getDirection());
  // Pure s polarization for the x-z plane of incidence.
  before.setAmplitude(Vector3d(0.0, 1.0, 0.0));
  after.setAmplitude(before.getAmplitude());
  Candidate candidate;
  candidate.previous = before;
  candidate.current = after;
  Discontinuity interface(
      new Plane(Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 1.0)),
      2.0, 1.0);
  interface.process(&candidate);
  bool const transmitted = !candidate.secondaries.empty();
  double transmittedSin = -1.0;
  double transmittedAmplitude = 0.0;
  if (transmitted) {
    transmittedSin = std::hypot(
        candidate.secondaries[0]->current.getDirection().x,
        candidate.secondaries[0]->current.getDirection().y);
    transmittedAmplitude =
        candidate.secondaries[0]->current.getAmplitude().getR();
  }
  std::cout << "interface," << incidenceDeg << ',' << int(transmitted) << ','
            << transmittedSin << ',' << transmittedAmplitude << ','
            << candidate.current.getAmplitude().getR() << "\n";
}

}  // namespace

int main() {
  std::cout << std::setprecision(17);
  std::cout << "record,a,b,c,d,e\n";
  for (double const step : {1.0, 0.1, 0.01}) {
    std::vector<std::array<double, 3>> path;
    Crossing const crossing =
        traceToHeight(step, step == 0.1 ? &path : nullptr);
    std::cout << "gradient," << step << ',' << crossing.xM << ','
              << crossing.timeS << ',' << crossing.sinTheta << ','
              << crossing.steps << "\n";
    for (auto const& sample : path) {
      std::cout << "path," << step << ',' << sample[0] << ',' << sample[1]
                << ',' << sample[2] << ",0\n";
    }
  }
  for (double const angle : {0.0, 10.0, 20.0, 29.0, 31.0, 45.0}) {
    emitInterfaceCase(angle);
  }
  return 0;
}
