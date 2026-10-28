/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <testCascade.hpp>

#include <corsika/framework/core/Cascade.hpp>

#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/NullModel.hpp>
#include <corsika/modules/StackInspector.hpp>

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/Logging.hpp>

#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>

#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/NuclearComposition.hpp>

#include <corsika/output/DummyOutputManager.hpp>

#include <SetupTestTrajectory.hpp>
#include <corsika/setup/SetupTrajectory.hpp>

#include <catch2/catch_all.hpp>

using namespace corsika;

#include <limits>
using namespace std;
using Catch::Approx;

/**
 * testCascade implements an e.m. Heitler model with energy splitting
 * and a critical energy.
 *
 * It resembles one of the most simple cascades you can simulate with CORSIKA8.
 */

/*
  The dummy env (here) doesn't need to have any propoerties
 */

auto make_dummy_env() {
  TestEnvironmentType env; // dummy environment
  auto& universe = *(env.getUniverse());

  auto world = TestEnvironmentType::createNode<Sphere>(
      Point{env.getCoordinateSystem(), 0_m, 0_m, 0_m},
      1_km * std::numeric_limits<double>::infinity());

  NuclearComposition const composition({Code::Proton}, {1.});
  world->setModelProperties<TestEnvironmentInterface>(19.2_g / cube(1_cm), composition);

  universe.addChild(std::move(world));
  return env;
}

/**
 *
 * For the Heitler model we don't need particle transport.
 */
class DummyTracking {

public:
  template <typename TParticle>
  auto getTrack(TParticle const& particle) {
    calls_++;
    VelocityVector const initialVelocity =
        particle.getMomentum() / particle.getEnergy() * constants::c;
    Line const theLine = Line(particle.getPosition(), initialVelocity);
    TimeType const tEnd = std::numeric_limits<TimeType::value_type>::infinity() * 1_s;
    return std::make_tuple(
        corsika::setup::testing::make_track<setup::Trajectory>(theLine, tEnd),
        // trajectory: just go ahead forever
        particle.getNode()); // next volume node
  }
  int getCalls() const { return calls_; }
  static std::string getName() { return "DummyTracking"; }
  static std::string getVersion() { return "1.0.0"; }
  int calls_ = 0;
};

class ProcessSplit : public InteractionProcess<ProcessSplit> {

public:
  CrossSectionType getCrossSection(Code const, Code const, FourMomentum const&,
                                   FourMomentum const&) const {
    return 1_mb;
  }

  template <typename TView>
  void doInteraction(TView& view, Code, Code, FourMomentum const&, FourMomentum const&) {
    ++calls_;
    auto vP = view.getProjectile();
    const HEPEnergyType Ekin = vP.getKineticEnergy();
    vP.addSecondary(
        std::make_tuple(vP.getPID(), Ekin / 2, vP.getMomentum().normalized()));
    vP.addSecondary(
        std::make_tuple(vP.getPID(), Ekin / 2, vP.getMomentum().normalized()));
  }

  int getCalls() const { return calls_; }

private:
  int calls_ = 0;
};

class ProcessCut : public SecondariesProcess<ProcessCut> {

public:
  ProcessCut(HEPEnergyType const e)
      : Ecrit_(e) {}

  template <typename TStack>
  void doSecondaries(TStack& vS) {
    calls_++;
    auto p = vS.begin();
    while (p != vS.end()) {
      HEPEnergyType E = p.getEnergy();
      if (E < Ecrit_) {
        p.erase();
        count_++;
      }
      ++p; // next particle
    }
    CORSIKA_LOG_DEBUG("ProcessCut::doSecondaries size={} count={}", vS.getEntries(),
                      count_);
  }

  int getCount() const { return count_; }
  int getCalls() const { return calls_; }

private:
  int count_ = 0;
  int calls_ = 0;
  HEPEnergyType Ecrit_;
};

class ProcessZero : public InteractionProcess<ProcessZero> {

public:
  CrossSectionType getCrossSection(Code const, Code const, FourMomentum const&,
                                   FourMomentum const&) const {
    return 0_mb;
  }

  template <typename TView>
  void doInteraction([[maybe_unused]] TView& view, Code, Code, FourMomentum const&,
                     FourMomentum const&) {
    FAIL("doInteraction of ProcessZero has been called! This should never happen.");
  }
};

// Continuous process that does nothing for the first `maxCalls`-1 calls, but absorbs
// the particle when called for the `maxCalls` time
class ContinuousCounter : public ContinuousProcess<ContinuousCounter> {
public:
  ContinuousCounter(int maxCalls)
      : maxCalls_(maxCalls){};

  template <typename D>
  ProcessReturn doContinuous([[maybe_unused]] Step<D>& d, [[maybe_unused]] bool flag) {
    if (++calls_ == maxCalls_) return ProcessReturn::ParticleAbsorbed;
    return ProcessReturn::Ok;
  }

  template <typename TParticle, typename TTrack>
  LengthType getMaxStepLength(TParticle&, TTrack&) {
    return meter * std::numeric_limits<double>::infinity();
  }

  int getCalls() const { return calls_; }

private:
  const int maxCalls_;
  int calls_ = 0;
};

class DummyDecay : public DecayProcess<DummyDecay> {
  // Dummy decay process that puts pre-predefined particles on stack
public:
  DummyDecay(Code code, HEPEnergyType ek, DirectionVector dir)
      : code_(code)
      , e0_(ek)
      , dir_(dir) {}

  Code code_;
  HEPEnergyType e0_;
  DirectionVector dir_;

  template <typename Particle>
  TimeType getLifetime(Particle&) const {
    return 1_s;
  }

  template <typename TView>
  void doDecay(TView& view) const {
    auto projectile = view.getProjectile();
    auto const dir = projectile.getMomentum().normalized();
    projectile.addSecondary(std::make_tuple(code_, e0_, dir_));
    CORSIKA_LOG_INFO("Particle stack size {}", view.getSize());
  }
};

TEST_CASE("Cascade", "[Cascade]") {

  logging::set_level(logging::level::info);

  HEPEnergyType E0 = 100_GeV;

  auto& rmng = RNGManager<>::getInstance();
  rmng.registerRandomStream("cascade");

  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();

  // Properties of primary particle
  auto const primCode = Code::Electron;
  auto const primEk = E0 - get_mass(Code::Electron);
  auto const primDir = DirectionVector(rootCS, {0, 0, -1});

  StackInspector<TestCascadeStack> stackInspect(100, true, E0);
  NullModel nullModel;
  DummyDecay decay(primCode, primEk, primDir); // decay product will be primary

  HEPEnergyType const Ecrit = 85_MeV;
  ProcessSplit split;
  ProcessCut cut(Ecrit);
  auto sequence = make_sequence(nullModel, decay, stackInspect, split, cut);
  TestCascadeStack stack;

  stack.clear();
  stack.addParticle(
      std::make_tuple(primCode, primEk, primDir, Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

  DummyTracking tracking;
  DummyOutputManager output;
  Cascade<DummyTracking, decltype(sequence), DummyOutputManager, TestCascadeStack> EAS(
      env, tracking, sequence, output, stack);

  SECTION("full cascade") {
    EAS.run();
    CHECK(tracking.getCalls() == 2047);
    CHECK(cut.getCount() == 2048);
    CHECK(cut.getCalls() == 2047); // final particle is still on stack and not yet deleted
    CHECK(split.getCalls() == 2047);
  }

  SECTION("forced interaction") {
    CHECK(tracking.getCalls() == 0);
    EAS.forceInteraction();
    CHECK(tracking.getCalls() == 0);
    CHECK(stack.getEntries() == 1);
    CHECK(stack.getSize() == 1);
    EAS.run();
    CHECK(tracking.getCalls() == 2046); // one LESS than without forceInteraction
    CHECK(stack.getEntries() == 0);
    CHECK(stack.getSize() == 13);
    CHECK(split.getCalls() == 2047);
  }

  SECTION("double_forcing_1") {
    EAS.forceInteraction();
    REQUIRE_THROWS(EAS.forceDecay());
  }

  SECTION("double_forcing_2") {
    EAS.forceDecay();
    REQUIRE_THROWS(EAS.forceInteraction());
  }

  SECTION("forced decay") {
    // Put anything new so that it won't trigger "decays into self" error
    stack.clear();
    stack.addParticle(std::make_tuple(Code::PiPlus,
                                      0.5_GeV, // Ekin
                                      DirectionVector(rootCS, {0, 0, -1}),
                                      Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

    CHECK(tracking.getCalls() == 0);
    EAS.forceDecay();
    CHECK(tracking.getCalls() == 0);
    CHECK(stack.getEntries() == 1);
    CHECK(stack.getSize() == 1);
    EAS.run();
    CHECK(tracking.getCalls() == 2047);
    CHECK(stack.getEntries() == 0);
    CHECK(stack.getSize() == 14);
    CHECK(split.getCalls() == 2047);
  }
}

TEST_CASE("Cascade Zero Interaction", "[Cascade]") {
  // In this test, we have an interaction with a crosssection of 0_mb, therefore this
  // should never be called. This test checks that this is indeed the case.
  // We also check that the particle is not erased too early - If no interaction can be
  // called, the particle should stay on the stack until we delete it with the
  // `ContinuousCounter` process (after 100 iterations).

  logging::set_level(logging::level::info);
  HEPEnergyType E0 = 100_GeV;

  auto& rmng = RNGManager<>::getInstance();
  rmng.registerRandomStream("cascade");

  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();

  ProcessZero zero; // process that has a crosssection of zero. should not be called!
  ContinuousCounter counter{100}; // limited to 100 calls to avoid endless loop
  auto sequence = make_sequence(zero, counter);
  TestCascadeStack stack;
  stack.clear();
  stack.addParticle(std::make_tuple(Code::Electron,
                                    E0 - get_mass(Code::Electron), // Ekin
                                    DirectionVector(rootCS, {0, 0, -1}),
                                    Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

  DummyTracking tracking;
  DummyOutputManager output;
  Cascade<DummyTracking, decltype(sequence), DummyOutputManager, TestCascadeStack> EAS(
      env, tracking, sequence, output, stack);

  CHECK(counter.getCalls() == 0);
  EAS.run();
  // expect 100 calls. if it would be less, this means that the particle has been erased
  // early by the Cascade.inl algorithm
  CHECK(counter.getCalls() == 100);
}
