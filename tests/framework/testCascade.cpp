/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <testCascade.hpp>

#include <corsika/framework/core/Cascade.hpp>
#include <corsika/framework/core/HybridCascade.hpp>
#include <corsika/framework/core/ScalarCascadeStepper.hpp>

#include <corsika/framework/process/ProcessSequence.hpp>
#include <corsika/framework/process/NullModel.hpp>
#include <corsika/setup/SetupStack.hpp>
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

TEST_CASE("ScalarCascadeStepper advances one particle", "[ScalarCascadeStepper]") {
  logging::set_level(logging::level::info);

  auto& rmng = RNGManager<>::getInstance();
  rmng.registerRandomStream("cascade");

  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();

  ProcessZero zero;
  ContinuousCounter counter{1};
  auto sequence = make_sequence(zero, counter);

  TestCascadeStack stack;
  stack.addParticle(std::make_tuple(
      Code::Electron, 100_GeV - get_mass(Code::Electron),
      DirectionVector(rootCS, {0, 0, -1}), Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

  DummyTracking tracking;
  ScalarCascadeStepper<DummyTracking, decltype(sequence), TestCascadeStack> stepper(
      env, tracking, sequence, stack);

  stepper.setNodes();
  auto particle = stack.getNextParticle();
  stepper.advance(particle);

  CHECK(tracking.getCalls() == 1);
  CHECK(counter.getCalls() == 1);
  CHECK(stack.getEntries() == 0);
}

TEST_CASE("Transport identity follows particle lineage", "[TransportIdentity]") {
  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();
  auto const direction = DirectionVector(rootCS, {0, 0, -1});
  auto const position = Point(rootCS, {0_m, 0_m, 10_km});

  TestCascadeIdentityStack stack;
  auto primary = stack.addParticle(std::make_tuple(
      Code::Electron, 100_GeV - get_mass(Code::Electron), direction, position, 0_ns));

  CHECK(primary.getHistoryId() == 1);
  CHECK(primary.getParentHistoryId() == transport::NoParentHistoryId);
  CHECK(primary.getGeneration() == 0);
  CHECK(primary.getStepId() == 0);

  TestCascadeIdentityStack::stack_view_type secondaries{primary};
  auto child = secondaries.addSecondary(
      std::make_tuple(Code::Electron, 50_GeV, direction));

  CHECK(child.getHistoryId() == 2);
  CHECK(child.getParentHistoryId() == primary.getHistoryId());
  CHECK(child.getGeneration() == 1);
  CHECK(child.getStepId() == 0);

  auto childOnMainStack = stack.last();
  auto grandchild = childOnMainStack.addSecondary(
      std::make_tuple(Code::Photon, 25_GeV, direction));

  CHECK(grandchild.getHistoryId() == 3);
  CHECK(grandchild.getParentHistoryId() == childOnMainStack.getHistoryId());
  CHECK(grandchild.getGeneration() == 2);

  CHECK(primary.beginTransportStep() == 0);
  CHECK(primary.beginTransportStep() == 1);
  CHECK(primary.getStepId() == 2);

  childOnMainStack.erase();
  stack.purge();
  REQUIRE(stack.getSize() == 2);
  CHECK(stack.at(0).getHistoryId() == 1);
  CHECK(stack.at(1).getHistoryId() == 3);

  stack.clear();
  auto nextPrimary = stack.addParticle(std::make_tuple(
      Code::Photon, 10_GeV, direction, position, 0_ns));
  CHECK(nextPrimary.getHistoryId() == 1);

  nextPrimary.setTransportIdentity(
      transport::TransportIdentity{42, 7, 3, 9});
  CHECK(nextPrimary.getHistoryId() == 42);
  CHECK(nextPrimary.getParentHistoryId() == 7);
  CHECK(nextPrimary.getGeneration() == 3);
  CHECK(nextPrimary.getStepId() == 9);

  auto importedChild = nextPrimary.addSecondary(
      std::make_tuple(Code::Electron, 5_GeV, direction));
  CHECK(importedChild.getHistoryId() == 43);
  CHECK(importedChild.getParentHistoryId() == 42);
  CHECK(importedChild.getGeneration() == 4);
  CHECK_THROWS(nextPrimary.setTransportIdentity(
      transport::TransportIdentity{44, 0, 2, 0}));
}

TEST_CASE("CPU-only scheduler exposes a deterministic scalar wavefront",
          "[CpuOnlyWavefrontScheduler]") {
  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();

  TestCascadeIdentityStack stack;
  stack.addParticle(std::make_tuple(
      Code::Electron, 10_GeV - get_mass(Code::Electron),
      DirectionVector(rootCS, {0, 0, -1}), Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

  CpuOnlyWavefrontScheduler<TestCascadeIdentityStack> scheduler{stack};
  auto scheduled = scheduler.acquireNext();

  CHECK(scheduled.history_id == 1);
  CHECK(scheduled.generation == 0);
  CHECK(scheduled.step_id == 0);
  CHECK(scheduled.particle.getStepId() == 1);
  CHECK(scheduler.statistics().acquired_particle_steps == 1);
  CHECK(scheduler.statistics().completed_particle_steps == 0);
  CHECK(scheduler.statistics().max_wavefront_size == 1);
  CHECK_THROWS(scheduler.acquireNext());

  scheduler.completeParticleStep();
  CHECK(scheduler.statistics().completed_particle_steps == 1);
  CHECK_THROWS(scheduler.completeParticleStep());

  scheduled.particle.erase();
  CHECK(scheduler.empty());
  CHECK_THROWS(scheduler.acquireNext());
}

TEST_CASE("CPU wavefront scheduler parks deferred histories",
          "[CpuOnlyWavefrontScheduler]") {
  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();
  auto const direction =
      DirectionVector(rootCS, {0, 0, -1});
  auto const position =
      Point(rootCS, {0_m, 0_m, 10_km});

  TestCascadeIdentityStack stack;
  auto lower = stack.addParticle(std::make_tuple(
      Code::Proton, 10_GeV - get_mass(Code::Proton),
      direction, position, 0_ns));
  auto upper = stack.addParticle(std::make_tuple(
      Code::PiPlus, 5_GeV - get_mass(Code::PiPlus),
      direction, position, 0_ns));

  CpuOnlyWavefrontScheduler<TestCascadeIdentityStack>
      scheduler{stack};
  auto first = scheduler.acquireNext();
  CHECK(first.history_id == upper.getHistoryId());
  scheduler.completeParticleStep(true);
  CHECK(scheduler.hasSuspendedParticles());
  CHECK(scheduler.suspendedParticleCount() == 1);

  auto second = scheduler.acquireNext();
  CHECK(second.history_id == lower.getHistoryId());
  scheduler.completeParticleStep();
  CHECK_FALSE(scheduler.empty());

  scheduler.resumeParticle(first.history_id);
  CHECK_FALSE(scheduler.hasSuspendedParticles());
  auto resumed = scheduler.acquireNext();
  CHECK(resumed.history_id == first.history_id);
  scheduler.completeParticleStep();
  CHECK(scheduler.statistics().suspended_particles == 1);
  CHECK(scheduler.statistics().resumed_particles == 1);
  CHECK(
      scheduler.statistics().maximum_suspended_particles ==
      1);
  CHECK_THROWS(
      scheduler.resumeParticle(first.history_id));
}

TEST_CASE("Production hybrid stack composes identity and scalar transport",
          "[HybridStack]") {
  auto& rmng = RNGManager<>::getInstance();
  rmng.registerRandomStream("cascade");

  auto env = make_dummy_env();
  auto const& rootCS = env.getCoordinateSystem();

  ProcessZero zero;
  ContinuousCounter counter{1};
  auto sequence = make_sequence(zero, counter);

  setup::HybridStack<TestEnvironmentType> stack;
  auto primary = stack.addParticle(std::make_tuple(
      Code::Electron, 10_GeV - get_mass(Code::Electron),
      DirectionVector(rootCS, {0, 0, -1}), Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));
  CHECK(primary.getHistoryId() == 1);

  DummyTracking tracking;
  DummyOutputManager output;
  HybridCascade<DummyTracking, decltype(sequence), DummyOutputManager,
                setup::HybridStack<TestEnvironmentType>>
      cascade(env, tracking, sequence, output, stack);
  cascade.run();

  CHECK(counter.getCalls() == 1);
  CHECK(tracking.getCalls() == 1);
  CHECK(cascade.schedulerStatistics().acquired_particle_steps == 1);
  CHECK(cascade.schedulerStatistics().completed_particle_steps == 1);

  using HybridHistoryStack = setup::detail::StackGenerator<
      TestEnvironmentType>::StackWithTransportIdentityAndHistory;
  HybridHistoryStack historyStack;
  auto historyPrimary = historyStack.addParticle(std::make_tuple(
      Code::Photon, 10_GeV, DirectionVector(rootCS, {0, 0, -1}),
      Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));
  HybridHistoryStack::stack_view_type historySecondaries{historyPrimary};
  auto historyChild = historySecondaries.addSecondary(
      std::make_tuple(Code::Electron, 5_GeV,
                      DirectionVector(rootCS, {0, 0, -1})));

  CHECK(historyPrimary.getHistoryId() == 1);
  CHECK(historyChild.getHistoryId() == 2);
  CHECK(historyChild.getParentHistoryId() == 1);
  CHECK(historyChild.getEvent() != nullptr);
}

TEST_CASE("HybridCascade scalar compatibility is RNG-equivalent", "[HybridCascade]") {
  struct RunResult {
    int tracking_calls;
    int split_calls;
    int cut_count;
    int cut_calls;
    std::string rng_state;
    CpuOnlyWavefrontStatistics scheduler;
  };

  auto& rmng = RNGManager<>::getInstance();
  rmng.registerRandomStream("cascade");
  constexpr RNGManager<>::seed_type seed = 0x5eed1234;

  auto resetRng = [&]() {
    rmng.setSeed(seed);
    rmng.getRandomStream("cascade").reset();
  };

  auto runScalar = [&]() {
    resetRng();
    auto env = make_dummy_env();
    auto const& rootCS = env.getCoordinateSystem();
    auto const direction = DirectionVector(rootCS, {0, 0, -1});
    HEPEnergyType const primaryEnergy = 100_GeV;
    HEPEnergyType const primaryKineticEnergy =
        primaryEnergy - get_mass(Code::Electron);

    NullModel nullModel;
    DummyDecay decay(Code::Electron, primaryKineticEnergy, direction);
    ProcessSplit split;
    ProcessCut cut(85_MeV);
    auto sequence = make_sequence(nullModel, decay, split, cut);

    TestCascadeStack stack;
    stack.addParticle(std::make_tuple(
        Code::Electron, primaryKineticEnergy, direction,
        Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

    DummyTracking tracking;
    DummyOutputManager output;
    Cascade<DummyTracking, decltype(sequence), DummyOutputManager, TestCascadeStack>
        cascade(env, tracking, sequence, output, stack);
    cascade.run();

    return RunResult{tracking.getCalls(), split.getCalls(), cut.getCount(),
                     cut.getCalls(), rmng.dumpState().str(), {}};
  };

  auto runHybrid = [&]() {
    resetRng();
    auto env = make_dummy_env();
    auto const& rootCS = env.getCoordinateSystem();
    auto const direction = DirectionVector(rootCS, {0, 0, -1});
    HEPEnergyType const primaryEnergy = 100_GeV;
    HEPEnergyType const primaryKineticEnergy =
        primaryEnergy - get_mass(Code::Electron);

    NullModel nullModel;
    DummyDecay decay(Code::Electron, primaryKineticEnergy, direction);
    ProcessSplit split;
    ProcessCut cut(85_MeV);
    auto sequence = make_sequence(nullModel, decay, split, cut);

    TestCascadeIdentityStack stack;
    stack.addParticle(std::make_tuple(
        Code::Electron, primaryKineticEnergy, direction,
        Point(rootCS, {0_m, 0_m, 10_km}), 0_ns));

    DummyTracking tracking;
    DummyOutputManager output;
    HybridCascade<DummyTracking, decltype(sequence), DummyOutputManager,
                  TestCascadeIdentityStack>
        cascade(env, tracking, sequence, output, stack);
    cascade.run();

    return RunResult{tracking.getCalls(), split.getCalls(), cut.getCount(),
                     cut.getCalls(), rmng.dumpState().str(),
                     cascade.schedulerStatistics()};
  };

  auto const scalar = runScalar();
  auto const hybrid = runHybrid();

  CHECK(hybrid.tracking_calls == scalar.tracking_calls);
  CHECK(hybrid.split_calls == scalar.split_calls);
  CHECK(hybrid.cut_count == scalar.cut_count);
  CHECK(hybrid.cut_calls == scalar.cut_calls);
  CHECK(hybrid.rng_state == scalar.rng_state);

  CHECK(hybrid.tracking_calls == 2047);
  CHECK(hybrid.split_calls == 2047);
  CHECK(hybrid.cut_count == 2048);
  CHECK(hybrid.cut_calls == 2047);
  CHECK(hybrid.scheduler.acquired_particle_steps == 2047);
  CHECK(hybrid.scheduler.completed_particle_steps == 2047);
  CHECK(hybrid.scheduler.max_wavefront_size == 1);
}
