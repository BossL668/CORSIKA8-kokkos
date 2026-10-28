/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <array>
#include <corsika/framework/core/Cascade.hpp>
#include <corsika/kokkos/scheduling/ScalarCascadeStepper.hpp>
#include <corsika/output/DummyOutputManager.hpp>
#include <corsika/setup/SetupTrajectory.hpp>
#include <string>
#include <tests/common/SetupTestEnvironment.hpp>
#include <tests/common/SetupTestTrajectory.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <vector>

namespace scalar_fixture {
  using namespace corsika;
  using Stack = test::TransportStack<DummyEnvironment>::type;
  using Node = DummyEnvironment::BaseNodeType;
  enum class Mode {
    Continuous,
    Geometry,
    ExitWorld,
    Interaction,
    NullInteraction,
    Decay,
    NullDecay,
    ForcedInteraction,
    ForcedDecay,
    SelfDecay,
    Absorbed,
    AlreadyErased,
    TiedContinuousGeometry
  };
  struct Trace {
    std::vector<std::string> calls_;
    std::vector<double> arguments_;
    std::vector<std::array<double, 13>> particles_;
    std::string rng_;
    std::vector<std::array<int, 4>> events_;
    bool threw_ = false;
  };
  template <typename TStack>
  void capture(TStack& stack, Node const* node, Trace& trace) {
    for (auto const& p : stack) {
      auto pos = p.getPosition().getCoordinates();
      auto dir = p.getDirection().getComponents();
      trace.particles_.push_back(
          {static_cast<double>(get_PDG(p.getPID())), p.getKineticEnergy() / 1_GeV,
           p.getTime() / 1_s, pos[0] / 1_m, pos[1] / 1_m, pos[2] / 1_m,
           dir[0].magnitude(), dir[1].magnitude(), dir[2].magnitude(), p.getWeight(),
           static_cast<double>(p.isErased()), p.getNode() == node ? 1. : 0.,
           static_cast<double>(p.getGeneration())});
      if constexpr (TStack::stack_view_type::has_event) {
        auto event = p.getEvent();
        trace.events_.push_back(
            {event ? static_cast<int>(event->eventType()) : -1,
             event ? static_cast<int>(event->secondaries().size()) : 0,
             p.getParentEventIndex(), event && event->hasParentEvent() ? 1 : 0});
      }
    }
  }
  struct Tracking {
    Mode mode_;
    Node const* universe_;
    Trace& trace_;
    static std::string getName() { return "controlled scalar-step audit"; }
    static std::string getVersion() { return "1"; }
    template <typename Particle>
    auto getTrack(Particle const& p) {
      trace_.calls_.push_back("track");
      auto velocity = p.getMomentum() / p.getEnergy() * constants::c;
      auto length = (mode_ == Mode::Geometry || mode_ == Mode::ExitWorld ||
                     mode_ == Mode::TiedContinuousGeometry)
                        ? 1_m
                        : 10000_m;
      auto track = setup::testing::make_track<setup::Trajectory>(
          Line(p.getPosition(), velocity), length / velocity.getNorm());
      return std::make_tuple(track, mode_ == Mode::ExitWorld ? universe_ : p.getNode());
    }
  };
  struct Processes {
    Mode mode_;
    Node const* node_;
    Trace& trace_;
    template <typename Particle>
    CrossSectionType getCrossSection(Particle const&, Code, FourMomentum const&) {
      trace_.calls_.push_back("cross-section");
      return (mode_ == Mode::Interaction || mode_ == Mode::NullInteraction ||
              mode_ == Mode::ForcedInteraction)
                 ? 1.e12_mb
                 : 0_mb;
    }
    template <typename Particle>
    InverseTimeType getInverseLifetime(Particle const&) {
      trace_.calls_.push_back("inverse-lifetime");
      return (mode_ == Mode::Decay || mode_ == Mode::NullDecay ||
              mode_ == Mode::ForcedDecay || mode_ == Mode::SelfDecay)
                 ? 1.e9 / 1_s
                 : 0. / 1_s;
    }
    template <typename Particle, typename Track>
    ContinuousProcessStepLength getMaxStepLength(Particle&, Track&) {
      trace_.calls_.push_back("continuous-limit");
      auto length =
          (mode_ == Mode::Continuous || mode_ == Mode::Absorbed ||
           mode_ == Mode::AlreadyErased || mode_ == Mode::TiedContinuousGeometry)
              ? 1_m
              : 100000_m;
      return {length, ContinuousProcessIndex(this)};
    }
    template <typename Particle>
    ProcessReturn doContinuous(Step<Particle>& step, ContinuousProcessIndex id) {
      trace_.calls_.push_back("continuous");
      trace_.arguments_.push_back(id.getIndex() == this ? 1. : 0.);
      trace_.arguments_.push_back(step.getDiffT() / 1_s);
      trace_.arguments_.push_back(step.getDisplacement().getNorm() / 1_m);
      step.add_dEkin(-.25_GeV);
      if (mode_ == Mode::AlreadyErased) {
        auto particle = step.getParticlePre();
        particle.erase();
      }
      return mode_ == Mode::Absorbed || mode_ == Mode::AlreadyErased
                 ? ProcessReturn::ParticleAbsorbed
                 : ProcessReturn::Ok;
    }
    template <typename View, typename Rng>
    ProcessReturn selectInteraction(View& view, FourMomentum const& p4,
                                    NuclearComposition const&, Rng& rng,
                                    CrossSectionType sample) {
      trace_.calls_.push_back("interaction");
      trace_.arguments_.push_back(p4.getTimeLikeComponent() / 1_GeV);
      trace_.arguments_.push_back(sample / 1_mb);
      trace_.arguments_.push_back(static_cast<double>(rng()));
      if (mode_ == Mode::NullInteraction) return ProcessReturn::Ok;
      view.addSecondary(
          std::make_tuple(Code::Photon, 2_GeV, view.getProjectile().getDirection()));
      view.addSecondary(
          std::make_tuple(Code::Electron, 3_GeV, view.getProjectile().getDirection()));
      return ProcessReturn::Interacted;
    }
    template <typename View>
    ProcessReturn selectDecay(View& view, InverseTimeType sample) {
      trace_.calls_.push_back("decay");
      trace_.arguments_.push_back(sample * 1_s);
      if (mode_ == Mode::NullDecay) return ProcessReturn::Ok;
      view.addSecondary(
          std::make_tuple(mode_ == Mode::SelfDecay ? Code::MuMinus : Code::Electron,
                          4_GeV, view.getProjectile().getDirection()));
      return ProcessReturn::Decayed;
    }
    template <typename View>
    void doSecondaries(View& view) {
      trace_.calls_.push_back("secondaries");
      for (auto& p : view) p.setWeight(2.);
    }
    template <typename Particle>
    void doBoundaryCrossing(Particle& p, Node const&, Node const&) {
      trace_.calls_.push_back("boundary");
      trace_.arguments_.push_back(p.isErased() ? 1. : 0.);
    }
    void initCascadeEquations() {}
    template <typename TStack>
    void doCascadeEquations(TStack&) {}
    template <typename TStack>
    void doStack(TStack& stack) {
      // Stop the native public Cascade after precisely the step under comparison.
      capture(stack, node_, trace_);
      for (auto& p : stack)
        if (!p.isErased()) p.erase();
    }
  };

  template <template <typename, typename, typename> typename Stepper,
            typename TStack = Stack>
  Trace run_stepper(Mode mode, std::uint64_t seed, bool timing = false) {
    auto& manager = RNGManager<>::getInstance();
    if (!manager.isRegistered("cascade")) manager.registerRandomStream("cascade");
    manager.setSeed(seed);
    manager.getRandomStream("cascade").reset();
    auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    TStack stack;
    Trace trace;
    auto p = stack.addParticle(std::make_tuple(Code::MuMinus, 100_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 0_m, 0_m, 0_m), 0_s));
    p.setWeight(3.);
    Tracking tracking{mode, environment->getUniverse().get(), trace};
    Processes process{mode, node, trace};
    Stepper<Tracking, Processes, TStack> stepper(*environment, tracking, process, stack);
    stepper.setNodes();
    stepper.enableDetailedPhaseTiming(timing);
    if (mode == Mode::ForcedInteraction) stepper.forceInteraction();
    if (mode == Mode::ForcedDecay) stepper.forceDecay();
    try {
      stepper.advance(p);
    } catch (std::runtime_error const&) { trace.threw_ = true; }
    capture(stack, node, trace);
    trace.rng_ = manager.dumpState().str();
    return trace;
  }

  template <typename TStack = Stack>
  Trace run_native(Mode mode, std::uint64_t seed) {
    auto& manager = RNGManager<>::getInstance();
    if (!manager.isRegistered("cascade")) manager.registerRandomStream("cascade");
    manager.setSeed(seed);
    manager.getRandomStream("cascade").reset();
    auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    TStack stack;
    Trace trace;
    auto p = stack.addParticle(std::make_tuple(Code::MuMinus, 100_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 0_m, 0_m, 0_m), 0_s));
    p.setWeight(3.);
    Tracking tracking{mode, environment->getUniverse().get(), trace};
    Processes process{mode, node, trace};
    DummyOutputManager output;
    Cascade cascade(*environment, tracking, process, output, stack);
    if (mode == Mode::ForcedInteraction) cascade.forceInteraction();
    if (mode == Mode::ForcedDecay) cascade.forceDecay();
    try {
      cascade.run();
    } catch (std::runtime_error const&) {
      trace.threw_ = true;
      capture(stack, node, trace);
    }
    trace.rng_ = manager.dumpState().str();
    return trace;
  }
} // namespace scalar_fixture
