/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
#include "ProductionScalarCascadeStepper.hpp"
#endif
#include "ScalarStepFixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <limits>

using namespace corsika;
namespace cp = corsika::kokkos;
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
namespace reference = corsika::production_stepper_reference;
#endif
namespace fixture = scalar_fixture;

TEST_CASE("Scalar stepping preserves native history events", "[scalar_stepper]") {
  using HistoryStack = test::TransportStack<DummyEnvironment>::history_type;
  for (auto mode : {fixture::Mode::Interaction, fixture::Mode::Decay,
                    fixture::Mode::ForcedInteraction, fixture::Mode::ForcedDecay}) {
    for (std::uint64_t seed : {1, 17, 99}) {
      CAPTURE(mode, seed);
      auto a = fixture::run_stepper<cp::ScalarCascadeStepper, HistoryStack>(mode, seed);
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
      auto b =
          fixture::run_stepper<reference::ScalarCascadeStepper, HistoryStack>(mode, seed);
#endif
      auto c = fixture::run_native<HistoryStack>(mode, seed);
      REQUIRE_FALSE(a.threw_);
      REQUIRE(
          a.events_.size() ==
          (mode == fixture::Mode::Decay || mode == fixture::Mode::ForcedDecay ? 1 : 2));
      auto expected = mode == fixture::Mode::Decay || mode == fixture::Mode::ForcedDecay
                          ? history::EventType::Decay
                          : history::EventType::Interaction;
      // Stack iteration skips erased parents; every remaining entry is a child.
      for (std::size_t i = 0; i < a.events_.size(); ++i) {
        CHECK(a.events_[i][0] == static_cast<int>(expected));
      }
      std::vector<fixture::Trace const*> comparisons{&c};
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
      comparisons.push_back(&b);
#endif
      for (auto const* other : comparisons) {
        CHECK(a.calls_ == other->calls_);
        CHECK(a.arguments_ == other->arguments_);
        CHECK(a.particles_ == other->particles_);
        CHECK(a.events_ == other->events_);
        CHECK(a.rng_ == other->rng_);
      }
    }
  }
}

namespace {
  template <template <typename, typename, typename> typename Stepper>
  void check_node_and_force_contracts() {
    auto& rng = RNGManager<>::getInstance();
    if (!rng.isRegistered("cascade")) rng.registerRandomStream("cascade");
    auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    fixture::Stack stack;
    fixture::Trace trace;
    auto p = stack.addParticle(std::make_tuple(Code::Proton, 10_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 0_m, 0_m, 0_m), 0_s));
    auto q = stack.addParticle(std::make_tuple(Code::Photon, 10_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 0_m, 0_m, 0_m), 0_s));
    p.setNode(environment->getUniverse().get());
    q.setNode(nullptr);
    fixture::Tracking tracking{fixture::Mode::Continuous,
                               environment->getUniverse().get(), trace};
    fixture::Processes process{fixture::Mode::Continuous, node, trace};
    Stepper<decltype(tracking), decltype(process), fixture::Stack> stepper(
        *environment, tracking, process, stack);
    auto before = rng.dumpState().str();
    stepper.setNodes(true);
    CHECK(p.getNode() == environment->getUniverse().get());
    CHECK(q.getNode() == node);
    stepper.setNodes();
    CHECK(p.getNode() == node);
    CHECK(q.getNode() == node);
    CHECK(rng.dumpState().str() == before);
    // Test both conflict orders on independent steppers; retain production
    // semantics.
    stepper.forceInteraction();
    CHECK_THROWS_AS(stepper.forceDecay(), std::runtime_error);
    Stepper<decltype(tracking), decltype(process), fixture::Stack> other(
        *environment, tracking, process, stack);
    other.forceDecay();
    CHECK_THROWS_AS(other.forceInteraction(), std::runtime_error);
    CHECK(rng.dumpState().str() == before);
  }

  template <template <typename, typename, typename> typename Stepper, typename Record>
  fixture::Trace invalid_commit(int kind, bool validFirst) {
    auto [environment, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    fixture::Stack stack;
    fixture::Trace trace;
    auto p = stack.addParticle(std::make_tuple(Code::Proton, 100_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 0_m, 0_m, 0_m), 0_s));
    p.setNode(node);
    fixture::Tracking tracking{fixture::Mode::Interaction,
                               environment->getUniverse().get(), trace};
    fixture::Processes process{fixture::Mode::Interaction, node, trace};
    Stepper<decltype(tracking), decltype(process), fixture::Stack> stepper(
        *environment, tracking, process, stack);
    Record bad{22, 0, 1., {0., 0., 1.}};
    // Aggregate initialization works for both independently named wire records.
    auto negative = Record{22, 0, -1., {0., 0., 1.}};
    auto nanEnergy =
        Record{22, 0, std::numeric_limits<double>::quiet_NaN(), {0., 0., 1.}};
    auto infiniteDirection =
        Record{22, 0, 1., {0., std::numeric_limits<double>::infinity(), 1.}};
    bad = kind == 0 ? negative : kind == 1 ? nanEnergy : infiniteDirection;
    std::vector<Record> records;
    if (validFirst) records.push_back(Record{22, 0, 2., {0., 0., 1.}});
    records.push_back(bad);
    auto before = RNGManager<>::getInstance().dumpState().str();
    CHECK_THROWS_AS(stepper.commitDeferredHadronicInteraction(p, records),
                    std::runtime_error);
    CHECK_FALSE(p.isErased());
    CHECK(stack.getEntries() == (validFirst ? 2 : 1));
    CHECK(trace.calls_.empty());
    CHECK(RNGManager<>::getInstance().dumpState().str() == before);
    fixture::capture(stack, node, trace);
    return trace;
  }
} // namespace

TEST_CASE("Logical nodes and forced-step conflicts keep production behavior",
          "[scalar_stepper]") {
  check_node_and_force_contracts<cp::ScalarCascadeStepper>();
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
  check_node_and_force_contracts<reference::ScalarCascadeStepper>();
#endif
}

TEST_CASE("Invalid deferred records do not trigger secondary processing",
          "[scalar_stepper]") {
  for (int kind = 0; kind < 3; ++kind) {
    for (bool first : {false, true}) {
      CAPTURE(kind, first);
      auto a = invalid_commit<cp::ScalarCascadeStepper, cp::HadronicSecondaryRecord>(
          kind, first);
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
      auto b = invalid_commit<reference::ScalarCascadeStepper,
                              reference::HadronicSecondaryRecord>(kind, first);
      CHECK(a.particles_ == b.particles_);
      CHECK(a.calls_ == b.calls_);
#endif
    }
  }
}
