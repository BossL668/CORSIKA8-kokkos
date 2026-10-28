/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "compareHybrid.hpp"
#include <algorithm>
#include <cstdlib>

namespace hybrid_fixture {
  template <bool Old>
  Result scalar(scalar_fixture::Mode mode, std::uint64_t seed, bool detailed) {
    reset_rng(seed);
    auto [environment, coordinates, node] =
        setup::testing::setup_environment(Code::Oxygen);
    Stack stack;
    Result result;
    auto particle = stack.addParticle(std::make_tuple(
        Code::MuMinus, 100_GeV, DirectionVector(*coordinates, {0., 0., 1.}),
        Point(*coordinates, 0_m, 0_m, 0_m), 0_s));
    particle.setWeight(3.);
    scalar_fixture::Tracking tracking{mode, environment->getUniverse().get(),
                                      result.trace_};
    Model<Old> model;
    Counter<Old> counter(model);
    Processes<Old> process(mode, node, result, counter, false, false);
    Output output{result.trace_};
    typename Implementation<Old>::template Cascade<decltype(tracking), decltype(process),
                                                   Output, Stack>
        cascade(*environment, tracking, process, output, stack);
    cascade.enableScalarDetailedPhaseTiming(detailed);
    if (mode == scalar_fixture::Mode::ForcedInteraction) cascade.forceInteraction();
    if (mode == scalar_fixture::Mode::ForcedDecay) cascade.forceDecay();
    try {
      cascade.run();
    } catch (std::runtime_error const&) { result.trace_.threw_ = true; }
    capture(stack, node, result);
    result.trace_.rng_ = RNGManager<>::getInstance().dumpState().str();
    snapshot(cascade.schedulerStatistics(), result);
    snapshot(cascade.timingStatistics(), result);
    return result;
  }

  template <bool Old, bool Basic>
  Result routed(unsigned scenario, std::uint64_t seed) {
    reset_rng(seed);
    auto [environment, coordinates, node] =
        setup::testing::setup_environment(Code::Oxygen);
    Stack stack;
    Result result;
    for (auto code : {Code::Photon, Code::Proton, Code::MuMinus}) {
      auto particle = stack.addParticle(
          std::make_tuple(code, 100_GeV, DirectionVector(*coordinates, {0., 0., 1.}),
                          Point(*coordinates, 0_m, 0_m, 0_m), 0_s));
      particle.setWeight(3.);
    }
    auto const mode = scalar_fixture::Mode::Absorbed;
    scalar_fixture::Tracking tracking{mode, environment->getUniverse().get(),
                                      result.trace_};
    Model<Old> model;
    Counter<Old> counter(model);
    Processes<Old> process(mode, node, result, counter, false, true);
    Output output{result.trace_};
    using RouterType = std::conditional_t<Basic, BasicRouter, Router>;
    auto router = [&]() {
      if constexpr (Basic)
        return BasicRouter(result.trace_);
      else
        return Router(result.trace_, *coordinates, node, scenario & 1, scenario & 2,
                      scenario & 4, scenario & 8);
    }();
    typename Implementation<Old>::template Cascade<decltype(tracking), decltype(process),
                                                   Output, Stack, RouterType>
        cascade(*environment, tracking, process, output, stack, router);
    if (scenario & 16) cascade.forceInteraction();
    if (scenario & 32) cascade.forceDecay();
    cascade.run();
    REQUIRE(stack.getEntries() == 0);
    REQUIRE_FALSE(router.pending());
    result.trace_.rng_ = RNGManager<>::getInstance().dumpState().str();
    snapshot(cascade.schedulerStatistics(), result);
    snapshot(cascade.timingStatistics(), result);
    return result;
  }

  template <bool Old>
  Result workers(unsigned trigger, std::uint64_t seed, std::filesystem::path executable) {
    reset_rng(seed);
    auto [environment, coordinates, node] =
        setup::testing::setup_environment(Code::Oxygen);
    Stack stack;
    Result result;
    for (unsigned i = 0; i < 7; ++i) {
      auto particle = stack.addParticle(std::make_tuple(
          i % 2 == 0 ? Code::Proton : Code::PiPlus, (10. + i * 17.) * 1_GeV,
          DirectionVector(*coordinates, {0., 0., 1.}), Point(*coordinates, 0_m, 0_m, 0_m),
          0_s));
      particle.setWeight(3.);
    }
    auto const mode = scalar_fixture::Mode::Interaction;
    scalar_fixture::Tracking tracking{mode, environment->getUniverse().get(),
                                      result.trace_};
    Model<Old> model;
    Counter<Old> counter(model);
    Processes<Old> process(mode, node, result, counter, true, false);
    Output output{result.trace_};
    typename Implementation<Old>::Pool pool(executable, 2);
    typename Implementation<Old>::Config config{
        seed, 55, trigger == 2 ? 100U : 2U, trigger == 0 ? .01 : 1.e6, 2, .1};
    typename Implementation<Old>::template Cascade<decltype(tracking), decltype(process),
                                                   Output, Stack>
        cascade(*environment, tracking, process, output, stack);
    cascade.configureHadronicProcessPool(pool, config);
    cascade.enableScalarDetailedPhaseTiming(true);
    cascade.run();
    REQUIRE(model.calls_ == 0);
    REQUIRE(counter.getCount() == 7);
    REQUIRE(result.commits_.size() == 14);
    REQUIRE(std::count(result.trace_.calls_.begin(), result.trace_.calls_.end(),
                       "secondaries") == 7);
    REQUIRE(stack.getEntries() == 0);
    for (std::size_t i = 0; i < result.commits_.size(); i += 2) {
      CHECK(result.commits_[i][0] == 7 - i / 2);
      CHECK(result.commits_[i][0] == result.commits_[i][2]);
      CHECK(result.commits_[i][3] == 1);
    }
    result.synchronousCalls_ = model.calls_;
    result.countedInteractions_ = counter.getCount();
    result.trace_.rng_ = RNGManager<>::getInstance().dumpState().str();
    snapshot(cascade.schedulerStatistics(), result);
    snapshot(cascade.timingStatistics(), result);
    return result;
  }
} // namespace hybrid_fixture

TEST_CASE("Whole scalar coordinator matches the frozen production cascade", "[hybrid]") {
  using namespace hybrid_fixture;
  using M = scalar_fixture::Mode;
  for (auto mode :
       {M::Continuous, M::Geometry, M::ExitWorld, M::Interaction, M::NullInteraction,
        M::Decay, M::NullDecay, M::ForcedInteraction, M::ForcedDecay, M::Absorbed,
        M::AlreadyErased, M::TiedContinuousGeometry})
    for (auto seed : {19U, 373U})
      for (bool detailed : {false, true}) {
        CAPTURE(static_cast<unsigned>(mode), seed, detailed);
        compare(scalar<false>(mode, seed, detailed), scalar<true>(mode, seed, detailed));
      }
}

TEST_CASE(
    "Router routing interleaving fallback and forced actions retain production order",
    "[hybrid][router]") {
  using namespace hybrid_fixture;
  compare(routed<false, true>(0, 19), routed<true, true>(0, 19));
  for (unsigned scenario : {0U, 1U, 2U, 3U, 6U, 7U, 8U, 10U, 15U, 16U, 17U, 32U, 33U}) {
    CAPTURE(scenario);
    auto actual = routed<false, false>(scenario, 373);
    compare(actual, routed<true, false>(scenario, 373));
    REQUIRE(actual.trace_.calls_.back() == "output-end");
    REQUIRE(std::count(actual.trace_.calls_.begin(), actual.trace_.calls_.end(),
                       "router-end") == 1);
    if (scenario & (16U | 32U)) {
      auto forced = std::find(actual.trace_.calls_.begin(), actual.trace_.calls_.end(),
                              scenario & 16U ? "interaction" : "decay");
      auto staged =
          std::find(actual.trace_.calls_.begin(), actual.trace_.calls_.end(), "stage");
      CHECK(forced < staged);
    }
  }
}

TEST_CASE("Deferred counter worker pool and ordered commit match production",
          "[hybrid][workers]") {
  using namespace hybrid_fixture;
  auto const* path = std::getenv("HYBRID_TEST_WORKER");
  REQUIRE(path != nullptr);
  for (unsigned trigger : {0U, 1U, 2U}) {
    CAPTURE(trigger);
    auto current = workers<false>(trigger, 827, path);
    compare(current, workers<true>(trigger, 827, path));
    REQUIRE_FALSE(current.loads_.empty());
    CHECK(current.loads_.front().front() == trigger);
  }
}

TEST_CASE("Coordinator retains mutually exclusive forced controls", "[hybrid]") {
  using namespace hybrid_fixture;
  reset_rng(3);
  auto [environment, coordinates, node] = setup::testing::setup_environment(Code::Oxygen);
  (void)coordinates;
  hybrid_fixture::Stack stack;
  Result result;
  scalar_fixture::Tracking tracking{scalar_fixture::Mode::Absorbed,
                                    environment->getUniverse().get(), result.trace_};
  Model<false> model;
  Counter<false> counter(model);
  Processes<false> process(scalar_fixture::Mode::Absorbed, node, result, counter, false,
                           false);
  Output output{result.trace_};
  cp::HybridCascade cascade(*environment, tracking, process, output, stack);
  cascade.forceInteraction();
  CHECK_THROWS_AS(cascade.forceDecay(), std::runtime_error);
  cp::HybridCascade other(*environment, tracking, process, output, stack);
  other.forceDecay();
  CHECK_THROWS_AS(other.forceInteraction(), std::runtime_error);
}
