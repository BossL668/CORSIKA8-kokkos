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
#include <corsika/kokkos/scheduling/CpuParticleScheduler.hpp>
#include <cstring>
#include <type_traits>
using namespace corsika;
namespace cp = corsika::kokkos;
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
namespace reference = corsika::production_stepper_reference;
#endif
namespace {
  template <typename Context, auto Model>
  struct DeferredProcesses : scalar_fixture::Processes {
    using scalar_fixture::Processes::Processes;
    DeferredProcesses(scalar_fixture::Mode mode, scalar_fixture::Node const* node,
                      scalar_fixture::Trace& trace)
        : scalar_fixture::Processes{mode, node, trace} {}
    template <typename View, typename Rng>
    ProcessReturn selectInteraction(View& view, FourMomentum const& p4,
                                    NuclearComposition const&, Rng&,
                                    CrossSectionType sample) {
      trace_.calls_.push_back("deferred-interaction");
      trace_.arguments_.push_back(sample / 1_mb);
      auto context = Context::current();
      REQUIRE(context != nullptr);
      FourMomentum target{
          get_mass(Code::Oxygen),
          MomentumVector(p4.getSpaceLikeComponents().getCoordinateSystem(),
                         {0_GeV, 0_GeV, 0_GeV})};
      context->tryPrepare(Model, view.getProjectile().getPID(), Code::Oxygen, p4, target);
      return ProcessReturn::Interacted;
    }
  };
  struct Result {
    scalar_fixture::Trace trace_;
    cp::HadronicInteractionRequest request_;
  };
  template <template <typename, typename, typename> typename Stepper, typename Context,
            typename Scope, typename Key, typename Record, auto Model,
            typename TStack = scalar_fixture::Stack>
  Result exercise(bool forced) {
    auto& manager = RNGManager<>::getInstance();
    if (!manager.isRegistered("cascade")) manager.registerRandomStream("cascade");
    manager.setSeed(289);
    manager.getRandomStream("cascade").reset();
    auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    TStack stack;
    Result result{};
    auto p = stack.addParticle(std::make_tuple(Code::Proton, 100_GeV,
                                               DirectionVector(*cs, {0., 0., 1.}),
                                               Point(*cs, 0_m, 0_m, 0_m), 0_s));
    p.setWeight(3.);
    cp::CpuParticleScheduler<TStack> scheduler(stack);
    auto selected = scheduler.acquireNext();
    auto mode = forced ? scalar_fixture::Mode::ForcedInteraction
                       : scalar_fixture::Mode::Interaction;
    scalar_fixture::Tracking tracking{mode, env->getUniverse().get(), result.trace_};
    DeferredProcesses<Context, Model> process(mode, node, result.trace_);
    Stepper<decltype(tracking), decltype(process), TStack> stepper(*env, tracking,
                                                                   process, stack);
    stepper.setNodes();
    if (forced) stepper.forceInteraction();
    Context context(Key{1, 2, 3, 4, 5, 6}, 789);
    {
      Scope scoped(context);
      auto advance = stepper.advance(selected.particle_);
      CHECK(static_cast<int>(advance) == 1);
    }
    REQUIRE(context.hasPrepared());
    CHECK_FALSE(selected.particle_.isErased());
    CHECK(stack.getEntries() == 1);
    CHECK(std::count(result.trace_.calls_.begin(), result.trace_.calls_.end(),
                     "secondaries") == 0);
    auto record = context.takePrepared();
    // Both wrapper records contain only the unchanged production wire request.
    static_assert(sizeof(record) == sizeof(result.request_));
    static_assert(std::is_trivially_copyable_v<decltype(record)>);
    static_assert(std::is_trivially_copyable_v<decltype(result.request_)>);
    std::memcpy(static_cast<void*>(&result.request_), &record, sizeof(record));
    scheduler.completeParticleStep(true);
    CHECK(scheduler.hasSuspendedParticles());
    CHECK(scheduler.empty());
    std::vector<Record> finalState{{22, 0, 2., {0., 0., 1.}}, {11, 0, 3., {0., 0., 1.}}};
    stepper.commitDeferredHadronicInteraction(selected.particle_, finalState);
    CHECK(selected.particle_.isErased());
    CHECK(std::count(result.trace_.calls_.begin(), result.trace_.calls_.end(),
                     "secondaries") == 1);
    CHECK_THROWS_AS(
        stepper.commitDeferredHadronicInteraction(selected.particle_, finalState),
        std::logic_error);
    scheduler.resumeParticle(selected.historyId_);
    CHECK_FALSE(scheduler.hasSuspendedParticles());
    CHECK_FALSE(scheduler.empty());
    scalar_fixture::capture(stack, node, result.trace_);
    result.trace_.rng_ = manager.dumpState().str();
    std::vector<Code> children;
    while (!scheduler.empty()) {
      auto child = scheduler.acquireNext();
      CHECK(child.parentHistoryId_ == selected.historyId_);
      CHECK(child.generation_ == selected.generation_ + 1);
      CHECK(child.particle_.getNode() == node);
      CHECK(child.particle_.getWeight() == 2.);
      if constexpr (TStack::stack_view_type::has_event) {
        REQUIRE(child.particle_.getEvent());
        CHECK(child.particle_.getEvent()->eventType() == history::EventType::Interaction);
      }
      children.push_back(child.particle_.getPID());
      child.particle_.erase();
      scheduler.completeParticleStep();
    }
    CHECK(children == std::vector<Code>{Code::Electron, Code::Photon});
    return result;
  }
} // namespace
TEST_CASE("Deferred scalar interactions return once to the native stack",
          "[scalar_stepper]") {
  for (bool forced : {false, true}) {
    CAPTURE(forced);
    auto current =
        exercise<cp::ScalarCascadeStepper, cp::HadronicInteractionDeferralContext,
                 cp::ScopedHadronicInteractionDeferral, cp::HadronicRandomKey,
                 cp::HadronicSecondaryRecord, cp::HadronicWorkerModel::Sibyll>(forced);
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
    auto old = exercise<
        reference::ScalarCascadeStepper, reference::HadronicInteractionDeferralContext,
        reference::ScopedHadronicInteractionDeferral, reference::HadronicRandomKey,
        reference::HadronicSecondaryRecord, reference::HadronicWorkerModel::Sibyll>(
        forced);
    CHECK(current.trace_.calls_ == old.trace_.calls_);
    CHECK(current.trace_.arguments_ == old.trace_.arguments_);
    CHECK(current.trace_.particles_ == old.trace_.particles_);
    CHECK(current.trace_.rng_ == old.trace_.rng_);
    CHECK(current.request_.sequenceId_ == old.request_.sequenceId_);
    CHECK(current.request_.projectileFourMomentumGeV_ ==
          old.request_.projectileFourMomentumGeV_);
    CHECK(current.request_.targetFourMomentumGeV_ == old.request_.targetFourMomentumGeV_);
    CHECK(cp::derive_hadronic_worker_seed(current.request_.randomKey_) ==
          cp::derive_hadronic_worker_seed(old.request_.randomKey_));
#endif
  }
}

TEST_CASE("Deferred commits retain native history events", "[scalar_stepper]") {
  using HistoryStack = test::TransportStack<DummyEnvironment>::history_type;
  for (bool forced : {false, true}) {
    auto current =
        exercise<cp::ScalarCascadeStepper, cp::HadronicInteractionDeferralContext,
                 cp::ScopedHadronicInteractionDeferral, cp::HadronicRandomKey,
                 cp::HadronicSecondaryRecord, cp::HadronicWorkerModel::Sibyll,
                 HistoryStack>(forced);
#ifdef CORSIKA_PRODUCTION_STEPPER_REFERENCE
    auto old = exercise<reference::ScalarCascadeStepper,
                        reference::HadronicInteractionDeferralContext,
                        reference::ScopedHadronicInteractionDeferral,
                        reference::HadronicRandomKey, reference::HadronicSecondaryRecord,
                        reference::HadronicWorkerModel::Sibyll, HistoryStack>(forced);
    CHECK(current.trace_.calls_ == old.trace_.calls_);
    CHECK(current.trace_.arguments_ == old.trace_.arguments_);
    CHECK(current.trace_.particles_ == old.trace_.particles_);
    CHECK(current.trace_.events_ == old.trace_.events_);
    CHECK(current.trace_.rng_ == old.trace_.rng_);
#endif
  }
}
