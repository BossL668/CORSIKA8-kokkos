/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/scheduling/FlukaInteractionDeferral.hpp>
#include <corsika/kokkos/scheduling/HadronicWorkQueue.hpp>
#include <corsika/kokkos/scheduling/CpuParticleScheduler.hpp>
#include "ScalarStepFixture.hpp"
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <numeric>

namespace counter_commit_fixture {
  using namespace corsika;
  struct Model {
    unsigned synchronousCalls_ = 0;
    template <typename TView>
    void doInteraction(TView& view, Code, Code, FourMomentum const& projectile,
                       FourMomentum const&) {
      ++synchronousCalls_;
      auto const cs = projectile.getSpaceLikeComponents().getCoordinateSystem();
      view.addSecondary(
          std::make_tuple(Code::Photon, 2_GeV, DirectionVector(cs, {0., 0., 1.})));
    }
    CrossSectionType getCrossSection(Code, Code, FourMomentum const&,
                                     FourMomentum const&) const {
      return 1.e12_mb;
    }
  };
} // namespace counter_commit_fixture

namespace counter_commit_fixture {
  using Counter = InteractionCounter<Model, kokkos::FlukaInteractionDeferral>;
  struct Processes : scalar_fixture::Processes {
    Counter& counter_;
    Processes(scalar_fixture::Mode mode, scalar_fixture::Node const* node,
              scalar_fixture::Trace& trace, Counter& counter)
        : scalar_fixture::Processes{mode, node, trace}
        , counter_(counter) {}
    template <typename View, typename Rng>
    ProcessReturn selectInteraction(View& view, FourMomentum const& p4,
                                    NuclearComposition const&, Rng&, CrossSectionType) {
      auto const cs = p4.getSpaceLikeComponents().getCoordinateSystem();
      FourMomentum target(get_mass(Code::Oxygen),
                          MomentumVector(cs, {0_GeV, 0_GeV, 0_GeV}));
      counter_.doInteraction(view, view.getProjectile().getPID(), Code::Oxygen, p4,
                             target);
      return ProcessReturn::Interacted;
    }
  };
} // namespace counter_commit_fixture

TEST_CASE("Counter capture passes through the scheduler queue and ordered stack commit",
          "[interaction_deferral][scalar_stepper]") {
  using namespace corsika;
  namespace cp = corsika::kokkos;
  using Stack = scalar_fixture::Stack;
  for (bool force : {false, true}) {
    auto& manager = RNGManager<>::getInstance();
    if (!manager.isRegistered("cascade")) manager.registerRandomStream("cascade");
    manager.setSeed(289);
    manager.getRandomStream("cascade").reset();
    auto [environment, coordinates, node] =
        setup::testing::setup_environment(Code::Oxygen);
    Stack stack;
    for (auto energy : {10_GeV, 40_GeV}) {
      auto particle = stack.addParticle(std::make_tuple(
          Code::Proton, energy, DirectionVector(*coordinates, {0., 0., 1.}),
          Point(*coordinates, 0_m, 0_m, 0_m), 0_s));
      particle.setWeight(3.);
    }
    scalar_fixture::Trace trace;
    auto const mode = force ? scalar_fixture::Mode::ForcedInteraction
                            : scalar_fixture::Mode::Interaction;
    scalar_fixture::Tracking tracking{mode, environment->getUniverse().get(), trace};
    counter_commit_fixture::Model model;
    counter_commit_fixture::Counter counter(model);
    counter_commit_fixture::Processes processes(mode, node, trace, counter);
    cp::ScalarCascadeStepper<decltype(tracking), decltype(processes), Stack> stepper(
        *environment, tracking, processes, stack);
    cp::CpuParticleScheduler<Stack> scheduler(stack);
    stepper.setNodes();
    using Particle = Stack::particle_type;
    struct Pending {
      std::uint64_t sequenceId_;
      transport::HistoryId historyId_;
      Particle particle_;
      cp::HadronicInteractionRequest request_;
    };
    cp::ClassifiedHadronicWorkQueue<Pending> queue;
    std::uint64_t sequence = 0;
    while (!scheduler.empty()) {
      auto scheduled = scheduler.acquireNext();
      if (force) stepper.forceInteraction();
      cp::HadronicInteractionDeferralContext context(
          {91, 12, scheduled.historyId_, scheduled.stepId_, 1U, 0U}, sequence);
      {
        cp::ScopedHadronicInteractionDeferral scope(context);
        CHECK(stepper.advance(scheduled.particle_) ==
              cp::ScalarCascadeAdvanceResult::DeferredHadronicInteraction);
      }
      REQUIRE(context.hasPrepared());
      auto request = context.takePrepared().request_;
      auto key = cp::classify_hadronic_work(scheduled.particle_.getPID(),
                                            scheduled.particle_.getKineticEnergy(), {});
      REQUIRE(key);
      queue.enqueue(
          *key, sequence, cp::deterministic_hadronic_interaction_cost(*key, .1),
          Pending{sequence, scheduled.historyId_, scheduled.particle_, request});
      scheduler.completeParticleStep(true);
      ++sequence;
    }
    CHECK(queue.size() == 2);
    CHECK(scheduler.suspendedParticleCount() == 2);
    CHECK(model.synchronousCalls_ == 0);
    CHECK(counter.getCount() == 2);
    CHECK(counter.getEnergyLedgerStatistics().deferredInteractions_ == 2);
    CHECK(std::count(trace.calls_.begin(), trace.calls_.end(), "secondaries") == 0);
    auto assignments = cp::plan_hadronic_worker_assignments(queue, 2, .1, 1);
    auto batches = cp::coalesce_hadronic_worker_assignments(assignments);
    CHECK(queue.empty());
    std::vector<Pending> pending;
    for (auto& batch : batches)
      for (auto& item : batch.payloads_) pending.push_back(std::move(item));
    std::sort(pending.begin(), pending.end(), [](auto const& left, auto const& right) {
      return left.sequenceId_ < right.sequenceId_;
    });
    // Controlled responses exercise commit ordering, not a FLUKA physics claim.
    for (auto& item : pending) {
      CHECK(item.request_.sequenceId_ == item.sequenceId_);
      CHECK(item.request_.randomKey_.historyId_ == item.historyId_);
      CHECK_FALSE(item.particle_.isErased());
      std::vector<cp::HadronicSecondaryRecord> secondaries{{22, 0, 2., {0., 0., 1.}},
                                                           {11, 0, 3., {0., 0., 1.}}};
      stepper.commitDeferredHadronicInteraction(item.particle_, secondaries);
      scheduler.resumeParticle(item.historyId_);
      CHECK(item.particle_.isErased());
    }
    CHECK_FALSE(scheduler.hasSuspendedParticles());
    CHECK(std::count(trace.calls_.begin(), trace.calls_.end(), "secondaries") == 2);
    std::vector<transport::HistoryId> parents;
    while (!scheduler.empty()) {
      auto child = scheduler.acquireNext();
      parents.push_back(child.parentHistoryId_);
      CHECK(child.particle_.getWeight() == 2.);
      CHECK(child.particle_.getNode() == node);
      CHECK(child.generation_ == 1);
      child.particle_.erase();
      scheduler.completeParticleStep();
    }
    REQUIRE(parents.size() == 4);
    for (auto const& item : pending)
      CHECK(std::count(parents.begin(), parents.end(), item.historyId_) == 2);
    CHECK(model.synchronousCalls_ == 0);
    CHECK(counter.getCount() == 2);
  }
}
