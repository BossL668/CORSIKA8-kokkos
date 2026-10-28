/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/CpuParticleScheduler.hpp>
#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <random>
#include <tuple>
#ifdef CORSIKA_PRODUCTION_SCHEDULING_REFERENCE
#include "ProductionScheduler.hpp"
#endif

using namespace corsika;
namespace cp = corsika::kokkos;
namespace {
  using stack_type = test::TransportStack<DummyEnvironment>::type;
  using scheduler_type = cp::CpuParticleScheduler<stack_type>;
  template <typename TStack>
  auto add_particle(TStack& stack, CoordinateSystemPtr const& cs,
                    Code code = Code::Proton) {
    return stack.addParticle(std::make_tuple(
        code, 10_GeV, DirectionVector(cs, {0., 0., 1.}), Point(cs, 0_m, 0_m, 0_m), 0_s));
  }

  using trace_type = std::vector<std::array<std::uint64_t, 6>>;
  template <typename TScheduler>
  trace_type trace(unsigned seed) {
    auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
    stack_type stack;
    TScheduler scheduler(stack);
    std::mt19937 random(seed);
    std::vector<transport::HistoryId> suspended;
    trace_type records;
    for (int j = 0; j < 6; ++j) add_particle(stack, *cs).setNode(node);
    for (unsigned j = 0; j < 240; ++j) {
      if (j % 7 == 0) add_particle(stack, *cs, Code::Photon).setNode(node);
      if (!suspended.empty() && (scheduler.empty() || random() % 4 == 0)) {
        auto n = random() % suspended.size();
        scheduler.resumeParticle(suspended[n]);
        suspended.erase(suspended.begin() + n);
      }
      if (scheduler.empty()) continue;
      auto selected = scheduler.acquireNext();
      auto p = selected.particle_;
      REQUIRE(p.getNode() == node);
      records.push_back({selected.historyId_, selected.parentHistoryId_,
                         selected.generation_, selected.stepId_, stack.getSize(),
                         static_cast<std::uint64_t>(p.getPID())});
      auto choice = random() % 5;
      if (choice == 0) {
        suspended.push_back(selected.historyId_);
        scheduler.completeParticleStep(true);
      } else if (choice == 1) {
        // This is the real router import/export boundary, with no re-search.
        auto state = cp::particle_transfer::to_device_state(
            p, *cs, selected.historyId_, selected.parentHistoryId_, selected.generation_,
            selected.stepId_, tree);
        p.erase();
        scheduler.completeParticleStep();
        cp::particle_transfer::import_particle(stack, state, *cs, tree);
      } else if (choice == 2) {
        // Native secondary allocation must preserve ancestry and LIFO order.
        typename stack_type::stack_view_type view(p);
        view.addSecondary(std::make_tuple(Code::Electron, 2_GeV, p.getDirection()));
        view.addSecondary(std::make_tuple(Code::Photon, 1_GeV, p.getDirection()));
        p.erase();
        scheduler.completeParticleStep();
      } else {
        if (choice == 3) p.erase();
        scheduler.completeParticleStep();
      }
    }
    for (auto id : suspended) scheduler.resumeParticle(id);
    while (!scheduler.empty()) {
      auto p = scheduler.acquireNext();
      records.push_back({p.historyId_, p.parentHistoryId_, p.generation_, p.stepId_,
                         stack.getSize(),
                         static_cast<std::uint64_t>(p.particle_.getPID())});
      p.particle_.erase();
      scheduler.completeParticleStep();
    }
    CHECK(stack.getEntries() == 0);
    CHECK_FALSE(scheduler.hasSuspendedParticles());
    auto s = scheduler.statistics();
    CHECK(s.acquiredParticleSteps_ == records.size());
    CHECK(s.completedParticleSteps_ == records.size());
    CHECK(s.suspendedParticles_ == s.resumedParticles_);
    CHECK(s.maxWavefrontSize_ == 1);
    records.push_back({s.acquiredParticleSteps_, s.completedParticleSteps_,
                       s.maxWavefrontSize_, s.suspendedParticles_, s.resumedParticles_,
                       s.maximumSuspendedParticles_});
    return records;
  }
#ifdef CORSIKA_PRODUCTION_SCHEDULING_REFERENCE
  class ProductionScheduler
      : public production_scheduling_reference::CpuOnlyWavefrontScheduler<stack_type> {
    using base_type =
        production_scheduling_reference::CpuOnlyWavefrontScheduler<stack_type>;

  public:
    using base_type::base_type;
    scheduler_type::ScheduledParticle acquireNext() {
      auto p = base_type::acquireNext();
      return {p.particle, p.history_id, p.parent_history_id, p.generation, p.step_id};
    }
    cp::CpuParticleSchedulerStatistics statistics() const {
      auto const& s = base_type::statistics();
      return {s.acquired_particle_steps, s.completed_particle_steps,
              s.max_wavefront_size,      s.suspended_particles,
              s.resumed_particles,       s.maximum_suspended_particles};
    }
  };
#endif
} // namespace

TEST_CASE("Host scheduling preserves LIFO, skipped holes and suspended identities",
          "[host_scheduler]") {
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  stack_type stack;
  scheduler_type scheduler(stack);
  REQUIRE(scheduler.empty());
  CHECK_THROWS_AS(scheduler.acquireNext(), std::logic_error);
  CHECK_THROWS_AS(scheduler.completeParticleStep(), std::logic_error);
  CHECK_THROWS_AS(scheduler.resumeParticle(42), std::logic_error);
  for (int j = 0; j < 5; ++j) add_particle(stack, *cs).setNode(node);
  stack.at(2).erase();
  auto p = scheduler.acquireNext();
  CHECK(p.historyId_ == 5);
  CHECK(p.stepId_ == 0);
  CHECK_THROWS_AS(scheduler.acquireNext(), std::logic_error);
  scheduler.completeParticleStep(true);
  CHECK(scheduler.suspendedParticleCount() == 1);
  p = scheduler.acquireNext();
  CHECK(p.historyId_ == 4);
  p.particle_.erase();
  scheduler.completeParticleStep();
  p = scheduler.acquireNext();
  CHECK(p.historyId_ == 2);
  CHECK_THROWS_AS(scheduler.resumeParticle(5), std::logic_error);
  scheduler.completeParticleStep(true);
  p = scheduler.acquireNext();
  CHECK(p.historyId_ == 1);
  scheduler.completeParticleStep(true);
  CHECK(scheduler.empty());
  CHECK(stack.getEntries() == 3);
  CHECK_THROWS_AS(scheduler.acquireNext(), std::logic_error);
  scheduler.resumeParticle(5);
  CHECK_THROWS_AS(scheduler.resumeParticle(5), std::logic_error);
  p = scheduler.acquireNext();
  CHECK(p.historyId_ == 5);
  CHECK(p.stepId_ == 1);
  p.particle_.erase();
  scheduler.completeParticleStep();
  for (auto id : {1U, 2U}) scheduler.resumeParticle(id);
  for (auto id : {2U, 1U}) {
    p = scheduler.acquireNext();
    CHECK(p.historyId_ == id);
    CHECK(p.stepId_ == 1);
    p.particle_.erase();
    scheduler.completeParticleStep();
  }
  CHECK(scheduler.empty());
  CHECK(stack.getEntries() == 0);
  CHECK(stack.getSize() == stack.getErased());
  CHECK(scheduler.statistics().acquiredParticleSteps_ == 7);
  CHECK(scheduler.statistics().maximumSuspendedParticles_ == 3);
}

TEST_CASE("Scheduler transfer and secondary traces preserve production order",
          "[host_scheduler]") {
  for (unsigned seed = 0; seed < 24; ++seed) {
    CAPTURE(seed);
    auto current = trace<scheduler_type>(seed);
    CHECK(current.size() > 200);
#ifdef CORSIKA_PRODUCTION_SCHEDULING_REFERENCE
    CHECK(current == trace<ProductionScheduler>(seed));
#endif
  }
}
