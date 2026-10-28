/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#ifdef CORSIKA_PRODUCTION_IDENTITY_REFERENCE
#include "ProductionIdentity.hpp"
#endif

using namespace corsika;

TEST_CASE("Transport identities follow native stack compaction and ancestry",
          "[transport_identity]") {
  using stack_type = test::TransportStack<DummyEnvironment>::type;
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  stack_type stack;
  auto add = [&] {
    return stack.addParticle(std::make_tuple(Code::Electron, 1_GeV,
                                             DirectionVector(*cs, {0., 0., 1.}),
                                             Point(*cs, 0_m, 0_m, 0_m), 0_s));
  };
  auto p = add();
  p.setNode(node);
  CHECK(p.getHistoryId() == 1);
  CHECK(p.getGeneration() == 0);
  CHECK(p.getStepId() == 0);
  CHECK(p.beginTransportStep() == 0);
  CHECK(p.beginTransportStep() == 1);
  typename stack_type::stack_view_type view(p);
  auto child = view.addSecondary(std::make_tuple(Code::Photon, .1_GeV, p.getDirection()));
  CHECK(child.getHistoryId() == 2);
  CHECK(child.getParentHistoryId() == 1);
  CHECK(child.getGeneration() == 1);
  CHECK(child.getStepId() == 0);
  auto q = add();
  CHECK(q.getHistoryId() == 3);
  stack.swap(stack.begin(), stack.begin() + 2);
  CHECK(stack.begin().getHistoryId() == 3);
  CHECK((stack.begin() + 2).getHistoryId() == 1);
  (stack.begin() + 1).erase();
  CHECK(stack.getEntries() == 2);
  // Native erase marks a slot; compaction is an explicit stack operation.
  stack.swap(stack.begin() + 1, stack.begin() + 2);
  REQUIRE(stack.purgeLastIfDeleted());
  CHECK((stack.begin() + 1).getHistoryId() == 1);
  stack.copy(stack.begin() + 1, stack.begin());
  CHECK(stack.begin().getStepId() == 2);
  stack.clear();
  CHECK(add().getHistoryId() == 1);
}

TEST_CASE("Transport identity ranges and overflow preserve production allocation",
          "[transport_identity]") {
  transport::TransportIdentityData data;
  data.incrementSize();
  data.incrementSize();
  CHECK(data.reserveHistoryIds(20) == 1);
  data.assignPrimary(0);
  CHECK(data.getTransportIdentity(0).historyId_ == 21);
  data.assignSecondary(1, data.getTransportIdentity(0));
  CHECK(data.getTransportIdentity(1).parentHistoryId_ == 21);
  CHECK_THROWS_AS(data.reserveHistoryIds(0), std::invalid_argument);
  CHECK_THROWS_AS(data.importIdentity(0, {0, 0, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(data.importIdentity(0, {1, 2, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(data.assignSecondary(
                      0, {1, 0, std::numeric_limits<transport::Generation>::max(), 0}),
                  std::overflow_error);
  auto const maximum = std::numeric_limits<transport::HistoryId>::max();
  data.importIdentity(0, {maximum - 1, 0, 0, 0});
  CHECK(data.reserveHistoryIds(1) == maximum);
  CHECK_THROWS_AS(data.reserveHistoryIds(1), std::overflow_error);
  data.clear();
  data.incrementSize();
  data.assignPrimary(0);
  CHECK(data.getTransportIdentity(0).historyId_ == 1);
#ifdef CORSIKA_PRODUCTION_IDENTITY_REFERENCE
  production_transport::TransportIdentityData original;
  data.clear();
  for (int i = 0; i < 128; ++i) {
    data.incrementSize();
    original.incrementSize();
    if (i % 3) {
      data.assignSecondary(i, data.getTransportIdentity(0));
      original.assignSecondary(i, original.getTransportIdentity(0));
    } else {
      data.assignPrimary(i);
      original.assignPrimary(i);
    }
    data.setStepId(i, i * 19);
    original.setStepId(i, i * 19);
  }
  for (int i = 0; i < 128; ++i) {
    data.swap(i, 127 - i);
    original.swap(i, 127 - i);
  }
  data.copy(2, 5);
  original.copy(2, 5);
  for (int i = 0; i < 128; ++i) {
    auto a = data.getTransportIdentity(i);
    auto b = original.getTransportIdentity(i);
    CHECK(a.historyId_ == b.history_id);
    CHECK(a.parentHistoryId_ == b.parent_history_id);
    CHECK(a.generation_ == b.generation);
    CHECK(a.stepId_ == b.step_id);
  }
#endif
}
