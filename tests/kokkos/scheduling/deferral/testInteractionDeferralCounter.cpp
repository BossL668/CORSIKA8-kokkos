/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "CounterFixture.hpp"
#include "SharedCounterModel.hpp"
#include "compareCounter.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <numeric>
#include <type_traits>

using namespace corsika;
using namespace counter_fixture;
namespace cp = corsika::kokkos;

namespace {
  template <typename T, typename = void>
  struct CanMakeFlukaCounter : std::false_type {};
  template <typename T>
  struct CanMakeFlukaCounter<
      T, std::void_t<decltype(cp::make_fluka_interaction_counter(std::declval<T&>()))>>
      : std::true_type {};
  static_assert(CanMakeFlukaCounter<corsika::fluka::InteractionModel>::value);
  static_assert(CanMakeFlukaCounter<corsika::fluka::Interaction>::value);
  static_assert(!CanMakeFlukaCounter<Process>::value);
  static_assert(!CanMakeFlukaCounter<WorkerProcess>::value);
} // namespace

TEST_CASE("Same model with two explicit policies coexists across translation units",
          "[interaction_deferral][headers]") {
  Inputs input;
  for (bool nativeFirst : {false, true}) {
    SharedModel model;
    cp::HadronicInteractionDeferralContext context({}, 80);
    cp::ScopedHadronicInteractionDeferral scope(context);
    auto native = [&] {
      auto result = native_counter_other_tu(model, input.projectileP4_, input.targetP4_);
      CHECK(result.count_ == 1);
      CHECK(result.deferred_ == 0);
      CHECK(result.histogram_ == 1);
      CHECK(model.calls_ == 1);
    };
    auto deferred = [&] {
      auto result =
          deferred_counter_other_tu(model, input.projectileP4_, input.targetP4_);
      CHECK(result.count_ == 1);
      CHECK(result.deferred_ == 1);
      CHECK(result.histogram_ == 1);
      CHECK(context.hasPrepared());
    };
    if (nativeFirst) {
      native();
      CHECK_FALSE(context.hasPrepared());
      deferred();
    } else {
      deferred();
      CHECK(model.calls_ == 0);
      native();
    }
    CHECK(model.calls_ == 1);
    CHECK(context.takePrepared().request_.sequenceId_ == 80);
  }
}

TEST_CASE("Native counters keep production accounting without a deferral scope",
          "[interaction_deferral]") {
  Inputs input;
  for (bool fail : {false, true}) {
    for (bool isotope : {false, true}) {
      for (double energy : {1., 120., 400.}) {
        CAPTURE(fail, isotope, energy);
        WorkerProcess currentProcess, oldProcess;
        currentProcess.fail_ = oldProcess.fail_ = fail;
        auto cs = get_root_CoordinateSystem();
        currentProcess.output_ =
            oldProcess.output_ = {{Code::Proton, energy * 1_GeV, 2.,
                                   MomentumVector(cs, {0_GeV, 0_GeV, 98_GeV})},
                                  {isotope ? Code::Oxygen : Code::Photon, 1_GeV, 2.,
                                   MomentumVector(cs, {0_GeV, 0_GeV, 1_GeV})}};
        InteractionCounter<WorkerProcess, cp::FlukaInteractionDeferral> current(
            currentProcess);
        reference::InteractionCounter old(oldProcess);
        static_assert(
            !std::is_same_v<decltype(current), InteractionCounter<WorkerProcess>>);
        View currentView, oldView;
        CHECK(current.getCrossSection(input.projectile_, input.target_,
                                      input.projectileP4_, input.targetP4_) == 73_mb);
        if (fail) {
          CHECK_THROWS_AS(input.invoke(current, currentView), std::runtime_error);
          CHECK_THROWS_AS(input.invoke(old, oldView), std::runtime_error);
          CHECK(current.getCount() == 0);
        } else {
          input.invoke(current, currentView);
          input.invoke(old, oldView);
          CHECK(current.getCount() == 1);
          CHECK(current.getEnergyLedgerStatistics().auditedInteractions_ == 1);
        }
        CHECK(currentProcess.calls_ == 1);
        CHECK(currentProcess.calls_ == oldProcess.calls_);
        compare_counters(current, old);
        auto const count = current.getCount();
        current.releaseTimingSamples();
        old.releaseTimingSamples();
        CHECK(current.getTimingSamples().empty());
        CHECK(current.getCount() == count);
        compare_counters(current, old);
      }
    }
  }
}

TEST_CASE("A selected interaction is counted before capture and not generated twice",
          "[interaction_deferral]") {
  Inputs input;
  WorkerProcess currentProcess, oldProcess;
  InteractionCounter<WorkerProcess, cp::FlukaInteractionDeferral> current(currentProcess);
  reference::InteractionCounter old(oldProcess);
  View currentView, oldView;
  cp::HadronicInteractionDeferralContext context({1, 2, 3, 4, 5, 6}, 771);
  production::HadronicInteractionDeferralContext oldContext({1, 2, 3, 4, 5, 6}, 771);
  {
    cp::ScopedHadronicInteractionDeferral scope(context);
    input.invoke(current, currentView);
    CHECK_THROWS_AS(input.invoke(current, currentView), std::logic_error);
  }
  {
    production::ScopedHadronicInteractionDeferral scope(oldContext);
    input.invoke(old, oldView);
    CHECK_THROWS_AS(input.invoke(old, oldView), std::logic_error);
  }
  REQUIRE(context.hasPrepared());
  REQUIRE(oldContext.hasPrepared());
  CHECK(cp::HadronicInteractionDeferralContext::current() == nullptr);
  CHECK(currentProcess.calls_ == 0);
  CHECK(currentView.secondaries_.empty());
  CHECK(current.getCount() == 1);
  CHECK(current.getEnergyLedgerStatistics().deferredInteractions_ == 1);
  CHECK(current.getEnergyLedgerStatistics().auditedInteractions_ == 0);
  CHECK(current.getTimingSamples().front().deferred_);
  CHECK(current.getTimingSamples().front().finalStateTimeMs_ == 0.);
  auto const& histogram = current.getHistogram().labHist();
  CHECK(std::accumulate(histogram.begin(), histogram.end(), 0) == 2);
  // Production fills the histogram before a duplicate-capture exception. Do not
  // silently add rollback semantics while migrating this execution adapter.
  compare_counters(current, old);
  auto const request = context.takePrepared().request_;
  auto const oldRequest = oldContext.takePrepared().request;
  static_assert(sizeof(request) == sizeof(oldRequest));
  CHECK(std::memcmp(&request, &oldRequest, sizeof(request)) == 0);
}

TEST_CASE("An unsupported model remains synchronous inside a worker scope",
          "[interaction_deferral]") {
  Inputs input;
  Process currentProcess, oldProcess;
  InteractionCounter current(currentProcess);
  reference::InteractionCounter old(oldProcess);
  View currentView, oldView;
  cp::HadronicInteractionDeferralContext context({}, 11);
  production::HadronicInteractionDeferralContext oldContext({}, 11);
  {
    cp::ScopedHadronicInteractionDeferral scope(context);
    input.invoke(current, currentView);
  }
  {
    production::ScopedHadronicInteractionDeferral scope(oldContext);
    input.invoke(old, oldView);
  }
  CHECK_FALSE(context.hasPrepared());
  CHECK_FALSE(oldContext.hasPrepared());
  CHECK(currentProcess.calls_ == 1);
  compare_counters(current, old);
}

TEST_CASE("The optional FLUKA policy links in either header order and restores scopes",
          "[interaction_deferral][headers]") {
  Inputs input;
  for (auto call : {header_counter_first, header_adapter_first}) {
    CHECK_FALSE(
        call(input.projectile_, input.target_, input.projectileP4_, input.targetP4_));
    cp::HadronicInteractionDeferralContext outer({}, 10), inner({}, 11);
    {
      cp::ScopedHadronicInteractionDeferral outerScope(outer);
      CHECK_THROWS_AS(([&] {
                        cp::ScopedHadronicInteractionDeferral innerScope(inner);
                        CHECK(call(input.projectile_, input.target_, input.projectileP4_,
                                   input.targetP4_));
                        throw std::runtime_error("controlled scope failure");
                      })(),
                      std::runtime_error);
      CHECK(cp::HadronicInteractionDeferralContext::current() == &outer);
      CHECK(call(input.projectile_, input.target_, input.projectileP4_, input.targetP4_));
    }
    CHECK(cp::HadronicInteractionDeferralContext::current() == nullptr);
    CHECK(outer.takePrepared().request_.sequenceId_ == 10);
    CHECK(inner.takePrepared().request_.model_ == cp::HadronicWorkerModel::Fluka);
  }
}
