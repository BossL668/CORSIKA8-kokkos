/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/ProposalCpuFallbackHandler.hpp>
#include <corsika/modules/ParticleCut.hpp>
#include <corsika/modules/thinning/EMThinning.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
#include "ProductionCpuHandler.hpp"
#endif

using namespace corsika;
namespace cp = corsika::kokkos;
namespace {
  struct Thresholds {
    std::vector<std::pair<Code, HEPEnergyType>> old_;
    Thresholds() {
      for (auto p : get_all_particles())
        old_.emplace_back(p, get_kinetic_energy_propagation_threshold(p));
    }
    ~Thresholds() {
      for (auto p : old_) set_kinetic_energy_propagation_threshold(p.first, p.second);
    }
  };
  struct Model {
    std::vector<std::string> calls_;
    proposal::ProposalInteractionRecord record_;
    std::vector<double> draws_;
    bool lowPhoton_ = true;
    ProcessReturn result_ = ProcessReturn::Ok;
    void prepareSpecifiedInteractionCalculator(proposal::ProposalInteractionRecord const&,
                                               HEPEnergyType) {
      calls_.push_back("cut");
    }
    void completeSelectedLoss(proposal::ProposalInteractionRecord& r, double q) {
      calls_.push_back("loss");
      CHECK(q == .25);
      r.vLoss_ = .3;
    }
    void completeNativeSelectionReplay(proposal::ProposalInteractionRecord& r) {
      calls_.push_back("replay");
      CHECK(r.selectionUniform_ == .4);
      r.vLoss_ = .3;
    }
    std::size_t requiredFinalStateRandomNumbers(
        proposal::ProposalInteractionRecord const&) {
      calls_.push_back("draw-count");
      return 3;
    }
    template <typename TView>
    ProcessReturn doSpecifiedInteraction(TView& view,
                                         proposal::ProposalInteractionRecord const& r,
                                         std::vector<double> draws) {
      calls_.push_back("final-state");
      record_ = r;
      draws_ = draws;
      CHECK(view.getProjectile().getHistoryId() == 42);
      if (result_ != ProcessReturn::Ok) return result_;
      view.addSecondary(
          std::make_tuple(Code::Electron, 2_GeV, view.getProjectile().getDirection()));
      view.addSecondary(std::make_tuple(Code::Photon, lowPhoton_ ? .1_GeV : 1_GeV,
                                        view.getProjectile().getDirection()));
      return result_;
    }
  };
  struct ProcessList {
    Model& model_;
    ParticleCut<> cut_{.2_GeV, .2_GeV, .2_GeV, .2_GeV, .2_GeV, true};
    EMThinning thinning_{100_GeV, 100.};
    std::size_t calls_ = 0, before_ = 0, after_ = 0;
    template <typename TView>
    void doSecondaries(TView& view) {
      ++calls_;
      model_.calls_.push_back("secondaries");
      before_ = view.getSize();
      CHECK(view.getProjectile().getHistoryId() == 42);
      for (auto& p : view) {
        CHECK(p.getParentHistoryId() == 42);
        CHECK(p.getGeneration() == 3);
        if constexpr (TView::has_event)
          CHECK(p.getEvent()->eventType() == history::EventType::Interaction);
      }
      cut_.doSecondaries(view);
      thinning_.doSecondaries(view);
      after_ = view.getEntries();
    }
  };
  void rng() {
    auto& r = RNGManager<>::getInstance();
    r.setSeed(1234);
    for (auto name : {"proposal", "thinning"}) {
      if (!r.isRegistered(name)) r.registerRandomStream(name);
      r.getRandomStream(name).reset();
    }
  }
  cp::FallbackRequest event(int node) {
    cp::FallbackRequest e{};
    e.particle_.pid_ = 11;
    e.particle_.energyGeV_ = 10.;
    e.particle_.direction_[2] = 1.;
    e.particle_.historyId_ = 42;
    e.particle_.parentHistoryId_ = 3;
    e.particle_.generation_ = 2;
    e.particle_.stepId_ = 7;
    e.particle_.environmentNode_ = node;
    e.mediumHash_ = 7654;
    e.interactionHash_ = 123;
    e.componentHash_ = 456;
    e.processId_ = cp::ElectronPairProcessId;
    e.reason_ = cp::FallbackReason::CpuOnlyProcess;
    e.selectionUniform_ = .4;
    e.lossQuantile_ = .25;
    e.energyFraction_ = .3;
    e.interactionVertexReached_ = 1;
    return e;
  }
  template <typename TStack>
  void check_handler() {
    Thresholds thresholds;
    rng();
    auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
    cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
    for (int branch = 0; branch < 3; ++branch) {
      CAPTURE(branch);
      TStack stack;
      Model model;
      ProcessList processes{model};
      cp::ProposalCpuFallbackHandler<TStack, Model, ProcessList, DummyEnvironment>
          handler(model, processes, *env, tree, *cs, 17, 19, .4_MeV);
      auto e = event(tree.getNodeIndex(*(node)));
      if (branch == 1) e.reason_ = cp::FallbackReason::NativeSelectionReplay;
      if (branch == 2) e.reason_ = cp::FallbackReason::InverseCdfUnavailable;
      if (branch == 1) e.lossQuantile_ = std::numeric_limits<double>::quiet_NaN();
      REQUIRE(handler.canHandle(e));
      handler.handle(stack, e);
      std::vector<std::string> expected{"cut"};
      if (branch) expected.push_back(branch == 1 ? "replay" : "loss");
      expected.insert(expected.end(), {"draw-count", "final-state", "secondaries"});
      CHECK(model.calls_ == expected);
      CHECK(model.record_.context_.mediumHash_ ==
            node->getModelProperties().getNuclearComposition().getHash());
      CHECK(model.record_.interactionHash_ == 123);
      CHECK(model.record_.componentHash_ == 456);
      CHECK(model.draws_ == cp::make_proposal_final_state_random_numbers(e, 3, 17, 19));
      CHECK(processes.calls_ == 1);
      CHECK(processes.before_ == 2);
      CHECK(processes.after_ == 1);
      REQUIRE(stack.getEntries() == 1);
      CHECK(stack.begin().getHistoryId() != 42);
      CHECK(stack.begin().getParentHistoryId() == 42);
      CHECK(stack.begin().getNode() == node);
      CHECK(stack.begin().getPID() == Code::Electron);
      CHECK(handler.statistics().specifiedInteractions_ == 1);
      CHECK(handler.statistics().generatedSecondaries_ == 2);
      CHECK(handler.statistics().completedSelectedLosses_ == (branch ? 1 : 0));
      CHECK(handler.statistics().completedNativeSelectionReplays_ ==
            (branch == 1 ? 1 : 0));
    }
  }
} // namespace

TEST_CASE("CPU handler preserves selected identity, postprocessing order and ancestry",
          "[cpu_handler]") {
  check_handler<test::TransportStack<DummyEnvironment>::type>();
  check_handler<test::TransportStack<DummyEnvironment>::history_type>();
}

TEST_CASE("CPU handler rejects invalid reasons and does not silently retry",
          "[cpu_handler]") {
  Thresholds thresholds;
  rng();
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
  using stack_type = test::TransportStack<DummyEnvironment>::type;
  stack_type stack;
  Model model;
  ProcessList processes{model};
  cp::ProposalCpuFallbackHandler<stack_type, Model, ProcessList, DummyEnvironment>
      handler(model, processes, *env, tree, *cs, 17, 19);
  auto e = event(tree.getNodeIndex(*(node)));
  e.reason_ = cp::FallbackReason::InvalidFinalState;
  CHECK_FALSE(handler.canHandle(e));
  CHECK_THROWS_AS(handler.handle(stack, e), std::invalid_argument);
  CHECK(stack.getEntries() == 0);
  CHECK(model.calls_.empty());
  e = event(0);
  CHECK_THROWS_AS(handler.handle(stack, e), std::runtime_error);
  CHECK(stack.getEntries() == 0);
  CHECK(processes.calls_ == 0);
  CHECK(handler.statistics().specifiedInteractions_ == 0);
  model.result_ = ProcessReturn::ParticleAbsorbed;
  e = event(tree.getNodeIndex(*(node)));
  CHECK_THROWS_AS(handler.handle(stack, e), std::runtime_error);
  CHECK(stack.getEntries() == 0);
  CHECK(processes.calls_ == 0);
  CHECK(handler.statistics().specifiedInteractions_ == 0);
}

#ifdef CORSIKA_PRODUCTION_HANDLER_REFERENCE
TEST_CASE("CPU handler matches the pinned production cut and thinning path",
          "[cpu_handler]") {
  Thresholds thresholds;
  rng();
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
  cp::production_reference::environment_tree::HostMap<DummyEnvironmentInterface>
      nativeMap(tree);
  using S = test::TransportStack<DummyEnvironment>::history_type;
  for (unsigned seed = 1; seed <= 32; ++seed) {
    CAPTURE(seed);
    S a, b;
    Model ma, mb;
    ma.lowPhoton_ = mb.lowPhoton_ = false;
    ProcessList pa{ma}, pb{mb};
    cp::ProposalCpuFallbackHandler<S, Model, ProcessList, DummyEnvironment> current(
        ma, pa, *env, tree, *cs, seed, 19);
    cp::production_reference::ProposalCpuFallbackHandler<S, Model, ProcessList,
                                                         DummyEnvironment>
        original(mb, pb, *env, *cs, seed, 19);
    original.setEnvironmentTree(nativeMap.view());
    auto e = event(tree.getNodeIndex(*(node)));
    auto& rng = RNGManager<>::getInstance();
    rng.setSeed(seed);
    rng.getRandomStream("thinning").reset();
    auto initial = rng.getRandomStream("thinning");
    original.handle(b, e);
    auto after = rng.dumpState().str();
    rng.getRandomStream("thinning") = initial;
    current.handle(a, e);
    CHECK(after == rng.dumpState().str());
    CHECK(ma.calls_ == mb.calls_);
    CHECK(ma.draws_ == mb.draws_);
    REQUIRE(a.getEntries() == b.getEntries());
    auto j = b.begin();
    for (auto i = a.begin(); i != a.end(); ++i, ++j) {
      CHECK(i.getPID() == j.getPID());
      CHECK(i.getEnergy() == j.getEnergy());
      CHECK(i.getWeight() == j.getWeight());
      CHECK(i.getHistoryId() == j.getHistoryId());
      CHECK(i.getParentHistoryId() == j.getParentHistoryId());
      CHECK(i.getNode() == j.getNode());
    }
    CHECK(current.statistics().generatedSecondaries_ ==
          original.statistics().generatedSecondaries_);
  }
}
#endif
