/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/CpuParticleScheduler.hpp>
#include <corsika/kokkos/em/proposal/ProposalCpuFallbackHandler.hpp>
#include <tests/common/TransportTestStack.hpp>
#include <tests/common/SetupTestEnvironment.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace corsika;
namespace cp = corsika::kokkos;
namespace {
  struct Model {
    void prepareSpecifiedInteractionCalculator(proposal::ProposalInteractionRecord const&,
                                               HEPEnergyType) {}
    void completeSelectedLoss(proposal::ProposalInteractionRecord&, double) {}
    void completeNativeSelectionReplay(proposal::ProposalInteractionRecord&) {}
    std::size_t requiredFinalStateRandomNumbers(
        proposal::ProposalInteractionRecord const&) {
      return 0;
    }
    template <typename TView>
    ProcessReturn doSpecifiedInteraction(TView& view,
                                         proposal::ProposalInteractionRecord const&,
                                         std::vector<double>) {
      view.addSecondary(
          std::make_tuple(Code::Electron, 7_GeV, view.getProjectile().getDirection()));
      view.addSecondary(
          std::make_tuple(Code::Photon, 3_GeV, view.getProjectile().getDirection()));
      return ProcessReturn::Ok;
    }
  };
  struct Processes {
    unsigned calls_ = 0;
    template <typename TView>
    void doSecondaries(TView&) {
      ++calls_;
    }
  };
} // namespace

TEST_CASE(
    "Specified CPU completion returns all children to the native scheduler exactly once",
    "[scheduled_fallback]") {
  auto [env, cs, node] = setup::testing::setup_environment(Code::Oxygen);
  cp::EnvironmentTopology<DummyEnvironmentInterface> tree(*env->getUniverse());
  using stack_type = test::TransportStack<DummyEnvironment>::type;
  stack_type stack;
  cp::CpuParticleScheduler<stack_type> scheduler(stack);
  Model model;
  Processes processes;
  cp::ProposalCpuFallbackHandler<stack_type, Model, Processes, DummyEnvironment> handler(
      model, processes, *env, tree, *cs, 123, 456);
  auto parent = stack.addParticle(std::make_tuple(Code::Electron, 10_GeV,
                                                  DirectionVector(*cs, {0., 0., 1.}),
                                                  Point(*cs, 0_m, 0_m, 0_m), 0_s));
  parent.setNode(node);
  parent.setWeight(2.);
  auto selected = scheduler.acquireNext();
  cp::FallbackRequest event{};
  event.particle_ = cp::particle_transfer::to_device_state(
      selected.particle_, *cs, selected.historyId_, selected.parentHistoryId_,
      selected.generation_, selected.stepId_, tree);
  event.mediumHash_ = 7;
  event.interactionHash_ = 8;
  event.componentHash_ = 9;
  event.energyFraction_ = .5;
  event.processId_ = cp::ElectronPairProcessId;
  event.reason_ = cp::FallbackReason::CpuOnlyProcess;
  event.interactionVertexReached_ = 1;
  selected.particle_.erase();
  scheduler.completeParticleStep();
  REQUIRE(scheduler.empty());
  REQUIRE(handler.canHandle(event));
  handler.handle(stack, event);
  CHECK(processes.calls_ == 1);
  CHECK(stack.getEntries() == 2);
  std::vector<Code> pids;
  std::vector<transport::HistoryId> histories;
  HEPEnergyType total = 0_GeV;
  while (!scheduler.empty()) {
    auto s = scheduler.acquireNext();
    CHECK(s.parentHistoryId_ == selected.historyId_);
    CHECK(s.generation_ == selected.generation_ + 1);
    CHECK(s.historyId_ != selected.historyId_);
    CHECK(s.stepId_ == 0);
    CHECK(s.particle_.getNode() == node);
    CHECK(s.particle_.getWeight() == 2.);
    pids.push_back(s.particle_.getPID());
    histories.push_back(s.historyId_);
    total += s.particle_.getEnergy();
    s.particle_.erase();
    scheduler.completeParticleStep();
  }
  CHECK(pids == std::vector<Code>{Code::Photon, Code::Electron});
  REQUIRE(histories.size() == 2);
  CHECK(histories[0] > histories[1]);
  // The native addParticle/addSecondary argument is kinetic energy.
  CHECK(total == event.particle_.energyGeV_ * 1_GeV);
  CHECK(handler.statistics().specifiedInteractions_ == 1);
  CHECK(handler.statistics().generatedSecondaries_ == 2);
  CHECK(scheduler.statistics().acquiredParticleSteps_ == 3);
  CHECK(scheduler.statistics().completedParticleSteps_ == 3);
}
