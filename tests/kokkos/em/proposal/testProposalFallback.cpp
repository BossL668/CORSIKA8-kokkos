/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/ProposalFallbackAdapter.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <limits>

using namespace corsika::kokkos;
namespace {
  FallbackRequest fixture() {
    FallbackRequest event{};
    event.particle_.pid_ = 13;
    event.particle_.energyGeV_ = 100.;
    event.particle_.positionM_[0] = 1.;
    event.particle_.positionM_[1] = -2.;
    event.particle_.direction_[2] = 1.;
    event.particle_.timeS_ = 3.e-8;
    event.particle_.historyId_ = 123;
    event.particle_.stepId_ = 7;
    event.interactionVertexReached_ = 1;
    event.processId_ = ElectronPairProcessId;
    event.reason_ = FallbackReason::CpuOnlyProcess;
    event.componentHash_ = 11;
    event.mediumHash_ = 12;
    event.interactionHash_ = 13;
    event.energyFraction_ = .2;
    event.selectionUniform_ = .3;
    event.lossQuantile_ = .4;
    event.randomProcessId_ = 101;
    event.randomDrawId_ = 27;
    return event;
  }
} // namespace
TEST_CASE("CPU fallback preserves vertex identity and units", "[proposal_fallback]") {
  auto event = fixture();
  REQUIRE(permits_specified_proposal_cpu_final_state(event));
  auto record = make_proposal_interaction_record(event, 19, 23);
  CHECK(record.context_.projectileId_ == corsika::Code::MuMinus);
  CHECK(record.context_.projectileEnergyMeV_ == 100000.);
  CHECK(record.context_.positionCm_[0] == 100.);
  CHECK(record.context_.positionCm_[1] == -200.);
  CHECK(record.context_.direction_[2] == 1.);
  CHECK(record.context_.timeS_ == event.particle_.timeS_);
  CHECK(record.context_.mediumHash_ == 12);
  CHECK(record.interactionHash_ == 13);
  CHECK(record.componentHash_ == 11);
  CHECK(record.vLoss_ == .2);
  CHECK(record.selectionUniform_ == .3);
  REQUIRE(record.randomKey_.has_value());
  CHECK(record.randomKey_->seed_ == 19);
  CHECK(record.randomKey_->showerId_ == 23);
  CHECK(record.randomKey_->historyId_ == 123);
  CHECK(record.randomKey_->stepId_ == 7);
  CHECK(record.randomKey_->processId_ == 101);
  CHECK(record.randomKey_->drawId_ == 27);
  for (int reason = 0; reason <= 29; ++reason) {
    CAPTURE(reason);
    event.reason_ = static_cast<FallbackReason>(reason);
    CHECK(permits_specified_proposal_cpu_final_state(event) ==
          (reason == 12 || reason == 27 || reason == 29));
  }
  for (int pid : {11, -11, 13, -13}) {
    event = fixture();
    event.particle_.pid_ = pid;
    CHECK(permits_specified_proposal_cpu_final_state(event));
  }
  event = fixture();
  event.processId_ = BremsProcessId;
  CHECK_FALSE(permits_specified_proposal_cpu_final_state(event));
  for (int kind = 0; kind < 7; ++kind) {
    event = fixture();
    if (kind == 0) event.interactionVertexReached_ = 0;
    if (kind == 1) event.interactionHash_ = 0;
    if (kind == 2) event.particle_.energyGeV_ = -1.;
    if (kind == 3) event.particle_.direction_[2] = 0.;
    if (kind == 4) event.selectionUniform_ = 1.;
    if (kind == 5) event.processId_ = -999;
    if (kind == 6)
      event.particle_.positionM_[0] = std::numeric_limits<double>::infinity();
    CHECK_FALSE(has_specified_proposal_interaction_identity(event));
    CHECK_THROWS_AS(make_proposal_interaction_record(event, 19, 23),
                    std::invalid_argument);
  }
}
TEST_CASE("Native replay and conditional-loss completion retain distinct contracts",
          "[proposal_fallback]") {
  auto event = fixture();
  event.reason_ = FallbackReason::NativeSelectionReplay;
  event.energyFraction_ = std::numeric_limits<double>::quiet_NaN();
  for (double quantile : {0., 1., std::numeric_limits<double>::quiet_NaN()}) {
    event.lossQuantile_ = quantile;
    CHECK(has_resolvable_proposal_selected_loss(event));
    auto record = make_proposal_interaction_record_for_selected_loss(event, 19, 23);
    CHECK(record.selectionUniform_ == .3);
    CHECK(record.componentHash_ == 11);
  }
  event.reason_ = FallbackReason::InverseCdfUnavailable;
  event.lossQuantile_ = .5;
  CHECK(has_resolvable_proposal_selected_loss(event));
  CHECK_FALSE(has_specified_proposal_final_state(event));
  event.lossQuantile_ = 1.;
  CHECK_FALSE(has_resolvable_proposal_selected_loss(event));
  CHECK_THROWS_AS(make_proposal_interaction_record_for_selected_loss(event, 19, 23),
                  std::invalid_argument);
}
TEST_CASE("CPU final-state random draws do not depend on queue ordering",
          "[proposal_fallback]") {
  std::array<FallbackRequest, 16> events;
  std::array<std::vector<double>, 16> expected;
  for (std::size_t i = 0; i < events.size(); ++i) {
    events[i] = fixture();
    events[i].particle_.historyId_ = i;
    expected[i] = make_proposal_final_state_random_numbers(events[i], 7, 19, 23);
    for (std::size_t k = 0; k < 7; ++k) {
      RandomNumberKey key{19,
                          23,
                          i,
                          7,
                          static_cast<std::uint32_t>(ElectronPairProcessId),
                          CpuProposalFinalStateDrawIdBase + k};
      CHECK(expected[i][k] == uniform_open01(key));
    }
  }
  std::reverse(events.begin(), events.end());
  for (auto& event : events) {
    event.inputIndex_ += 1000;
    CHECK(make_proposal_final_state_random_numbers(event, 7, 19, 23) ==
          expected[event.particle_.historyId_]);
  }
}
