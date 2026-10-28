/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include "PhotonCascadeOracle.hpp"
using namespace photon_cascade_test;
TEST_CASE("Photon cascade empty input and invalid limits", "[photon_cascade]") {
  ensure_runtime();
  Fixture f;
  auto c = f.context_.getDeviceView();
  std::optional<cp::FirstInteractionRecord> first;
  auto empty = cp::run_photon_cascade<execution_space_type>(c, {}, 0, 0, 0, first);
  REQUIRE(empty.completed_);
  REQUIRE(empty.inputParticles_ == 0);
  REQUIRE(empty.wavefronts_ == 0);
  auto p = particles(3);
  REQUIRE_THROWS_AS(cp::run_photon_cascade<execution_space_type>(c, p, 0, 2, 1, first),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      cp::run_photon_cascade<execution_space_type>(c, p, FirstHistory, 0, 1, first),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      cp::run_photon_cascade<execution_space_type>(c, p, FirstHistory, 2, 0, first),
      std::invalid_argument);
  auto result =
      cp::run_photon_cascade<execution_space_type>(c, p, FirstHistory, 2, 1, first);
  REQUIRE(result.completed_);
  REQUIRE(result.fallbackEvents_.size() == 3);
  REQUIRE(result.wavefronts_ == 1);
  REQUIRE_FALSE(first);
  for (std::size_t i = 0; i < p.size(); ++i)
    equal(result.fallbackEvents_[i].particle_, p[i]);
}
TEST_CASE("Photon cascade budget checkpoints preserve pending input",
          "[photon_cascade]") {
  ensure_runtime();
  Fixture f;
  auto p = particles(4);
  auto q = particles(2);
  q[0].historyId_ += 100;
  q[1].historyId_ += 100;
  for (unsigned deny : {1, 2, 3}) {
    cp::PendingParticleQueue<execution_space_type> pending("test pending photons");
    auto device = upload(p);
    REQUIRE(pending.tryAppend(device, p.size(), 100, f.execution_));
    f.execution_.fence("pending input ready");
    cp::PhotonFrontWorkspace<execution_space_type> workspace;
    std::optional<cp::FirstInteractionRecord> first;
    Budget budget;
    budget.reject_ = deny;
    auto result = cp::run_photon_cascade<execution_space_type>(
        f.context_.getDeviceView(), q, FirstHistory, 5, 1, first, false, {}, nullptr,
        &workspace, &pending, nullptr, 3, 100, f.execution_, false, 0, budget.gate());
    REQUIRE(result.workspaceLimitCheckpoint_);
    REQUIRE(result.completed_);
    REQUIRE(result.inputParticles_ == 5);
    REQUIRE(result.wavefronts_ == 0);
    REQUIRE(result.cpuSpillParticles_.size() == 5);
    REQUIRE(pending.size() == 1);
    equal(pending.downloadPrefix(1, f.execution_)[0], p[3]);
    std::vector<cp::ParticleState> expected(p.begin(), p.begin() + 3);
    expected.insert(expected.end(), q.begin(), q.end());
    equal(result.cpuSpillParticles_, expected);
    REQUIRE(budget.calls_ == deny);
    REQUIRE(workspace.queue_.empty());
  }
}

#ifdef CORSIKA_TEST_CASCADE_NATIVE
TEST_CASE("Photon cascade native rounds match the independent front driver",
          "[photon_cascade][native]") {
  ensure_runtime();
  auto const table = cp::test::make_native_tables();
  auto const lpm = cp::test::make_auxiliary_dataset().snapshots_.front().photon_;
  std::size_t runs = 0, rounds = 0, suppressions = 0;
  std::map<int, std::size_t> channels;
  auto check = [&](int process, double density, bool thin, std::size_t n, double energy,
                   unsigned mode) {
    INFO("process " << process << " density " << density << " thinning " << thin
                    << " count " << n << " mode " << mode);
    Fixture f(table, process, density, lpm, thin);
    auto input = particles(n, energy);
    if (n >= 255)
      for (std::size_t i = 0; i < n; ++i) input[i].energyGeV_ *= 1u << (i % 4);
    std::optional<cp::FirstInteractionRecord> first, expectedFirst;
    cp::ProfileAccumulatorStorage<execution_space_type> profile, referenceProfile;
    bool const withProfile = mode == 2, project = mode != 0;
    if (withProfile) {
      profile.initialize(f.profile_, f.execution_);
      referenceProfile.initialize(f.profile_, f.execution_);
    }
    auto actual = cp::run_photon_cascade<execution_space_type>(
        f.context_.getDeviceView(), input, FirstHistory, 4, 1, first, project,
        f.projection_, withProfile ? &profile : nullptr, nullptr, nullptr, nullptr, 0,
        100000, f.execution_, withProfile);
    auto expected =
        reference_loop(f, input, 4, 1, expectedFirst, project,
                       withProfile ? &referenceProfile : nullptr, withProfile);
    equal(actual, expected);
    equal(first, expectedFirst);
    if (withProfile) {
      auto a =
          profile.downloadFixed("cascade", cp::CooperativeEndpoint::OpenMP, f.execution_);
      auto b = referenceProfile.downloadFixed("cascade", cp::CooperativeEndpoint::OpenMP,
                                              f.execution_);
      REQUIRE(a.histograms_ == b.histograms_);
      equal(a.counters_, b.counters_);
      REQUIRE(a.counters_.fixedPointOverflows_ == 0);
      REQUIRE(a.counters_.invalidRecords_ == 0);
      REQUIRE(actual.finalStateRecords_.empty());
    }
    for (auto const& r : expected.finalStateRecords_) ++channels[r.processId_];
    for (auto const& r : expected.fallbackEvents_) ++channels[r.processId_];
    ++runs;
    rounds += actual.wavefronts_;
    suppressions += actual.lpmSuppressions_;
  };
  for (int process :
       {0, cp::PhotonPairProcessId, cp::ComptonProcessId, cp::PhotoelectricProcessId,
        cp::PhotonMuonPairProcessId, cp::PhotoproductionProcessId})
    for (double density : {.001, 1.})
      for (bool thin : {false, true}) check(process, density, thin, 33, 1000., runs % 3);
  for (std::size_t n : {255, 256, 257}) check(cp::ComptonProcessId, 1., false, n, 1., 0);
  check(cp::PhotonPairProcessId, 1.e6, false, 257, 1.e4, 2);
  REQUIRE(suppressions > 0);
  REQUIRE(rounds > runs);
  for (int process :
       {cp::PhotonPairProcessId, cp::ComptonProcessId, cp::PhotoelectricProcessId,
        cp::PhotonMuonPairProcessId, cp::PhotoproductionProcessId})
    REQUIRE(channels[process] > 0);
  std::cout << "cascade native configurations=" << runs << " rounds=" << rounds
            << " LPM suppressions=" << suppressions << "\n";
}
TEST_CASE("Photon cascade checkpoints and reused workspace preserve histories",
          "[photon_cascade][native]") {
  ensure_runtime();
  auto const table = cp::test::make_native_tables();
  Fixture f(table, cp::ComptonProcessId, 1.);
  cp::PhotonFrontWorkspace<execution_space_type> workspace;
  cp::ResidentExecutionWait<execution_space_type> waiter;
#ifdef CORSIKA_EXPECT_CUDA
  waiter.setBlocking(true);
#endif
  auto input = particles(1);
  std::optional<cp::FirstInteractionRecord> first, expectedFirst;
  auto a = cp::run_photon_cascade<execution_space_type>(
      f.context_.getDeviceView(), input, FirstHistory, 4, 2, first, false, {}, nullptr,
      &workspace, nullptr, nullptr, 0, 100, f.execution_, false, 0, {}, &waiter);
  auto expected = reference_loop(f, input, 4, 2, expectedFirst);
  equal(a, expected);
  equal(first, expectedFirst);
  REQUIRE(a.wavefronts_ == 1);
  REQUIRE(a.belowMinimumBatchCheckpoint_);
  REQUIRE(a.remainingPhotons_.size() == 1);
  auto nextHistory = FirstHistory + a.processStatistics_.physicalSecondariesGenerated_;
  auto b = cp::run_photon_cascade<execution_space_type>(
      f.context_.getDeviceView(), a.remainingPhotons_, nextHistory, 3, 1, first, false,
      {}, nullptr, &workspace);
  auto expectedB = reference_loop(f, expected.remainingPhotons_, 3, 1, expectedFirst,
                                  false, nullptr, false, nextHistory);
  equal(b, expectedB);
  equal(first, expectedFirst);
  auto completeFirst = std::optional<cp::FirstInteractionRecord>{};
  auto complete = reference_loop(f, input, 4, 1, completeFirst);
  a.electromagneticSecondaries_.insert(a.electromagneticSecondaries_.end(),
                                       b.electromagneticSecondaries_.begin(),
                                       b.electromagneticSecondaries_.end());
  equal(a.electromagneticSecondaries_, complete.electromagneticSecondaries_);
  equal(b.remainingPhotons_, complete.remainingPhotons_);
  REQUIRE(a.wavefronts_ + b.wavefronts_ == complete.wavefronts_);
  equal(first, completeFirst);
  // Refuse growth after one round: the exact next front must be returned for CPU
  // handling.
  Budget budget;
  budget.reject_ = 4;
  std::optional<cp::FirstInteractionRecord> spillFirst;
  auto spill = cp::run_photon_cascade<execution_space_type>(
      f.context_.getDeviceView(), input, FirstHistory, 4, 1, spillFirst, false, {},
      nullptr, &workspace, nullptr, nullptr, 0, 100, f.execution_, false, 0,
      budget.gate());
  REQUIRE(spill.workspaceLimitCheckpoint_);
  REQUIRE(spill.wavefronts_ == 1);
  REQUIRE(spill.completed_);
  REQUIRE(workspace.queue_.empty());
  equal(spill.cpuSpillParticles_, expected.remainingPhotons_);
}
TEST_CASE("Photon cascade pending queues retain ownership on success and refusal",
          "[photon_cascade][native]") {
  ensure_runtime();
  auto const table = cp::test::make_native_tables();
  Fixture f(table, cp::PhotonPairProcessId, 1.,
            cp::test::make_auxiliary_dataset().snapshots_.front().photon_);
  auto input = particles(8);
  auto host = particles(3);
  for (auto& p : host) {
    p.historyId_ += 100;
    p.generation_ = 2;
  }
  std::vector<cp::ParticleState> combined(input.begin(), input.begin() + 5);
  combined.insert(combined.end(), host.begin(), host.end());
  std::optional<cp::FirstInteractionRecord> expectedFirst;
  auto expected = reference_loop(f, combined, 3, 1, expectedFirst);
  REQUIRE_FALSE(expected.electromagneticSecondaries_.empty());
  for (unsigned mode : {0, 1, 2}) {
    cp::PendingParticleQueue<execution_space_type> photons("pending photons"),
        leptons("pending leptons");
    REQUIRE(photons.tryAppend(upload(input), input.size(), 100, f.execution_));
    Budget budget;
    budget.reject_ = mode == 2 ? 4 : 0;
    std::optional<cp::FirstInteractionRecord> first;
    auto actual = cp::run_photon_cascade<execution_space_type>(
        f.context_.getDeviceView(), host, FirstHistory, 3, 1, first, false, {}, nullptr,
        nullptr, &photons, &leptons, 5, mode == 1 ? 0 : 100, f.execution_, false, 0,
        budget.gate());
    REQUIRE(photons.size() == 3);
    equal(photons.downloadPrefix(3, f.execution_),
          std::vector<cp::ParticleState>(input.begin() + 5, input.end()));
    REQUIRE(actual.electromagneticSecondaries_.empty());
    if (mode == 0) {
      equal(leptons.downloadPrefix(leptons.size(), f.execution_),
            expected.electromagneticSecondaries_);
      actual.electromagneticSecondaries_ =
          leptons.downloadPrefix(leptons.size(), f.execution_);
    } else {
      REQUIRE(leptons.empty());
      equal(actual.cpuSpillParticles_, expected.electromagneticSecondaries_);
      actual.electromagneticSecondaries_ = std::move(actual.cpuSpillParticles_);
      actual.cpuSpillParticles_.clear();
    }
    equal(actual, expected);
    equal(first, expectedFirst);
  }
}
TEST_CASE("Photon cascade invalid pending input and exceptional exits remain drained",
          "[photon_cascade][native]") {
  ensure_runtime();
  auto const table = cp::test::make_native_tables();
  Fixture f(table, cp::PhotonPairProcessId, 1.,
            cp::test::make_auxiliary_dataset().snapshots_.front().photon_);
  cp::PhotonFrontWorkspace<execution_space_type> workspace;
  cp::PendingParticleQueue<execution_space_type> pending("empty");
  std::optional<cp::FirstInteractionRecord> first;
  auto input = particles(2);
  REQUIRE_THROWS_AS(cp::run_photon_cascade<execution_space_type>(
                        f.context_.getDeviceView(), {}, FirstHistory, 2, 1, first, false,
                        {}, nullptr, &workspace, nullptr, nullptr, 1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(cp::run_photon_cascade<execution_space_type>(
                        f.context_.getDeviceView(), {}, FirstHistory, 2, 1, first, false,
                        {}, nullptr, &workspace, &pending, nullptr, 1),
                    std::invalid_argument);
  input[1].generation_ = 0;
  REQUIRE_THROWS_AS(cp::run_photon_cascade<execution_space_type>(
                        f.context_.getDeviceView(), input, FirstHistory, 2, 1, first,
                        false, {}, nullptr, &workspace),
                    std::runtime_error);
  input[1].generation_ = 2;
  REQUIRE_THROWS_AS(
      cp::run_photon_cascade<execution_space_type>(
          f.context_.getDeviceView(), input, std::numeric_limits<std::uint64_t>::max(), 2,
          1, first, false, {}, nullptr, &workspace),
      std::overflow_error);
  first.reset();
  std::optional<cp::FirstInteractionRecord> expectedFirst;
  auto actual = cp::run_photon_cascade<execution_space_type>(
      f.context_.getDeviceView(), input, FirstHistory, 2, 1, first, false, {}, nullptr,
      &workspace);
  auto expected = reference_loop(f, input, 2, 1, expectedFirst);
  equal(actual, expected);
  equal(first, expectedFirst);
  // Existing snapshots are never overwritten by subsequent work.
  REQUIRE(first);
  auto preserved = first;
  input[1].generation_ = 0;
  REQUIRE_NOTHROW(cp::run_photon_cascade<execution_space_type>(
      f.context_.getDeviceView(), input, FirstHistory + 100, 2, 1, first, false, {},
      nullptr, &workspace));
  equal(first, preserved);
}
#endif
