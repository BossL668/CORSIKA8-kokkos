/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>
#include <corsika/kokkos/scheduling/HadronicWorkQueue.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using namespace corsika;
using namespace corsika::kokkos;

TEST_CASE("Hadronic work classification is physical and deterministic",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  config.transitionEnergyGeV_ = 80.;
  config.energyBinsPerOctave_ = 2;

  CHECK_FALSE(classify_hadronic_work(Code::Electron, 1_TeV, config));

  auto const low_proton = classify_hadronic_work(Code::Proton, 40_GeV, config);
  auto const high_proton = classify_hadronic_work(Code::Proton, 160_GeV, config);
  auto const pion = classify_hadronic_work(Code::PiMinus, 40_GeV, config);
  auto const kaon = classify_hadronic_work(Code::KPlus, 40_GeV, config);
  auto const iron = classify_hadronic_work(Code::Iron, 1_TeV, config);

  REQUIRE(low_proton);
  REQUIRE(high_proton);
  REQUIRE(pion);
  REQUIRE(kaon);
  REQUIRE(iron);
  CHECK(low_proton->model_ == HadronicModelClass::LowEnergy);
  CHECK(high_proton->model_ == HadronicModelClass::HighEnergy);
  CHECK(low_proton->species_ == HadronicSpeciesClass::Nucleon);
  CHECK(pion->species_ == HadronicSpeciesClass::Pion);
  CHECK(kaon->species_ == HadronicSpeciesClass::Kaon);
  CHECK(iron->species_ == HadronicSpeciesClass::Nucleus);
  CHECK(iron->massNumberBin_ > 0);
  CHECK(low_proton->energyBin_ < high_proton->energyBin_);

  auto const lower = hadronic_energy_bin_lower_gev(*low_proton, config);
  auto const upper = hadronic_energy_bin_upper_gev(*low_proton, config);
  CHECK(lower <= 40.);
  CHECK(upper > 40.);
  CHECK(classify_hadronic_work(Code::Proton, 40_GeV, config) == low_proton);
}

TEST_CASE("Hadronic class timing records tail cost", "[HadronicWorkQueue]") {
  HadronicWorkClassStatistics statistics;
  statistics.record(1.);
  statistics.record(2.);
  statistics.record(6.);

  CHECK(statistics.steps_ == 3);
  CHECK(statistics.totalTimeMs_ == Catch::Approx(9.));
  CHECK(statistics.meanTimeMs() == Catch::Approx(3.));
  CHECK(statistics.minimumTimeMs_ == Catch::Approx(1.));
  CHECK(statistics.maximumTimeMs_ == Catch::Approx(6.));
  CHECK(statistics.standardDeviationMs() == Catch::Approx(std::sqrt(14. / 3.)));
  CHECK_THROWS_AS(statistics.record(-1.), std::invalid_argument);
}

TEST_CASE("Hadronic scheduling cost is deterministic and class sensitive",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const low_energy_nucleon = *classify_hadronic_work(Code::Proton, 0.3_GeV, config);
  auto const high_energy_nucleon = *classify_hadronic_work(Code::Proton, 40_GeV, config);
  auto const low_energy_pion = *classify_hadronic_work(Code::PiPlus, 0.3_GeV, config);
  auto const high_energy_pion = *classify_hadronic_work(Code::PiPlus, 40_GeV, config);

  auto const reference = 0.1;
  auto const low_nucleon_cost =
      deterministic_hadronic_interaction_cost(low_energy_nucleon, reference);
  auto const high_nucleon_cost =
      deterministic_hadronic_interaction_cost(high_energy_nucleon, reference);
  auto const low_pion_cost =
      deterministic_hadronic_interaction_cost(low_energy_pion, reference);
  auto const high_pion_cost =
      deterministic_hadronic_interaction_cost(high_energy_pion, reference);

  CHECK(low_nucleon_cost > 0.);
  CHECK(high_nucleon_cost > low_nucleon_cost);
  CHECK(low_pion_cost > low_nucleon_cost);
  CHECK(high_pion_cost > low_pion_cost);
  CHECK(deterministic_hadronic_interaction_cost(high_energy_nucleon, reference) ==
        high_nucleon_cost);
  CHECK(deterministic_hadronic_interaction_cost(high_energy_nucleon, 2. * reference) ==
        Catch::Approx(2. * high_nucleon_cost));
  CHECK_THROWS_AS(deterministic_hadronic_interaction_cost(low_energy_nucleon, 0.),
                  std::invalid_argument);
}

TEST_CASE("Classified hadronic queue returns homogeneous cost-balanced batches",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const nucleon = *classify_hadronic_work(Code::Proton, 1_TeV, config);
  auto const pion = *classify_hadronic_work(Code::PiPlus, 10_GeV, config);

  ClassifiedHadronicWorkQueue<int> queue;
  queue.enqueue(nucleon, 10, 4., 100);
  queue.enqueue(nucleon, 11, 4., 101);
  queue.enqueue(nucleon, 12, 4., 102);
  queue.enqueue(pion, 20, 1., 200);
  queue.enqueue(pion, 21, 1., 201);
  queue.enqueue(pion, 22, 1., 202);
  queue.enqueue(pion, 23, 1., 203);

  CHECK(queue.size() == 7);
  CHECK(queue.classCount() == 2);
  CHECK(queue.pendingCost() == Catch::Approx(16.));

  std::set<std::uint64_t> seen;
  std::vector<double> batch_costs;
  while (!queue.empty()) {
    auto batch = queue.popBalancedBatch(8., 8);
    REQUIRE_FALSE(batch.payloads_.empty());
    REQUIRE(batch.payloads_.size() == batch.sequenceIds_.size());
    CHECK(batch.estimatedCost_ <= 8.);
    batch_costs.push_back(batch.estimatedCost_);
    for (auto const id : batch.sequenceIds_) { CHECK(seen.insert(id).second); }

    if (batch.key_ == nucleon) {
      CHECK(
          std::all_of(batch.payloads_.begin(), batch.payloads_.end(),
                      [](int const payload) { return payload >= 100 && payload < 200; }));
    } else {
      CHECK(batch.key_ == pion);
      CHECK(std::all_of(batch.payloads_.begin(), batch.payloads_.end(),
                        [](int const payload) { return payload >= 200; }));
    }
  }

  CHECK(seen.size() == 7);
  CHECK(queue.pendingCost() == 0.);
  CHECK(batch_costs.front() == Catch::Approx(8.));
  CHECK_THROWS_AS(queue.popBalancedBatch(8., 8), std::logic_error);
}

TEST_CASE("Hadronic live flush uses total predicted worker cost", "[HadronicWorkQueue]") {
  CHECK_FALSE(decide_hadronic_queue_flush(63, 100., 4, 64, 5., 256));
  CHECK_FALSE(decide_hadronic_queue_flush(128, 19.99, 4, 64, 5., 256));

  auto const cost_trigger = decide_hadronic_queue_flush(128, 20., 4, 64, 5., 256);
  REQUIRE(cost_trigger);
  CHECK(*cost_trigger == HadronicQueueFlushTrigger::EstimatedCost);

  auto const capacity_trigger = decide_hadronic_queue_flush(1024, 19., 4, 64, 5., 256);
  REQUIRE(capacity_trigger);
  CHECK(*capacity_trigger == HadronicQueueFlushTrigger::Capacity);

  // Draining is a cascade-state decision rather than a queue-size decision.
  CHECK_FALSE(decide_hadronic_queue_flush(10, 1., 4, 64, 5., 256));
  CHECK(hadronic_queue_flush_trigger_name(HadronicQueueFlushTrigger::EstimatedCost) ==
        "estimated_cost");
  CHECK(hadronic_queue_flush_trigger_name(HadronicQueueFlushTrigger::Capacity) ==
        "capacity");
  CHECK(hadronic_queue_flush_trigger_name(HadronicQueueFlushTrigger::Drain) == "drain");
  CHECK_THROWS_AS(decide_hadronic_queue_flush(64, 20., 0, 64, 5., 256),
                  std::invalid_argument);
  CHECK_THROWS_AS(decide_hadronic_queue_flush(64, -1., 4, 64, 5., 256),
                  std::invalid_argument);
}

TEST_CASE("Hadronic worker assignments are balanced and deterministic",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const nucleon = *classify_hadronic_work(Code::Proton, 1_TeV, config);
  auto const pion = *classify_hadronic_work(Code::PiPlus, 10_GeV, config);

  auto makeQueue = [&]() {
    ClassifiedHadronicWorkQueue<int> queue;
    queue.enqueue(nucleon, 10, 8., 100);
    queue.enqueue(nucleon, 11, 8., 101);
    queue.enqueue(nucleon, 12, 8., 102);
    for (std::uint64_t i = 0; i < 6; ++i) {
      queue.enqueue(pion, 20 + i, 2., 200 + static_cast<int>(i));
    }
    return queue;
  };

  auto queue = makeQueue();
  auto assignments = plan_hadronic_worker_assignments(queue, 3, 8., 16);
  REQUIRE(queue.empty());
  REQUIRE(assignments.size() == 3);

  double minimumCost = std::numeric_limits<double>::infinity();
  double maximumCost = 0.;
  std::set<std::uint64_t> sequenceIds;
  for (std::size_t worker = 0; worker < assignments.size(); ++worker) {
    auto const& assignment = assignments[worker];
    CHECK(assignment.workerId_ == worker);
    minimumCost = std::min(minimumCost, assignment.estimatedCost_);
    maximumCost = std::max(maximumCost, assignment.estimatedCost_);
    for (auto const& batch : assignment.batches_) {
      REQUIRE_FALSE(batch.payloads_.empty());
      CHECK(batch.estimatedCost_ <= 8.);
      for (auto const sequenceId : batch.sequenceIds_) {
        CHECK(sequenceIds.insert(sequenceId).second);
      }
    }
  }
  CHECK(sequenceIds.size() == 9);
  // LPT guarantees that the imbalance is no larger than the largest batch here.
  CHECK(maximumCost - minimumCost <= 8.);

  auto repeatedQueue = makeQueue();
  auto repeated = plan_hadronic_worker_assignments(repeatedQueue, 3, 8., 16);
  REQUIRE(repeated.size() == assignments.size());
  for (std::size_t worker = 0; worker < assignments.size(); ++worker) {
    CHECK(repeated[worker].estimatedCost_ ==
          Catch::Approx(assignments[worker].estimatedCost_));
    REQUIRE(repeated[worker].batches_.size() == assignments[worker].batches_.size());
    for (std::size_t batch = 0; batch < assignments[worker].batches_.size(); ++batch) {
      CHECK(repeated[worker].batches_[batch].key_ ==
            assignments[worker].batches_[batch].key_);
      CHECK(repeated[worker].batches_[batch].sequenceIds_ ==
            assignments[worker].batches_[batch].sequenceIds_);
      CHECK(repeated[worker].batches_[batch].payloads_ ==
            assignments[worker].batches_[batch].payloads_);
    }
  }

  CHECK_THROWS_AS(plan_hadronic_worker_assignments(repeatedQueue, 0, 8., 16),
                  std::invalid_argument);
}

TEST_CASE("Hadronic classified micro-batches coalesce once per worker",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const nucleon = *classify_hadronic_work(Code::Proton, 1_TeV, config);
  auto const pion = *classify_hadronic_work(Code::PiPlus, 10_GeV, config);

  ClassifiedHadronicWorkQueue<int> queue;
  queue.enqueue(nucleon, 10, 4., 100);
  queue.enqueue(nucleon, 11, 4., 101);
  queue.enqueue(nucleon, 12, 4., 102);
  queue.enqueue(pion, 20, 1., 200);
  queue.enqueue(pion, 21, 1., 201);
  queue.enqueue(pion, 22, 1., 202);
  queue.enqueue(pion, 23, 1., 203);

  auto assignments = plan_hadronic_worker_assignments(queue, 2, 4., 8);
  std::vector<std::vector<std::uint64_t>> expected_sequence_ids(assignments.size());
  std::vector<std::vector<int>> expected_payloads(assignments.size());
  std::vector<std::size_t> expected_classified_batches(assignments.size());
  for (auto const& assignment : assignments) {
    expected_classified_batches[assignment.workerId_] = assignment.batches_.size();
    for (auto const& batch : assignment.batches_) {
      expected_sequence_ids[assignment.workerId_].insert(
          expected_sequence_ids[assignment.workerId_].end(), batch.sequenceIds_.begin(),
          batch.sequenceIds_.end());
      expected_payloads[assignment.workerId_].insert(
          expected_payloads[assignment.workerId_].end(), batch.payloads_.begin(),
          batch.payloads_.end());
    }
  }

  auto super_batches = coalesce_hadronic_worker_assignments(assignments);
  REQUIRE(super_batches.size() == 2);
  std::set<std::uint64_t> seen;
  for (auto const& batch : super_batches) {
    REQUIRE(batch.workerId_ < 2);
    CHECK(batch.classifiedBatches_ == expected_classified_batches[batch.workerId_]);
    CHECK(batch.sequenceIds_ == expected_sequence_ids[batch.workerId_]);
    CHECK(batch.payloads_ == expected_payloads[batch.workerId_]);
    REQUIRE(batch.sequenceIds_.size() == batch.payloads_.size());
    for (auto const sequence_id : batch.sequenceIds_) {
      CHECK(seen.insert(sequence_id).second);
    }
  }
  CHECK(seen.size() == 7);
}

TEST_CASE("Hadronic process protocol is POD, keyed, and fail-fast",
          "[HadronicBatchProtocol]") {
  HadronicBatchHeader header;
  header.requestCount_ = 2;
  header.batchId_ = 7;
  CHECK_NOTHROW(validate_hadronic_batch_header(header));

  auto invalidHeader = header;
  invalidHeader.protocolVersion_ += 1;
  CHECK_THROWS_AS(validate_hadronic_batch_header(invalidHeader), std::invalid_argument);

  HadronicInteractionRequest request;
  request.sequenceId_ = 19;
  request.randomKey_ = HadronicRandomKey{12345, 2, 991, 17, 3, 0};
  request.projectilePdg_ = 2212;
  request.targetPdg_ = 1000080160;
  request.projectileFourMomentumGeV_ = {100.01, 0., 0., 100.};
  request.targetFourMomentumGeV_ = {14.899, 0., 0., 0.};
  CHECK_NOTHROW(validate_hadronic_interaction_request(request));

  auto const seed = derive_hadronic_worker_seed(request.randomKey_);
  CHECK(seed == derive_hadronic_worker_seed(request.randomKey_));
  auto changedKey = request.randomKey_;
  ++changedKey.historyId_;
  CHECK(seed != derive_hadronic_worker_seed(changedKey));

  auto invalidRequest = request;
  invalidRequest.projectileFourMomentumGeV_[2] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(validate_hadronic_interaction_request(invalidRequest),
                  std::invalid_argument);
}
