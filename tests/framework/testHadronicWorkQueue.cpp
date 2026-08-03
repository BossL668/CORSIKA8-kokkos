/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/HadronicBatchProtocol.hpp>
#include <corsika/framework/core/HadronicWorkQueue.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using namespace corsika;

TEST_CASE("Hadronic work classification is physical and deterministic",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  config.transition_energy_GeV = 80.;
  config.energy_bins_per_octave = 2;

  CHECK_FALSE(classifyHadronicWork(
      Code::Electron, 1_TeV, config));

  auto const low_proton =
      classifyHadronicWork(Code::Proton, 40_GeV, config);
  auto const high_proton =
      classifyHadronicWork(Code::Proton, 160_GeV, config);
  auto const pion =
      classifyHadronicWork(Code::PiMinus, 40_GeV, config);
  auto const kaon =
      classifyHadronicWork(Code::KPlus, 40_GeV, config);
  auto const iron =
      classifyHadronicWork(Code::Iron, 1_TeV, config);

  REQUIRE(low_proton);
  REQUIRE(high_proton);
  REQUIRE(pion);
  REQUIRE(kaon);
  REQUIRE(iron);
  CHECK(low_proton->model == HadronicModelClass::LowEnergy);
  CHECK(high_proton->model == HadronicModelClass::HighEnergy);
  CHECK(low_proton->species == HadronicSpeciesClass::Nucleon);
  CHECK(pion->species == HadronicSpeciesClass::Pion);
  CHECK(kaon->species == HadronicSpeciesClass::Kaon);
  CHECK(iron->species == HadronicSpeciesClass::Nucleus);
  CHECK(iron->mass_number_bin > 0);
  CHECK(low_proton->energy_bin < high_proton->energy_bin);

  auto const lower =
      hadronicEnergyBinLowerGeV(*low_proton, config);
  auto const upper =
      hadronicEnergyBinUpperGeV(*low_proton, config);
  CHECK(lower <= 40.);
  CHECK(upper > 40.);
  CHECK(classifyHadronicWork(Code::Proton, 40_GeV, config) ==
        low_proton);
}

TEST_CASE("Hadronic class timing records tail cost",
          "[HadronicWorkQueue]") {
  HadronicWorkClassStatistics statistics;
  statistics.record(1.);
  statistics.record(2.);
  statistics.record(6.);

  CHECK(statistics.steps == 3);
  CHECK(statistics.total_time_ms == Catch::Approx(9.));
  CHECK(statistics.meanTimeMs() == Catch::Approx(3.));
  CHECK(statistics.minimum_time_ms == Catch::Approx(1.));
  CHECK(statistics.maximum_time_ms == Catch::Approx(6.));
  CHECK(statistics.standardDeviationMs() ==
        Catch::Approx(std::sqrt(14. / 3.)));
  CHECK_THROWS_AS(statistics.record(-1.), std::invalid_argument);
}

TEST_CASE("Hadronic scheduling cost is deterministic and class sensitive",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const low_energy_nucleon =
      *classifyHadronicWork(
          Code::Proton, 0.3_GeV, config);
  auto const high_energy_nucleon =
      *classifyHadronicWork(
          Code::Proton, 40_GeV, config);
  auto const low_energy_pion =
      *classifyHadronicWork(
          Code::PiPlus, 0.3_GeV, config);
  auto const high_energy_pion =
      *classifyHadronicWork(
          Code::PiPlus, 40_GeV, config);

  auto const reference = 0.1;
  auto const low_nucleon_cost =
      deterministicHadronicInteractionCost(
          low_energy_nucleon, reference);
  auto const high_nucleon_cost =
      deterministicHadronicInteractionCost(
          high_energy_nucleon, reference);
  auto const low_pion_cost =
      deterministicHadronicInteractionCost(
          low_energy_pion, reference);
  auto const high_pion_cost =
      deterministicHadronicInteractionCost(
          high_energy_pion, reference);

  CHECK(low_nucleon_cost > 0.);
  CHECK(high_nucleon_cost > low_nucleon_cost);
  CHECK(low_pion_cost > low_nucleon_cost);
  CHECK(high_pion_cost > low_pion_cost);
  CHECK(
      deterministicHadronicInteractionCost(
          high_energy_nucleon, reference) ==
      high_nucleon_cost);
  CHECK(
      deterministicHadronicInteractionCost(
          high_energy_nucleon, 2. * reference) ==
      Catch::Approx(2. * high_nucleon_cost));
  CHECK_THROWS_AS(
      deterministicHadronicInteractionCost(
          low_energy_nucleon, 0.),
      std::invalid_argument);
}

TEST_CASE("Classified hadronic queue returns homogeneous cost-balanced batches",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const nucleon = *classifyHadronicWork(
      Code::Proton, 1_TeV, config);
  auto const pion = *classifyHadronicWork(
      Code::PiPlus, 10_GeV, config);

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
    REQUIRE_FALSE(batch.payloads.empty());
    REQUIRE(batch.payloads.size() ==
            batch.sequence_ids.size());
    CHECK(batch.estimated_cost <= 8.);
    batch_costs.push_back(batch.estimated_cost);
    for (auto const id : batch.sequence_ids) {
      CHECK(seen.insert(id).second);
    }

    if (batch.key == nucleon) {
      CHECK(std::all_of(
          batch.payloads.begin(), batch.payloads.end(),
          [](int const payload) {
            return payload >= 100 && payload < 200;
          }));
    } else {
      CHECK(batch.key == pion);
      CHECK(std::all_of(
          batch.payloads.begin(), batch.payloads.end(),
          [](int const payload) {
            return payload >= 200;
          }));
    }
  }

  CHECK(seen.size() == 7);
  CHECK(queue.pendingCost() == 0.);
  CHECK(batch_costs.front() == Catch::Approx(8.));
  CHECK_THROWS_AS(queue.popBalancedBatch(8., 8),
                  std::logic_error);
}

TEST_CASE("Hadronic live flush uses total predicted worker cost",
          "[HadronicWorkQueue]") {
  CHECK_FALSE(decideHadronicQueueFlush(
      63, 100., 4, 64, 5., 256));
  CHECK_FALSE(decideHadronicQueueFlush(
      128, 19.99, 4, 64, 5., 256));

  auto const cost_trigger = decideHadronicQueueFlush(
      128, 20., 4, 64, 5., 256);
  REQUIRE(cost_trigger);
  CHECK(
      *cost_trigger ==
      HadronicQueueFlushTrigger::EstimatedCost);

  auto const capacity_trigger = decideHadronicQueueFlush(
      1024, 19., 4, 64, 5., 256);
  REQUIRE(capacity_trigger);
  CHECK(
      *capacity_trigger ==
      HadronicQueueFlushTrigger::Capacity);

  // Draining is a cascade-state decision rather than a queue-size decision.
  CHECK_FALSE(decideHadronicQueueFlush(
      10, 1., 4, 64, 5., 256));
  CHECK(
      hadronicQueueFlushTriggerName(
          HadronicQueueFlushTrigger::EstimatedCost) ==
      "estimated_cost");
  CHECK(
      hadronicQueueFlushTriggerName(
          HadronicQueueFlushTrigger::Capacity) ==
      "capacity");
  CHECK(
      hadronicQueueFlushTriggerName(
          HadronicQueueFlushTrigger::Drain) ==
      "drain");
  CHECK_THROWS_AS(
      decideHadronicQueueFlush(
          64, 20., 0, 64, 5., 256),
      std::invalid_argument);
  CHECK_THROWS_AS(
      decideHadronicQueueFlush(
          64, -1., 4, 64, 5., 256),
      std::invalid_argument);
}

TEST_CASE("Hadronic worker assignments are balanced and deterministic",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const nucleon = *classifyHadronicWork(
      Code::Proton, 1_TeV, config);
  auto const pion = *classifyHadronicWork(
      Code::PiPlus, 10_GeV, config);

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
  auto assignments =
      planHadronicWorkerAssignments(queue, 3, 8., 16);
  REQUIRE(queue.empty());
  REQUIRE(assignments.size() == 3);

  double minimumCost = std::numeric_limits<double>::infinity();
  double maximumCost = 0.;
  std::set<std::uint64_t> sequenceIds;
  for (std::size_t worker = 0; worker < assignments.size(); ++worker) {
    auto const& assignment = assignments[worker];
    CHECK(assignment.worker_id == worker);
    minimumCost = std::min(minimumCost, assignment.estimated_cost);
    maximumCost = std::max(maximumCost, assignment.estimated_cost);
    for (auto const& batch : assignment.batches) {
      REQUIRE_FALSE(batch.payloads.empty());
      CHECK(batch.estimated_cost <= 8.);
      for (auto const sequenceId : batch.sequence_ids) {
        CHECK(sequenceIds.insert(sequenceId).second);
      }
    }
  }
  CHECK(sequenceIds.size() == 9);
  // LPT guarantees that the imbalance is no larger than the largest batch here.
  CHECK(maximumCost - minimumCost <= 8.);

  auto repeatedQueue = makeQueue();
  auto repeated =
      planHadronicWorkerAssignments(repeatedQueue, 3, 8., 16);
  REQUIRE(repeated.size() == assignments.size());
  for (std::size_t worker = 0; worker < assignments.size(); ++worker) {
    CHECK(repeated[worker].estimated_cost ==
          Catch::Approx(assignments[worker].estimated_cost));
    REQUIRE(repeated[worker].batches.size() ==
            assignments[worker].batches.size());
    for (std::size_t batch = 0;
         batch < assignments[worker].batches.size(); ++batch) {
      CHECK(repeated[worker].batches[batch].key ==
            assignments[worker].batches[batch].key);
      CHECK(repeated[worker].batches[batch].sequence_ids ==
            assignments[worker].batches[batch].sequence_ids);
      CHECK(repeated[worker].batches[batch].payloads ==
            assignments[worker].batches[batch].payloads);
    }
  }

  CHECK_THROWS_AS(
      planHadronicWorkerAssignments(repeatedQueue, 0, 8., 16),
      std::invalid_argument);
}

TEST_CASE("Hadronic classified micro-batches coalesce once per worker",
          "[HadronicWorkQueue]") {
  HadronicWorkClassifierConfig config;
  auto const nucleon = *classifyHadronicWork(
      Code::Proton, 1_TeV, config);
  auto const pion = *classifyHadronicWork(
      Code::PiPlus, 10_GeV, config);

  ClassifiedHadronicWorkQueue<int> queue;
  queue.enqueue(nucleon, 10, 4., 100);
  queue.enqueue(nucleon, 11, 4., 101);
  queue.enqueue(nucleon, 12, 4., 102);
  queue.enqueue(pion, 20, 1., 200);
  queue.enqueue(pion, 21, 1., 201);
  queue.enqueue(pion, 22, 1., 202);
  queue.enqueue(pion, 23, 1., 203);

  auto assignments =
      planHadronicWorkerAssignments(queue, 2, 4., 8);
  std::vector<std::vector<std::uint64_t>>
      expected_sequence_ids(assignments.size());
  std::vector<std::vector<int>>
      expected_payloads(assignments.size());
  std::vector<std::size_t>
      expected_classified_batches(assignments.size());
  for (auto const& assignment : assignments) {
    expected_classified_batches[assignment.worker_id] =
        assignment.batches.size();
    for (auto const& batch : assignment.batches) {
      expected_sequence_ids[assignment.worker_id].insert(
          expected_sequence_ids[assignment.worker_id].end(),
          batch.sequence_ids.begin(),
          batch.sequence_ids.end());
      expected_payloads[assignment.worker_id].insert(
          expected_payloads[assignment.worker_id].end(),
          batch.payloads.begin(),
          batch.payloads.end());
    }
  }

  auto super_batches =
      coalesceHadronicWorkerAssignments(assignments);
  REQUIRE(super_batches.size() == 2);
  std::set<std::uint64_t> seen;
  for (auto const& batch : super_batches) {
    REQUIRE(batch.worker_id < 2);
    CHECK(
        batch.classified_batches ==
        expected_classified_batches[batch.worker_id]);
    CHECK(
        batch.sequence_ids ==
        expected_sequence_ids[batch.worker_id]);
    CHECK(
        batch.payloads ==
        expected_payloads[batch.worker_id]);
    REQUIRE(
        batch.sequence_ids.size() ==
        batch.payloads.size());
    for (auto const sequence_id : batch.sequence_ids) {
      CHECK(seen.insert(sequence_id).second);
    }
  }
  CHECK(seen.size() == 7);
}

TEST_CASE("Hadronic process protocol is POD, keyed, and fail-fast",
          "[HadronicBatchProtocol]") {
  HadronicBatchHeader header;
  header.request_count = 2;
  header.batch_id = 7;
  CHECK_NOTHROW(validateHadronicBatchHeader(header));

  auto invalidHeader = header;
  invalidHeader.protocol_version += 1;
  CHECK_THROWS_AS(
      validateHadronicBatchHeader(invalidHeader),
      std::invalid_argument);

  HadronicInteractionRequest request;
  request.sequence_id = 19;
  request.random_key =
      HadronicRandomKey{12345, 2, 991, 17, 3, 0};
  request.projectile_pdg = 2212;
  request.target_pdg = 1000080160;
  request.projectile_four_momentum_GeV =
      {100.01, 0., 0., 100.};
  request.target_four_momentum_GeV =
      {14.899, 0., 0., 0.};
  CHECK_NOTHROW(validateHadronicInteractionRequest(request));

  auto const seed = deriveHadronicWorkerSeed(request.random_key);
  CHECK(seed == deriveHadronicWorkerSeed(request.random_key));
  auto changedKey = request.random_key;
  ++changedKey.history_id;
  CHECK(seed != deriveHadronicWorkerSeed(changedKey));

  auto invalidRequest = request;
  invalidRequest.projectile_four_momentum_GeV[2] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(
      validateHadronicInteractionRequest(invalidRequest),
      std::invalid_argument);
}
