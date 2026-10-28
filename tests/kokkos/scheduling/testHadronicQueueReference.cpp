/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/HadronicWorkQueue.hpp>
#include <catch2/catch_all.hpp>
#include <memory>
#include <set>

using namespace corsika;
using namespace corsika::kokkos;

TEST_CASE("Hadronic coalescing preserves move-only particle payloads",
          "[HadronicWorkQueue]") {
  ClassifiedHadronicWorkQueue<std::unique_ptr<int>> queue;
  HadronicWorkClassifierConfig config;
  for (std::uint64_t i = 0; i < 97; ++i) {
    auto const key = *classify_hadronic_work(i % 2 ? Code::PiPlus : Code::Iron,
                                             (i + 1.) * 1_GeV, config);
    queue.enqueue(key, i, 1. + i % 5, std::make_unique<int>(i));
  }
  auto assignments = plan_hadronic_worker_assignments(queue, 4, 9., 3);
  auto batches = coalesce_hadronic_worker_assignments(assignments);
  std::set<std::uint64_t> seen;
  for (auto const& batch : batches) {
    REQUIRE(batch.sequenceIds_.size() == batch.payloads_.size());
    for (std::size_t i = 0; i < batch.sequenceIds_.size(); ++i) {
      REQUIRE(batch.payloads_[i]);
      CHECK(static_cast<std::uint64_t>(*batch.payloads_[i]) == batch.sequenceIds_[i]);
      CHECK(seen.insert(batch.sequenceIds_[i]).second);
    }
  }
  CHECK(seen.size() == 97);
  CHECK(queue.empty());
  CHECK(queue.pendingCost() == 0.);

  std::vector<HadronicWorkerAssignment<int>> invalid(1);
  invalid.front().batches_.push_back({});
  invalid.front().batches_.front().sequenceIds_.push_back(1);
  CHECK_THROWS_AS(coalesce_hadronic_worker_assignments(invalid), std::logic_error);
}

#ifdef CORSIKA_PRODUCTION_WORKER_REFERENCE
#include "ProductionHadronicWorkQueue.hpp"
namespace production = corsika::production_worker_reference;

namespace {
  void check_key(HadronicWorkKey const& a, production::HadronicWorkKey const& b) {
    CHECK(static_cast<int>(a.model_) == static_cast<int>(b.model));
    CHECK(static_cast<int>(a.species_) == static_cast<int>(b.species));
    CHECK(a.energyBin_ == b.energy_bin);
    CHECK(a.massNumberBin_ == b.mass_number_bin);
  }
} // namespace

TEST_CASE("Hadronic classification and fixed plans match pinned production",
          "[HadronicWorkQueue][production]") {
  std::vector<Code> const species = {Code::Proton,   Code::AntiProton, Code::Neutron,
                                     Code::PiPlus,   Code::PiMinus,    Code::Pi0,
                                     Code::KPlus,    Code::KMinus,     Code::Iron,
                                     Code::Electron, Code::Photon,     Code::MuMinus};
  for (int const bins : {1, 2, 8, 64}) {
    HadronicWorkClassifierConfig config;
    production::HadronicWorkClassifierConfig referenceConfig;
    config.energyBinsPerOctave_ = referenceConfig.energy_bins_per_octave = bins;
    for (auto const code : species) {
      for (double const energy : {0., 1.e-12, 1.e-9, 0.1, 1., 40., 79.43282347242813,
                                  79.43282347242814, 80., 1.e3, 1.e8, 1.e20}) {
        auto const actual = classify_hadronic_work(code, energy * 1_GeV, config);
        auto const reference =
            production::classifyHadronicWork(code, energy * 1_GeV, referenceConfig);
        REQUIRE(bool(actual) == bool(reference));
        if (!actual) continue;
        check_key(*actual, *reference);
        CHECK(deterministic_hadronic_interaction_cost(*actual, 0.125) ==
              production::deterministicHadronicInteractionCost(*reference, 0.125));
        CHECK(hadronic_energy_bin_lower_gev(*actual, config) ==
              production::hadronicEnergyBinLowerGeV(*reference, referenceConfig));
        CHECK(hadronic_energy_bin_upper_gev(*actual, config) ==
              production::hadronicEnergyBinUpperGeV(*reference, referenceConfig));
      }
    }
  }

  for (std::size_t const workers : {1, 2, 4, 16}) {
    for (std::size_t const limit : {1, 7, 128}) {
      ClassifiedHadronicWorkQueue<std::uint64_t> queue;
      production::ClassifiedHadronicWorkQueue<std::uint64_t> referenceQueue;
      HadronicWorkClassifierConfig config;
      production::HadronicWorkClassifierConfig referenceConfig;
      for (std::uint64_t i = 0; i < 257; ++i) {
        auto const code = species[i % 9];
        auto const energy = std::exp2(static_cast<int>(i % 21) - 7) * 1_GeV;
        auto const key = *classify_hadronic_work(code, energy, config);
        auto const referenceKey =
            *production::classifyHadronicWork(code, energy, referenceConfig);
        double const cost = 0.125 * (1 + i % 13);
        queue.enqueue(key, 1000 + i, cost, i);
        referenceQueue.enqueue(referenceKey, 1000 + i, cost, i);
      }
      auto actual = plan_hadronic_worker_assignments(queue, workers, 8., limit);
      auto reference =
          production::planHadronicWorkerAssignments(referenceQueue, workers, 8., limit);
      REQUIRE(actual.size() == reference.size());
      for (std::size_t w = 0; w < actual.size(); ++w) {
        CHECK(actual[w].workerId_ == reference[w].worker_id);
        CHECK(actual[w].estimatedCost_ == reference[w].estimated_cost);
        REQUIRE(actual[w].batches_.size() == reference[w].batches.size());
        for (std::size_t b = 0; b < actual[w].batches_.size(); ++b) {
          auto const& a = actual[w].batches_[b];
          auto const& r = reference[w].batches[b];
          check_key(a.key_, r.key);
          CHECK(a.estimatedCost_ == r.estimated_cost);
          CHECK(a.sequenceIds_ == r.sequence_ids);
          CHECK(a.payloads_ == r.payloads);
        }
      }
      auto combined = coalesce_hadronic_worker_assignments(actual);
      auto expected = production::coalesceHadronicWorkerAssignments(reference);
      REQUIRE(combined.size() == expected.size());
      for (std::size_t w = 0; w < combined.size(); ++w) {
        CHECK(combined[w].workerId_ == expected[w].worker_id);
        CHECK(combined[w].estimatedCost_ == expected[w].estimated_cost);
        CHECK(combined[w].classifiedBatches_ == expected[w].classified_batches);
        CHECK(combined[w].sequenceIds_ == expected[w].sequence_ids);
        CHECK(combined[w].payloads_ == expected[w].payloads);
      }
      CHECK(queue.empty());
      CHECK(queue.pendingCost() == referenceQueue.pendingCost());
    }
  }
}
#endif
