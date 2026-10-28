/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#ifdef CORSIKA_PRODUCTION_HADRONIC_REFERENCE
#include "ProductionHadronicBatchProtocol.hpp"
namespace reference = corsika::production_hadronic_reference;
#endif
namespace cp = corsika::kokkos;

TEST_CASE("Hadronic protocol validates only the production contract",
          "[hadronic_protocol]") {
  CHECK(cp::HadronicBatchMagic == 0x43384842U);
  CHECK(cp::HadronicBatchProtocolVersion == 3U);
  CHECK(cp::HadronicBatchEndianMarker == 0x01020304U);
  cp::HadronicBatchHeader header{};
  CHECK_NOTHROW(cp::validate_hadronic_batch_header(header));
  auto wrong = header;
  wrong.magic_ ^= 1;
  CHECK_THROWS_AS(cp::validate_hadronic_batch_header(wrong), std::invalid_argument);
  wrong = header;
  wrong.protocolVersion_ = 2;
  CHECK_THROWS_AS(cp::validate_hadronic_batch_header(wrong), std::invalid_argument);
  wrong = header;
  wrong.endianMarker_ = 0x04030201U;
  CHECK_THROWS_AS(cp::validate_hadronic_batch_header(wrong), std::invalid_argument);
  cp::HadronicInteractionRequest request{};
  request.projectilePdg_ = 2212;
  request.targetPdg_ = 1000080160;
  request.projectileFourMomentumGeV_ = {100., 0., 0., 99.};
  request.targetFourMomentumGeV_ = {16., 0., 0., 0.};
  CHECK_NOTHROW(cp::validate_hadronic_interaction_request(request));
  auto check_invalid = [](auto const& item) {
    CHECK_THROWS_AS(cp::validate_hadronic_interaction_request(item),
                    std::invalid_argument);
#ifdef CORSIKA_PRODUCTION_HADRONIC_REFERENCE
    reference::HadronicInteractionRequest old{};
    static_assert(sizeof(old) == sizeof(item));
    std::memcpy(&old, &item, sizeof(old));
    CHECK_THROWS_AS(reference::validateHadronicInteractionRequest(old),
                    std::invalid_argument);
#endif
  };
  auto bad = request;
  bad.protocolVersion_ = 2;
  check_invalid(bad);
  bad = request;
  bad.projectilePdg_ = 0;
  check_invalid(bad);
  bad = request;
  bad.targetPdg_ = 0;
  check_invalid(bad);
  for (int side = 0; side < 2; ++side) {
    for (int axis = 0; axis < 4; ++axis) {
      for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
        bad = request;
        (side == 0 ? bad.projectileFourMomentumGeV_ : bad.targetFourMomentumGeV_)[axis] =
            invalid;
        check_invalid(bad);
      }
    }
    for (double energy : {0., -1.}) {
      bad = request;
      (side == 0 ? bad.projectileFourMomentumGeV_ : bad.targetFourMomentumGeV_)[0] =
          energy;
      check_invalid(bad);
    }
  }
  // The wire validator does not select a model or enforce a mass shell.
  // Those responsibilities remain with the caller and the generator.
  request.model_ = static_cast<cp::HadronicWorkerModel>(255);
  request.projectileFourMomentumGeV_ = {1., 2., 3., 4.};
  CHECK_NOTHROW(cp::validate_hadronic_interaction_request(request));
}

TEST_CASE("Hadronic history seeds are independent of worker and dispatch order",
          "[hadronic_protocol]") {
  std::vector<cp::HadronicRandomKey> keys;
  std::vector<std::uint64_t> seeds;
  std::mt19937_64 generator(197);
  for (unsigned i = 0; i < 4096; ++i) {
    cp::HadronicRandomKey key{generator(),
                              generator(),
                              generator(),
                              generator(),
                              static_cast<std::uint32_t>(generator()),
                              static_cast<std::uint32_t>(generator())};
    keys.push_back(key);
    seeds.push_back(cp::derive_hadronic_worker_seed(key));
#ifdef CORSIKA_PRODUCTION_HADRONIC_REFERENCE
    reference::HadronicRandomKey old{};
    std::memcpy(&old, &key, sizeof(old));
    CHECK(seeds.back() == reference::deriveHadronicWorkerSeed(old));
#endif
  }
  std::vector<unsigned> order(keys.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), generator);
  for (unsigned chunk : {1U, 16U, 257U, 4096U}) {
    for (std::size_t begin = 0; begin < order.size(); begin += chunk) {
      for (std::size_t i = begin; i < std::min(begin + chunk, order.size()); ++i) {
        auto index = order[i];
        CHECK(cp::derive_hadronic_worker_seed(keys[index]) == seeds[index]);
      }
    }
  }
}

#ifdef CORSIKA_PRODUCTION_HADRONIC_REFERENCE
static_assert(sizeof(cp::HadronicRandomKey) == sizeof(reference::HadronicRandomKey));
static_assert(alignof(cp::HadronicRandomKey) == alignof(reference::HadronicRandomKey));
static_assert(offsetof(cp::HadronicRandomKey, seed_) ==
              offsetof(reference::HadronicRandomKey, seed));
static_assert(offsetof(cp::HadronicRandomKey, showerId_) ==
              offsetof(reference::HadronicRandomKey, shower_id));
static_assert(offsetof(cp::HadronicRandomKey, historyId_) ==
              offsetof(reference::HadronicRandomKey, history_id));
static_assert(offsetof(cp::HadronicRandomKey, stepId_) ==
              offsetof(reference::HadronicRandomKey, step_id));
static_assert(offsetof(cp::HadronicRandomKey, processId_) ==
              offsetof(reference::HadronicRandomKey, process_id));
static_assert(offsetof(cp::HadronicRandomKey, drawDomain_) ==
              offsetof(reference::HadronicRandomKey, draw_domain));
static_assert(sizeof(cp::HadronicBatchHeader) == sizeof(reference::HadronicBatchHeader));
static_assert(alignof(cp::HadronicBatchHeader) ==
              alignof(reference::HadronicBatchHeader));
static_assert(offsetof(cp::HadronicBatchHeader, magic_) ==
              offsetof(reference::HadronicBatchHeader, magic));
static_assert(offsetof(cp::HadronicBatchHeader, protocolVersion_) ==
              offsetof(reference::HadronicBatchHeader, protocol_version));
static_assert(offsetof(cp::HadronicBatchHeader, endianMarker_) ==
              offsetof(reference::HadronicBatchHeader, endian_marker));
static_assert(offsetof(cp::HadronicBatchHeader, requestCount_) ==
              offsetof(reference::HadronicBatchHeader, request_count));
static_assert(offsetof(cp::HadronicBatchHeader, batchId_) ==
              offsetof(reference::HadronicBatchHeader, batch_id));
static_assert(sizeof(cp::HadronicInteractionRequest) ==
              sizeof(reference::HadronicInteractionRequest));
static_assert(alignof(cp::HadronicInteractionRequest) ==
              alignof(reference::HadronicInteractionRequest));
static_assert(offsetof(cp::HadronicInteractionRequest, protocolVersion_) ==
              offsetof(reference::HadronicInteractionRequest, protocol_version));
static_assert(offsetof(cp::HadronicInteractionRequest, model_) ==
              offsetof(reference::HadronicInteractionRequest, model));
static_assert(offsetof(cp::HadronicInteractionRequest, reserved_) ==
              offsetof(reference::HadronicInteractionRequest, reserved));
static_assert(offsetof(cp::HadronicInteractionRequest, sequenceId_) ==
              offsetof(reference::HadronicInteractionRequest, sequence_id));
static_assert(offsetof(cp::HadronicInteractionRequest, randomKey_) ==
              offsetof(reference::HadronicInteractionRequest, random_key));
static_assert(offsetof(cp::HadronicInteractionRequest, projectilePdg_) ==
              offsetof(reference::HadronicInteractionRequest, projectile_pdg));
static_assert(offsetof(cp::HadronicInteractionRequest, targetPdg_) ==
              offsetof(reference::HadronicInteractionRequest, target_pdg));
static_assert(offsetof(cp::HadronicInteractionRequest, projectileFourMomentumGeV_) ==
              offsetof(reference::HadronicInteractionRequest,
                       projectile_four_momentum_GeV));
static_assert(offsetof(cp::HadronicInteractionRequest, targetFourMomentumGeV_) ==
              offsetof(reference::HadronicInteractionRequest, target_four_momentum_GeV));
static_assert(sizeof(cp::HadronicInteractionResultHeader) ==
              sizeof(reference::HadronicInteractionResultHeader));
static_assert(alignof(cp::HadronicInteractionResultHeader) ==
              alignof(reference::HadronicInteractionResultHeader));
static_assert(offsetof(cp::HadronicInteractionResultHeader, protocolVersion_) ==
              offsetof(reference::HadronicInteractionResultHeader, protocol_version));
static_assert(offsetof(cp::HadronicInteractionResultHeader, status_) ==
              offsetof(reference::HadronicInteractionResultHeader, status));
static_assert(offsetof(cp::HadronicInteractionResultHeader, sequenceId_) ==
              offsetof(reference::HadronicInteractionResultHeader, sequence_id));
static_assert(offsetof(cp::HadronicInteractionResultHeader, secondaryCount_) ==
              offsetof(reference::HadronicInteractionResultHeader, secondary_count));
static_assert(offsetof(cp::HadronicInteractionResultHeader, randomValuesSupplied_) ==
              offsetof(reference::HadronicInteractionResultHeader,
                       random_values_supplied));
static_assert(offsetof(cp::HadronicInteractionResultHeader, finalStateTimeMs_) ==
              offsetof(reference::HadronicInteractionResultHeader, final_state_time_ms));
static_assert(sizeof(cp::HadronicSecondaryRecord) ==
              sizeof(reference::HadronicSecondaryRecord));
static_assert(alignof(cp::HadronicSecondaryRecord) ==
              alignof(reference::HadronicSecondaryRecord));
static_assert(offsetof(cp::HadronicSecondaryRecord, pdg_) ==
              offsetof(reference::HadronicSecondaryRecord, pdg));
static_assert(offsetof(cp::HadronicSecondaryRecord, reserved_) ==
              offsetof(reference::HadronicSecondaryRecord, reserved));
static_assert(offsetof(cp::HadronicSecondaryRecord, kineticEnergyGeV_) ==
              offsetof(reference::HadronicSecondaryRecord, kinetic_energy_GeV));
static_assert(offsetof(cp::HadronicSecondaryRecord, direction_) ==
              offsetof(reference::HadronicSecondaryRecord, direction));
static_assert(static_cast<unsigned>(cp::HadronicWorkerModel::Fluka) ==
              static_cast<unsigned>(reference::HadronicWorkerModel::Fluka));
static_assert(static_cast<unsigned>(cp::HadronicWorkerModel::Sibyll) ==
              static_cast<unsigned>(reference::HadronicWorkerModel::Sibyll));
static_assert(static_cast<unsigned>(cp::HadronicWorkerModel::QgsJetII) ==
              static_cast<unsigned>(reference::HadronicWorkerModel::QgsJetII));
static_assert(static_cast<unsigned>(cp::HadronicWorkerModel::EposLhc) ==
              static_cast<unsigned>(reference::HadronicWorkerModel::EposLhc));
static_assert(static_cast<unsigned>(cp::HadronicWorkerModel::Pythia8) ==
              static_cast<unsigned>(reference::HadronicWorkerModel::Pythia8));
static_assert(static_cast<unsigned>(cp::HadronicWorkerStatus::Success) ==
              static_cast<unsigned>(reference::HadronicWorkerStatus::Success));
static_assert(static_cast<unsigned>(cp::HadronicWorkerStatus::InvalidProtocol) ==
              static_cast<unsigned>(reference::HadronicWorkerStatus::InvalidProtocol));
static_assert(static_cast<unsigned>(cp::HadronicWorkerStatus::InvalidRequest) ==
              static_cast<unsigned>(reference::HadronicWorkerStatus::InvalidRequest));
static_assert(
    static_cast<unsigned>(cp::HadronicWorkerStatus::UnsupportedProjectile) ==
    static_cast<unsigned>(reference::HadronicWorkerStatus::UnsupportedProjectile));
static_assert(static_cast<unsigned>(cp::HadronicWorkerStatus::UnsupportedTarget) ==
              static_cast<unsigned>(reference::HadronicWorkerStatus::UnsupportedTarget));
static_assert(static_cast<unsigned>(cp::HadronicWorkerStatus::GeneratorFailure) ==
              static_cast<unsigned>(reference::HadronicWorkerStatus::GeneratorFailure));
static_assert(static_cast<unsigned>(cp::HadronicWorkerStatus::WorkerFailure) ==
              static_cast<unsigned>(reference::HadronicWorkerStatus::WorkerFailure));
#endif
