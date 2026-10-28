/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/HadronicProcessPool.hpp>
#include <catch2/catch_all.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <tuple>
#include <sys/wait.h>

#ifdef CORSIKA_PRODUCTION_WORKER_REFERENCE
#include "ProductionHadronicProcessPool.hpp"
namespace production = corsika::production_worker_reference;
#endif

using namespace corsika::kokkos;

namespace {
  class WorkerFixture {
  public:
    WorkerFixture() {
      auto path =
          (std::filesystem::temp_directory_path() / "cp8-ipc-test-XXXXXX").string();
      auto const result = ::mkdtemp(path.data());
      if (!result) throw std::runtime_error("Cannot create IPC fixture directory");
      directory_ = result;
    }
    ~WorkerFixture() { std::filesystem::remove_all(directory_); }
    std::filesystem::path executable(std::string const& mode) const {
      auto const path = directory_ / mode;
      std::filesystem::create_symlink(HADRONIC_PROTOCOL_TEST_WORKER, path);
      return path;
    }

  private:
    std::filesystem::path directory_;
  };

  std::set<std::filesystem::path> runtime_directories() {
    std::set<std::filesystem::path> result;
    auto const prefix = "corsika8-hadronic-pool-" + std::to_string(::getpid()) + "-";
    for (auto const& item :
         std::filesystem::directory_iterator(std::filesystem::temp_directory_path())) {
      if (item.path().filename().string().find(prefix) == 0) result.insert(item.path());
    }
    return result;
  }

  std::vector<HadronicRequestBatch> requests(std::size_t workers, bool explicitWorkers) {
    std::vector<HadronicRequestBatch> result;
    for (std::size_t b = 0; b < 23; ++b) {
      HadronicRequestBatch batch;
      batch.batchId_ = 700 + b;
      if (explicitWorkers) batch.workerId_ = (b * 3 + 1) % workers;
      for (std::size_t r = 0; r < 1 + b % 7; ++r) {
        HadronicInteractionRequest request;
        request.sequenceId_ = 100 * b + r;
        request.randomKey_ = {991, 3, request.sequenceId_, 7, 1, 0};
        request.projectilePdg_ = 2212;
        request.targetPdg_ = 1000080160;
        request.projectileFourMomentumGeV_ = {100. + b, 0., 0., 99.};
        request.targetFourMomentumGeV_ = {15., 0., 0., 0.};
        batch.requests_.push_back(request);
      }
      result.push_back(std::move(batch));
    }
    return result;
  }

  using row_type = std::tuple<std::uint64_t, std::size_t, std::uint64_t, std::uint32_t,
                              std::int32_t, double, std::array<double, 3>>;
  std::vector<row_type> flatten(std::vector<HadronicCompletedBatch> const& batches) {
    std::vector<row_type> rows;
    for (auto const& batch : batches)
      for (auto const& response : batch.responses_) {
        auto const& h = response.header_;
        CHECK(h.status_ == HadronicWorkerStatus::Success);
        CHECK(h.secondaryCount_ == response.secondaries_.size());
        rows.emplace_back(batch.batchId_, batch.workerId_, h.sequenceId_,
                          h.randomValuesSupplied_, -1, h.finalStateTimeMs_,
                          std::array<double, 3>{});
        for (auto const& particle : response.secondaries_)
          rows.emplace_back(batch.batchId_, batch.workerId_, h.sequenceId_,
                            particle.reserved_, particle.pdg_, particle.kineticEnergyGeV_,
                            particle.direction_);
      }
    return rows;
  }

#ifdef CORSIKA_PRODUCTION_WORKER_REFERENCE
  std::vector<production::HadronicRequestBatch> convert(
      std::vector<HadronicRequestBatch> const& batches) {
    std::vector<production::HadronicRequestBatch> result;
    for (auto const& batch : batches) {
      production::HadronicRequestBatch out;
      out.batch_id = batch.batchId_;
      out.worker_id = batch.workerId_;
      for (auto const& request : batch.requests_) {
        production::HadronicInteractionRequest r;
        static_assert(sizeof(r) == sizeof(request));
        r.protocol_version = request.protocolVersion_;
        r.model = static_cast<production::HadronicWorkerModel>(request.model_);
        r.reserved = request.reserved_;
        r.sequence_id = request.sequenceId_;
        auto const& key = request.randomKey_;
        r.random_key = {key.seed_,   key.showerId_,  key.historyId_,
                        key.stepId_, key.processId_, key.drawDomain_};
        r.projectile_pdg = request.projectilePdg_;
        r.target_pdg = request.targetPdg_;
        r.projectile_four_momentum_GeV = request.projectileFourMomentumGeV_;
        r.target_four_momentum_GeV = request.targetFourMomentumGeV_;
        out.requests.push_back(r);
      }
      result.push_back(std::move(out));
    }
    return result;
  }
  std::vector<row_type> flatten(
      std::vector<production::HadronicCompletedBatch> const& batches) {
    std::vector<row_type> rows;
    for (auto const& batch : batches)
      for (auto const& response : batch.responses) {
        auto const& h = response.header;
        CHECK(h.status == production::HadronicWorkerStatus::Success);
        CHECK(h.secondary_count == response.secondaries.size());
        rows.emplace_back(batch.batch_id, batch.worker_id, h.sequence_id,
                          h.random_values_supplied, -1, h.final_state_time_ms,
                          std::array<double, 3>{});
        for (auto const& particle : response.secondaries)
          rows.emplace_back(batch.batch_id, batch.worker_id, h.sequence_id,
                            particle.reserved, particle.pdg, particle.kinetic_energy_GeV,
                            particle.direction);
      }
    return rows;
  }
#endif
} // namespace

TEST_CASE("Hadronic pool preserves fixed workers and persistent request order",
          "[HadronicProcessPool]") {
  WorkerFixture fixture;
  auto const worker = fixture.executable("normal");
  auto const before = runtime_directories();
  for (std::size_t const count : {1, 2, 4}) {
    for (bool const assigned : {false, true}) {
      auto const input = requests(count, assigned);
      std::vector<row_type> actual;
      HadronicProcessPoolStatistics stats;
      {
        HadronicProcessPool pool(worker, count);
        CHECK(pool.workerCount() == count);
        CHECK(pool.execute({}).empty());
        std::vector<std::size_t> ordinals(count);
        for (int repeat = 0; repeat < 2; ++repeat) {
          auto const output = pool.execute(input);
          REQUIRE(output.size() == input.size());
          for (std::size_t b = 0; b < input.size(); ++b) {
            auto const w = assigned ? input[b].workerId_ : b % count;
            CHECK(output[b].batchId_ == input[b].batchId_);
            CHECK(output[b].workerId_ == w);
            REQUIRE(output[b].responses_.size() == input[b].requests_.size());
            for (std::size_t r = 0; r < input[b].requests_.size(); ++r) {
              auto const& response = output[b].responses_[r];
              auto const& request = input[b].requests_[r];
              ++ordinals[w];
              CHECK(response.header_.sequenceId_ == request.sequenceId_);
              REQUIRE(response.secondaries_.size() == request.sequenceId_ % 4);
              for (auto const& particle : response.secondaries_) {
                CHECK(particle.direction_[0] == w);
                CHECK(particle.direction_[1] == ordinals[w]);
              }
            }
          }
          auto const rows = flatten(output);
          actual.insert(actual.end(), rows.begin(), rows.end());
        }
        stats = pool.statistics();
        CHECK(std::isfinite(stats.executeWallTimeMs_));
        CHECK(stats.executeWallTimeMs_ >= 0.);
        CHECK(stats.batches_ == input.size() * 2);
        CHECK(stats.bulkResponseBatches_ == stats.batches_);
      }
      CHECK(runtime_directories() == before);
#ifdef CORSIKA_PRODUCTION_WORKER_REFERENCE
      std::vector<row_type> expected;
      {
        production::HadronicProcessPool pool(worker, count);
        for (int repeat = 0; repeat < 2; ++repeat) {
          auto const rows = flatten(pool.execute(convert(input)));
          expected.insert(expected.end(), rows.begin(), rows.end());
        }
        auto const& reference = pool.statistics();
        CHECK(stats.requests_ == reference.requests);
        CHECK(stats.secondaries_ == reference.secondaries);
        CHECK(stats.bytesSent_ == reference.bytes_sent);
        CHECK(stats.bytesReceived_ == reference.bytes_received);
        CHECK(stats.bulkResponsePayloadReads_ == reference.bulk_response_payload_reads);
      }
      CHECK(actual == expected);
      CHECK(runtime_directories() == before);
#endif
    }
  }
  int status = 0;
  CHECK(::waitpid(-1, &status, WNOHANG) == -1);
  CHECK(errno == ECHILD);
}

TEST_CASE("Hadronic pool rejects bad responses and cleans up failed workers",
          "[HadronicProcessPool]") {
  WorkerFixture fixture;
  auto const before = runtime_directories();
  auto input = requests(2, false);
  input.resize(1);
  for (std::string const mode : {"batch", "count", "magic", "endian", "version", "order",
                                 "failure", "overflow", "truncated", "disconnect"}) {
    CAPTURE(mode);
    auto const worker = fixture.executable(mode);
    std::string message;
    try {
      HadronicProcessPool pool(worker, 2);
      pool.execute(input);
    } catch (std::exception const& error) { message = error.what(); }
    REQUIRE_FALSE(message.empty());
    CHECK(runtime_directories() == before);
#ifdef CORSIKA_PRODUCTION_WORKER_REFERENCE
    std::string reference;
    try {
      production::HadronicProcessPool pool(worker, 2);
      pool.execute(convert(input));
    } catch (std::exception const& error) { reference = error.what(); }
    CHECK(message == reference);
    CHECK(runtime_directories() == before);
#endif
  }
  int status = 0;
  CHECK(::waitpid(-1, &status, WNOHANG) == -1);
  CHECK(errno == ECHILD);
}

TEST_CASE("Hadronic pool validates complete requests before dispatch",
          "[HadronicProcessPool]") {
  WorkerFixture fixture;
  auto const worker = fixture.executable("normal");
  CHECK_THROWS_AS(HadronicProcessPool(worker, 0), std::invalid_argument);
  CHECK_THROWS_AS(HadronicProcessPool(worker / "absent", 1), std::invalid_argument);
  HadronicProcessPool pool(worker, 2);
  for (int scenario = 0; scenario < 4; ++scenario) {
    auto input = requests(2, false);
    if (scenario == 0) input.back().requests_.clear();
    if (scenario == 1) input.back().workerId_ = 2;
    if (scenario == 2) input.back().requests_.back().projectilePdg_ = 0;
    if (scenario == 3)
      input.back().requests_.back().projectileFourMomentumGeV_[0] =
          std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(pool.execute(input), std::invalid_argument);
    CHECK(pool.statistics().batches_ == 0);
    CHECK(pool.statistics().bytesSent_ == 0);
  }
  CHECK(pool.execute(requests(2, false)).size() == 23);
}
