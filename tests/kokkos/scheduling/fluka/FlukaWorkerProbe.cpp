/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/scheduling/FlukaWorker.hpp>
#include <corsika/kokkos/scheduling/HadronicProcessPool.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include "FlukaWorkerFixture.hpp"
#include "FlukaProductionFixture.hpp"
#ifdef CORSIKA_PRODUCTION_FLUKA_WORKER_REFERENCE
#include "ProductionFlukaWorker.hpp"
#endif
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

namespace {
  using namespace corsika;
  using namespace corsika::kokkos;

  void require(bool value, char const* message) {
    if (!value) throw std::runtime_error(message);
  }

  auto make_requests(std::set<Code> const& targets) {
    std::vector<HadronicInteractionRequest> result;
    for (std::uint64_t seed : {17, 103, 2026}) {
      for (Code target : targets) {
        for (Code projectile : {Code::Proton, Code::PiPlus, Code::PiMinus, Code::KPlus}) {
          for (double momentum : {1., 20., 80.}) {
            for (bool tilted : {false, true}) {
              HadronicInteractionRequest request;
              request.sequenceId_ = result.size();
              request.randomKey_ = {seed, 7, 1000 + result.size(), 3, 1, 0};
              request.projectilePdg_ = static_cast<std::int32_t>(get_PDG(projectile));
              request.targetPdg_ = static_cast<std::int32_t>(get_PDG(target));
              request.projectileFourMomentumGeV_ = {
                  calculate_total_energy(momentum * 1_GeV, get_mass(projectile)) / 1_GeV,
                  tilted ? 0.6 * momentum : 0., 0., tilted ? 0.8 * momentum : momentum};
              request.targetFourMomentumGeV_ = {get_mass(target) / 1_GeV, 0., 0., 0.};
              result.push_back(request);
            }
          }
        }
      }
    }
    return result;
  }

  FlukaWorkerResult execute(corsika::fluka::InteractionModel& model,
                            HadronicInteractionRequest const& request,
                            CoordinateSystemPtr const& frame, bool reference) {
    if (!reference) return execute_fluka_request(model, request, frame);
#ifdef CORSIKA_PRODUCTION_FLUKA_WORKER_REFERENCE
    // Protocol ABI has an independent field/offset test in the contract slice.
    using OriginalRequest = production_fluka_worker_reference::HadronicInteractionRequest;
    static_assert(sizeof(OriginalRequest) == sizeof(request));
    static_assert(std::is_trivially_copyable_v<OriginalRequest>);
    OriginalRequest original;
    std::memcpy(static_cast<void*>(&original), &request, sizeof(original));
    auto raw = production_fluka_worker_reference::executeRequest(model, original, frame);
    FlukaWorkerResult result;
    static_assert(sizeof(result.header_) == sizeof(raw.header));
    std::memcpy(static_cast<void*>(&result.header_), &raw.header, sizeof(result.header_));
    result.secondaries_.resize(raw.secondaries.size());
    static_assert(sizeof(HadronicSecondaryRecord) == sizeof(raw.secondaries.front()));
    if (!raw.secondaries.empty()) {
      std::memcpy(static_cast<void*>(result.secondaries_.data()), raw.secondaries.data(),
                  raw.secondaries.size() * sizeof(HadronicSecondaryRecord));
    }
    return result;
#else
    throw std::invalid_argument("Production reference is disabled");
#endif
  }

  template <typename TResult>
  void write_result(std::ostream& output, TResult const& result, std::uint64_t sequence) {
    auto const& header = result.header_;
    require(header.sequenceId_ == sequence, "Response order changed");
    require(header.status_ == HadronicWorkerStatus::Success, "Reaction failed");
    require(header.secondaryCount_ == result.secondaries_.size(), "Wrong particle count");
    require(std::isfinite(header.finalStateTimeMs_) && header.finalStateTimeMs_ >= 0.,
            "Invalid timer");
    output << "event " << sequence << ' ' << header.protocolVersion_ << ' '
           << header.secondaryCount_ << ' ' << header.randomValuesSupplied_ << '\n';
    for (auto const& particle : result.secondaries_) {
      output << particle.pdg_ << ' ' << particle.kineticEnergyGeV_;
      for (double component : particle.direction_) output << ' ' << component;
      output << '\n';
    }
  }

  void check_errors(corsika::fluka::InteractionModel& model,
                    HadronicInteractionRequest request, CoordinateSystemPtr const& frame,
                    bool reference) {
    auto original = request;
    request.model_ = HadronicWorkerModel::Sibyll;
    require(execute(model, request, frame, reference).header_.status_ ==
                HadronicWorkerStatus::InvalidRequest,
            "Wrong model accepted");
    request = original;
    request.projectilePdg_ = 11;
    require(execute(model, request, frame, reference).header_.status_ ==
                HadronicWorkerStatus::UnsupportedProjectile,
            "Unsupported projectile accepted");
    request = original;
    request.targetPdg_ = static_cast<std::int32_t>(get_PDG(Code::Uranium));
    require(execute(model, request, frame, reference).header_.status_ ==
                HadronicWorkerStatus::UnsupportedTarget,
            "Uninitialized target accepted");
    request = original;
    request.projectileFourMomentumGeV_[0] = -1.;
    require(execute(model, request, frame, reference).header_.status_ ==
                HadronicWorkerStatus::GeneratorFailure,
            "Invalid four-momentum accepted");
  }
} // namespace

int main(int argc, char** argv) {
  try {
    // Test-only configuration inherited by each exec'ed worker, not a runtime API.
    char const* material = std::getenv("CORSIKA_TEST_FLUKA_MATERIAL");
    require(material != nullptr, "Missing material fixture");
    auto environment = corsika::fluka_worker_test::make_environment(material);
    auto const targets = get_all_elements_in_universe(environment);
    auto const& frame = environment.getCoordinateSystem();
    auto& rng = RNGManager<>::getInstance();
    rng.registerRandomStream("fluka");
    char const* initialSeed = std::getenv("CORSIKA_TEST_FLUKA_INITIAL_SEED");
    rng.setSeed(initialSeed ? std::stoull(initialSeed) : 2026);

    if (argc == 5 && std::string{argv[1]} == "--server-fd" &&
        std::string{argv[3]} == "--worker-id") {
      corsika::fluka::InteractionModel model{targets};
#ifdef CORSIKA_PRODUCTION_FLUKA_WORKER_REFERENCE
      if (std::getenv("CORSIKA_TEST_ORIGINAL_FLUKA_WORKER")) {
        return production_fluka_worker_reference::serve_original_worker(
            model, frame, std::stoi(argv[2]), std::stoul(argv[4]));
      }
#endif
      return serve_fluka_worker(model, frame, std::stoi(argv[2]), std::stoul(argv[4]));
    }
    require(argc == 3, "Expected mode and output path");
    std::string const mode{argv[1]};
    bool const selfTest = mode.rfind("self-", 0) == 0;
    auto const requests =
        selfTest ? corsika::kokkos::fluka_worker_test::make_self_test_requests(96)
                 : make_requests(targets);
    std::ofstream output{argv[2]};
    require(bool(output), "Cannot open audit output");
    output << std::hexfloat;

    std::size_t completedRequests = requests.size();
    if (mode == "self-adjacent" || mode == "self-reference-adjacent") {
      corsika::fluka::InteractionModel model{targets};
      for (auto const& request : requests) {
        auto const first =
            execute(model, request, frame, mode == "self-reference-adjacent");
        auto const repeated =
            execute(model, request, frame, mode == "self-reference-adjacent");
        require(first.header_.status_ == HadronicWorkerStatus::Success &&
                    corsika::kokkos::fluka_worker_test::equal_self_test_result(first,
                                                                               repeated),
                "Production adjacent-repeat self-test changed");
        write_result(output, first, request.sequenceId_);
        write_result(output, repeated, request.sequenceId_);
      }
    } else if (mode.rfind("self-pool", 0) == 0) {
      auto const workers = std::stoul(mode.substr(9));
      require(workers == 1 || workers == 2 || workers == 4,
              "Invalid self-test worker count");
      auto const batches =
          corsika::kokkos::fluka_worker_test::make_self_test_batches(requests, workers);
      HadronicProcessPool pool{std::filesystem::absolute(argv[0]), workers};
      auto const first = pool.execute(batches);
      auto const repeated = pool.execute(batches);
      require(first.size() == repeated.size(),
              "Production self-test batch count changed");
      for (std::size_t i = 0; i < first.size(); ++i) {
        require(first[i].batchId_ == repeated[i].batchId_ &&
                    first[i].responses_.size() == repeated[i].responses_.size(),
                "Production self-test batch identity changed");
        for (std::size_t j = 0; j < first[i].responses_.size(); ++j) {
          require(corsika::kokkos::fluka_worker_test::equal_self_test_response(
                      first[i].responses_[j], repeated[i].responses_[j]),
                  "Production persistent-pool repeated self-test changed");
        }
      }
      for (auto const* pass : {&first, &repeated}) {
        for (std::size_t i = 0; i < pass->size(); ++i) {
          for (std::size_t j = 0; j < (*pass)[i].responses_.size(); ++j) {
            write_result(output, (*pass)[i].responses_[j],
                         batches[i].requests_[j].sequenceId_);
          }
        }
      }
    } else if (mode == "direct" || mode == "reference" ||
               mode.rfind("direct-history:", 0) == 0) {
      std::size_t workers = 1, worker = 0;
      if (mode.rfind("direct-history:", 0) == 0) {
        auto const split = mode.find(':', 15);
        require(split != std::string::npos, "Missing worker history id");
        workers = std::stoul(mode.substr(15, split - 15));
        worker = std::stoul(mode.substr(split + 1));
        require((workers == 1 || workers == 2 || workers == 4) && worker < workers,
                "Invalid direct worker history");
      }
      completedRequests = 0;
      for (std::size_t i = 0; i < requests.size(); ++i) {
        if ((i / 13) % workers == worker) ++completedRequests;
      }
      corsika::fluka::InteractionModel model{targets};
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i < requests.size(); ++i) {
          if ((i / 13) % workers != worker) continue;
          auto const& request = requests[i];
          auto const result = execute(model, request, frame, mode == "reference");
          write_result(output, result, request.sequenceId_);
        }
      }
      check_errors(model, requests.front(), frame, mode == "reference");
    } else {
      require(mode == "pool1" || mode == "pool2" || mode == "pool4", "Unknown mode");
      auto const workers = std::stoul(mode.substr(4));
      // No FLUKA model is constructed in the parent before fork/exec.
      HadronicProcessPool pool{std::filesystem::absolute(argv[0]), workers};
      std::vector<HadronicRequestBatch> batches;
      for (std::size_t offset = 0; offset < requests.size(); offset += 13) {
        HadronicRequestBatch batch;
        batch.batchId_ = 0xC08A0000ULL + batches.size();
        batch.workerId_ = batches.size() % workers;
        auto const end = std::min(offset + 13, requests.size());
        batch.requests_.assign(requests.begin() + offset, requests.begin() + end);
        batches.push_back(std::move(batch));
      }
      for (int pass = 0; pass < 2; ++pass) {
        auto const results = pool.execute(batches);
        require(results.size() == batches.size(), "Missing completed batch");
        for (std::size_t i = 0; i < results.size(); ++i) {
          require(results[i].batchId_ == batches[i].batchId_ &&
                      results[i].workerId_ == batches[i].workerId_,
                  "Batch identity changed");
          require(results[i].responses_.size() == batches[i].requests_.size(),
                  "Missing request response");
          for (std::size_t j = 0; j < results[i].responses_.size(); ++j) {
            write_result(output, results[i].responses_[j],
                         batches[i].requests_[j].sequenceId_);
          }
        }
      }
    }
    output << "complete " << completedRequests << " 2\n";
    output.flush();
    require(bool(output), "Incomplete audit output");
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
