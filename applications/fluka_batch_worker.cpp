/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/core/HadronicBatchProtocol.hpp>
#include <corsika/framework/core/HadronicProcessPool.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/modules/FLUKA.hpp>

#include <FLUKA.hpp>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__linux__)
#include <cerrno>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

  using namespace corsika;

  struct EncodedResult {
    HadronicInteractionResultHeader header{};
    std::vector<HadronicSecondaryRecord> secondaries;
  };

  template <typename TValue>
  TValue readRecord(std::istream& input, char const* description) {
    TValue value{};
    input.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(value)));
    if (!input) {
      throw std::runtime_error(
          std::string{"Could not read "} + description);
    }
    return value;
  }

  template <typename TValue>
  void writeRecord(
      std::ostream& output, TValue const& value,
      char const* description) {
    output.write(
        reinterpret_cast<char const*>(&value),
        static_cast<std::streamsize>(sizeof(value)));
    if (!output) {
      throw std::runtime_error(
          std::string{"Could not write "} + description);
    }
  }

#if defined(__unix__) || defined(__linux__)
  bool readAllFromDescriptor(
      int const descriptor, void* destination,
      std::size_t const bytes, bool const allowCleanEof) {
    auto* cursor = static_cast<unsigned char*>(destination);
    std::size_t remaining = bytes;
    while (remaining > 0) {
      auto const received =
          ::recv(descriptor, cursor, remaining, 0);
      if (received < 0 && errno == EINTR) {
        continue;
      }
      if (received == 0 && allowCleanEof &&
          remaining == bytes) {
        return false;
      }
      if (received <= 0) {
        throw std::runtime_error(
            "Could not read complete record from persistent worker socket");
      }
      cursor += static_cast<std::size_t>(received);
      remaining -= static_cast<std::size_t>(received);
    }
    return true;
  }

  void writeAllToDescriptor(
      int const descriptor, void const* source,
      std::size_t const bytes) {
    auto const* cursor =
        static_cast<unsigned char const*>(source);
    std::size_t remaining = bytes;
    while (remaining > 0) {
      auto const sent =
          ::send(
              descriptor, cursor, remaining,
              MSG_NOSIGNAL);
      if (sent < 0 && errno == EINTR) {
        continue;
      }
      if (sent <= 0) {
        throw std::runtime_error(
            "Could not write complete record to persistent worker socket");
      }
      cursor += static_cast<std::size_t>(sent);
      remaining -= static_cast<std::size_t>(sent);
    }
  }

  template <typename TValue>
  bool readDescriptorRecord(
      int const descriptor, TValue& value,
      bool const allowCleanEof = false) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    return readAllFromDescriptor(
        descriptor, &value, sizeof(value),
        allowCleanEof);
  }

  template <typename TValue>
  void writeDescriptorRecord(
      int const descriptor, TValue const& value) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    writeAllToDescriptor(
        descriptor, &value, sizeof(value));
  }
#endif

  FourMomentum decodeFourMomentum(
      std::array<double, 4> const& values,
      CoordinateSystemPtr const& coordinateSystem) {
    return FourMomentum{
        values[0] * 1_GeV,
        MomentumVector{
            coordinateSystem,
            {values[1] * 1_GeV, values[2] * 1_GeV,
             values[3] * 1_GeV}}};
  }

  EncodedResult executeRequest(
      corsika::fluka::InteractionModel& model,
      HadronicInteractionRequest const& request,
      CoordinateSystemPtr const& coordinateSystem) {
    EncodedResult result;
    result.header.sequence_id = request.sequence_id;

    try {
      validateHadronicInteractionRequest(request);
      if (request.model != HadronicWorkerModel::Fluka) {
        result.header.status =
            HadronicWorkerStatus::InvalidRequest;
        return result;
      }

      auto const projectile =
          convert_from_PDG(
              static_cast<PDGCode>(request.projectile_pdg));
      auto const target =
          convert_from_PDG(
              static_cast<PDGCode>(request.target_pdg));
      auto const projectileP4 = decodeFourMomentum(
          request.projectile_four_momentum_GeV, coordinateSystem);
      auto const targetP4 = decodeFourMomentum(
          request.target_four_momentum_GeV, coordinateSystem);
      if (!model.isValid(
              projectile, target,
              (projectileP4 + targetP4).getNorm())) {
        result.header.status =
            model.getMaterialIndex(target) < 0
                ? HadronicWorkerStatus::UnsupportedTarget
                : HadronicWorkerStatus::UnsupportedProjectile;
        return result;
      }

      auto keyedRng = default_prng_type{
          deriveHadronicWorkerSeed(request.random_key), 0};
      std::uint64_t supplied = 0;
      ::fluka::set_rng_function(
          [rng = std::move(keyedRng), &supplied](
              double* destination,
              std::size_t const count) mutable {
            std::uniform_real_distribution<double> uniform{0., 1.};
            std::generate_n(
                destination, count,
                [&]() { return uniform(rng); });
            supplied += count;
          });

      auto const finalStateStart =
          std::chrono::steady_clock::now();
      auto const finalState = model.generateFinalState(
          projectile, target, projectileP4, targetP4);
      result.header.final_state_time_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() -
              finalStateStart)
              .count();
      if (finalState.size() >
          std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "FLUKA final state exceeds protocol count range");
      }
      result.secondaries.reserve(finalState.size());
      for (auto const& [pid, kineticEnergy, direction] :
           finalState) {
        HadronicSecondaryRecord record{};
        record.pdg =
            static_cast<std::int32_t>(get_PDG(pid));
        record.kinetic_energy_GeV =
            kineticEnergy / 1_GeV;
        auto const components =
            direction.getComponents(coordinateSystem)
                .getEigenVector();
        record.direction = {
            components[0], components[1], components[2]};
        if (!std::isfinite(record.kinetic_energy_GeV) ||
            record.kinetic_energy_GeV < 0. ||
            !std::all_of(
                record.direction.begin(),
                record.direction.end(),
                [](double const value) {
                  return std::isfinite(value);
                })) {
          throw std::runtime_error(
              "FLUKA generated an invalid secondary record");
        }
        result.secondaries.push_back(record);
      }
      result.header.status = HadronicWorkerStatus::Success;
      result.header.secondary_count =
          static_cast<std::uint32_t>(
              result.secondaries.size());
      result.header.random_values_supplied =
          supplied >
                  std::numeric_limits<std::uint32_t>::max()
              ? std::numeric_limits<std::uint32_t>::max()
              : static_cast<std::uint32_t>(supplied);
    } catch (std::exception const& error) {
      std::cerr << "FLUKA worker request "
                << request.sequence_id
                << " failed: " << error.what() << '\n';
      result.header.status =
          HadronicWorkerStatus::GeneratorFailure;
      result.header.secondary_count = 0;
      result.secondaries.clear();
    }
    return result;
  }

  HadronicInteractionRequest makeSelfTestRequest(
      std::uint64_t const sequenceId,
      Code const projectile, Code const target,
      double const momentumGeV) {
    HadronicInteractionRequest request;
    request.sequence_id = sequenceId;
    request.random_key = HadronicRandomKey{
        0xC08A2026ULL, 7, 1000 + sequenceId,
        3, 1, 0};
    request.projectile_pdg =
        static_cast<std::int32_t>(get_PDG(projectile));
    request.target_pdg =
        static_cast<std::int32_t>(get_PDG(target));
    auto const momentum = momentumGeV * 1_GeV;
    request.projectile_four_momentum_GeV = {
        calculate_total_energy(
            momentum, get_mass(projectile)) /
            1_GeV,
        0., 0., momentumGeV};
    request.target_four_momentum_GeV = {
        get_mass(target) / 1_GeV, 0., 0., 0.};
    return request;
  }

  bool equalResult(
      EncodedResult const& lhs,
      EncodedResult const& rhs) {
    if (lhs.header.protocol_version !=
            rhs.header.protocol_version ||
        lhs.header.status != rhs.header.status ||
        lhs.header.sequence_id != rhs.header.sequence_id ||
        lhs.header.secondary_count !=
            rhs.header.secondary_count ||
        lhs.header.random_values_supplied !=
            rhs.header.random_values_supplied ||
        lhs.secondaries.size() !=
            rhs.secondaries.size()) {
      return false;
    }
    if (!std::isfinite(lhs.header.final_state_time_ms) ||
        !std::isfinite(rhs.header.final_state_time_ms) ||
        lhs.header.final_state_time_ms < 0. ||
        rhs.header.final_state_time_ms < 0.) {
      return false;
    }
    for (std::size_t i = 0;
         i < lhs.secondaries.size(); ++i) {
      auto const& a = lhs.secondaries[i];
      auto const& b = rhs.secondaries[i];
      if (a.pdg != b.pdg ||
          a.kinetic_energy_GeV !=
              b.kinetic_energy_GeV ||
          a.direction != b.direction) {
        return false;
      }
    }
    return true;
  }

  bool equalResponse(
      HadronicInteractionResponse const& lhs,
      HadronicInteractionResponse const& rhs) {
    EncodedResult left{lhs.header, lhs.secondaries};
    EncodedResult right{rhs.header, rhs.secondaries};
    return equalResult(left, right);
  }

  std::uint64_t hashResponses(
      std::vector<HadronicCompletedBatch> const& batches) {
    std::uint64_t state = 1469598103934665603ULL;
    auto append = [&](void const* data, std::size_t const size) {
      auto const* bytes =
          static_cast<unsigned char const*>(data);
      for (std::size_t i = 0; i < size; ++i) {
        state ^= bytes[i];
        state *= 1099511628211ULL;
      }
    };
    std::vector<HadronicInteractionResponse const*>
        ordered;
    for (auto const& batch : batches) {
      for (auto const& response : batch.responses) {
        ordered.push_back(&response);
      }
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](auto const* lhs, auto const* rhs) {
          return lhs->header.sequence_id <
                 rhs->header.sequence_id;
    });
    for (auto const* response : ordered) {
      auto physicalHeader = response->header;
      physicalHeader.final_state_time_ms = 0.;
      append(&physicalHeader, sizeof(physicalHeader));
      if (!response->secondaries.empty()) {
        append(
            response->secondaries.data(),
            response->secondaries.size() *
                sizeof(response->secondaries.front()));
      }
    }
    return state;
  }

} // namespace

int main(int argc, char** argv) {
  CLI::App app{
      "Process-isolated deterministic FLUKA final-state batch worker"};
  std::filesystem::path inputPath;
  std::filesystem::path outputPath;
  std::size_t selfTestRequests = 0;
  std::size_t poolSelfTestRequests = 0;
  std::size_t poolWorkers = 4;
  std::size_t generatedTestRequests = 0;
  int serverDescriptor = -1;
  std::size_t workerId = 0;
  app.add_option(
      "--input", inputPath,
      "Binary HadronicBatchProtocol request file");
  app.add_option(
      "--output", outputPath,
      "Binary HadronicBatchProtocol response file");
  app.add_option(
      "--self-test", selfTestRequests,
      "Run this many deterministic requests without input files");
  app.add_option(
      "--pool-self-test", poolSelfTestRequests,
      "Run this many requests twice through persistent worker processes");
  app.add_option(
      "--pool-workers", poolWorkers,
      "Persistent workers used by --pool-self-test");
  app.add_option(
      "--server-fd", serverDescriptor,
      "Run a persistent binary protocol server on this socket descriptor");
  app.add_option(
      "--worker-id", workerId,
      "Diagnostic worker identifier for persistent server mode");
  app.add_option(
      "--generate-test-input", generatedTestRequests,
      "Write this many deterministic requests to --output and exit");
  CLI11_PARSE(app, argc, argv);

  if (selfTestRequests == 0 &&
      poolSelfTestRequests == 0 &&
      serverDescriptor < 0 &&
      generatedTestRequests == 0 &&
      (inputPath.empty() || outputPath.empty())) {
    std::cerr
        << "Either --self-test or both --input and --output are required\n";
    return EXIT_FAILURE;
  }
  if (generatedTestRequests > 0 && outputPath.empty()) {
    std::cerr
        << "--generate-test-input requires --output\n";
    return EXIT_FAILURE;
  }

  try {
    std::array<Code, 4> const selfTestProjectiles{
        Code::Proton, Code::PiPlus,
        Code::PiMinus, Code::KPlus};
    std::array<Code, 3> const selfTestTargets{
        Code::Nitrogen, Code::Oxygen, Code::Argon};
    auto makeSelfTestRequests = [&](std::size_t const count) {
      std::vector<HadronicInteractionRequest> requests;
      requests.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        requests.push_back(makeSelfTestRequest(
            static_cast<std::uint64_t>(i),
            selfTestProjectiles[
                i % selfTestProjectiles.size()],
            selfTestTargets[i % selfTestTargets.size()],
            1. + static_cast<double>(i % 80)));
      }
      return requests;
    };

    if (generatedTestRequests > 0) {
      if (generatedTestRequests >
          std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Generated test request count exceeds protocol range");
      }
      auto const requests =
          makeSelfTestRequests(generatedTestRequests);
      HadronicBatchHeader header;
      header.batch_id = 0xC08A2026ULL;
      header.request_count =
          static_cast<std::uint32_t>(requests.size());
      std::ofstream output{
          outputPath,
          std::ios::binary | std::ios::trunc};
      if (!output) {
        throw std::runtime_error(
            "Could not open generated request file: " +
            outputPath.string());
      }
      writeRecord(
          output, header, "generated hadronic batch header");
      if (!requests.empty()) {
        output.write(
            reinterpret_cast<char const*>(requests.data()),
            static_cast<std::streamsize>(
                requests.size() * sizeof(requests.front())));
      }
      if (!output) {
        throw std::runtime_error(
            "Could not write generated hadronic requests");
      }
      std::cout << "Generated " << requests.size()
                << " deterministic FLUKA requests in "
                << outputPath << '\n';
      return EXIT_SUCCESS;
    }

    if (poolSelfTestRequests > 0) {
      if (poolWorkers == 0) {
        throw std::invalid_argument(
            "--pool-workers must be positive");
      }
      auto const requests =
          makeSelfTestRequests(poolSelfTestRequests);
      auto const batchCount =
          std::min<std::size_t>(
              std::max<std::size_t>(poolWorkers * 4, 1),
              std::max<std::size_t>(requests.size(), 1));
      std::vector<HadronicRequestBatch> batches(
          batchCount);
      for (std::size_t i = 0; i < batchCount; ++i) {
        batches[i].batch_id =
            0xC08A000000000000ULL + i;
      }
      for (std::size_t i = 0; i < requests.size(); ++i) {
        batches[i % batchCount].requests.push_back(
            requests[i]);
      }
      batches.erase(
          std::remove_if(
              batches.begin(), batches.end(),
              [](auto const& batch) {
                return batch.requests.empty();
              }),
          batches.end());

      auto const construction_start =
          std::chrono::steady_clock::now();
      HadronicProcessPool pool{
          std::filesystem::absolute(argv[0]),
          poolWorkers};
      auto const construction_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() -
              construction_start)
              .count();
      auto const first_start =
          std::chrono::steady_clock::now();
      auto const first = pool.execute(batches);
      auto const first_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() -
              first_start)
              .count();
      auto const second_start =
          std::chrono::steady_clock::now();
      auto const repeated = pool.execute(batches);
      auto const second_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() -
              second_start)
              .count();
      if (first.size() != repeated.size()) {
        throw std::runtime_error(
            "Persistent worker self-test changed batch count");
      }
      std::uint64_t secondaries = 0;
      for (std::size_t batch = 0;
           batch < first.size(); ++batch) {
        if (first[batch].batch_id !=
                repeated[batch].batch_id ||
            first[batch].responses.size() !=
                repeated[batch].responses.size()) {
          throw std::runtime_error(
              "Persistent worker self-test changed batch identity");
        }
        for (std::size_t request = 0;
             request < first[batch].responses.size();
             ++request) {
          if (!equalResponse(
                  first[batch].responses[request],
                  repeated[batch].responses[request])) {
            throw std::runtime_error(
                "Persistent worker self-test is not deterministic");
          }
          secondaries +=
              first[batch].responses[request]
                  .secondaries.size();
        }
      }
      auto const& statistics = pool.statistics();
      auto const responseHash =
          hashResponses(first);
      std::cout
          << "Persistent FLUKA process pool self-test passed: workers="
          << pool.workerCount()
          << ", requests=" << requests.size()
          << ", batches_per_pass=" << batches.size()
          << ", secondaries_first_pass="
          << secondaries
          << ", construction_ms=" << construction_ms
          << ", cold_pass_ms=" << first_ms
          << ", warm_pass_ms=" << second_ms
          << ", response_hash=" << responseHash
          << ", bytes_sent=" << statistics.bytes_sent
          << ", bytes_received="
          << statistics.bytes_received << '\n';
      return EXIT_SUCCESS;
    }

    auto& manager = RNGManager<>::getInstance();
    manager.registerRandomStream("fluka");
    manager.setSeed(1);

    std::set<Code> const atmosphericTargets{
        Code::Hydrogen, Code::Nitrogen,
        Code::Oxygen, Code::Argon};
    corsika::fluka::InteractionModel model{
        atmosphericTargets};
    auto const& coordinateSystem =
        get_root_CoordinateSystem();

    if (selfTestRequests > 0) {
      auto const requests =
          makeSelfTestRequests(selfTestRequests);

      std::uint64_t secondaryCount = 0;
      std::uint64_t suppliedValues = 0;
      for (auto const& request : requests) {
        auto const first =
            executeRequest(
                model, request, coordinateSystem);
        auto const repeated =
            executeRequest(
                model, request, coordinateSystem);
        if (first.header.status !=
                HadronicWorkerStatus::Success ||
            !equalResult(first, repeated)) {
          std::cerr
              << "Deterministic FLUKA worker self-test failed for sequence "
              << request.sequence_id << '\n';
          return EXIT_FAILURE;
        }
        secondaryCount += first.secondaries.size();
        suppliedValues +=
            first.header.random_values_supplied;
      }
      std::cout
          << "FLUKA batch worker self-test passed: requests="
          << requests.size()
          << ", secondaries=" << secondaryCount
          << ", random_values_supplied="
          << suppliedValues << '\n';
      return EXIT_SUCCESS;
    }

#if defined(__unix__) || defined(__linux__)
    if (serverDescriptor >= 0) {
      constexpr std::uint32_t maximumRequests =
          1'000'000;
      std::uint64_t batchesProcessed = 0;
      std::uint64_t requestsProcessed = 0;
      while (true) {
        HadronicBatchHeader header;
        if (!readDescriptorRecord(
                serverDescriptor, header, true)) {
          break;
        }
        validateHadronicBatchHeader(header);
        if (header.request_count > maximumRequests) {
          throw std::runtime_error(
              "Persistent hadronic batch exceeds request safety limit");
        }
        std::vector<HadronicInteractionRequest> requests(
            header.request_count);
        if (!requests.empty()) {
          readAllFromDescriptor(
              serverDescriptor, requests.data(),
              requests.size() * sizeof(requests.front()),
              false);
        }

        std::vector<EncodedResult> results;
        results.reserve(requests.size());
        std::size_t totalSecondaries = 0;
        for (auto const& request : requests) {
          results.push_back(
              executeRequest(
                  model, request,
                  coordinateSystem));
          if (results.back().secondaries.size() >
              std::numeric_limits<std::size_t>::max() -
                  totalSecondaries) {
            throw std::overflow_error(
                "Persistent hadronic response secondary count overflow");
          }
          totalSecondaries +=
              results.back().secondaries.size();
        }
        std::vector<HadronicInteractionResultHeader>
            resultHeaders;
        resultHeaders.reserve(results.size());
        std::vector<HadronicSecondaryRecord>
            flattenedSecondaries;
        flattenedSecondaries.reserve(
            totalSecondaries);
        for (auto& result : results) {
          resultHeaders.push_back(result.header);
          flattenedSecondaries.insert(
              flattenedSecondaries.end(),
              result.secondaries.begin(),
              result.secondaries.end());
        }

        writeDescriptorRecord(
            serverDescriptor, header);
        if (!resultHeaders.empty()) {
          writeAllToDescriptor(
              serverDescriptor,
              resultHeaders.data(),
              resultHeaders.size() *
                  sizeof(resultHeaders.front()));
        }
        if (!flattenedSecondaries.empty()) {
          writeAllToDescriptor(
              serverDescriptor,
              flattenedSecondaries.data(),
              flattenedSecondaries.size() *
                  sizeof(flattenedSecondaries.front()));
        }
        ++batchesProcessed;
        requestsProcessed += requests.size();
      }
      std::cerr
          << "Persistent FLUKA worker " << workerId
          << " stopped cleanly after " << batchesProcessed
          << " batches and " << requestsProcessed
          << " requests\n";
      return EXIT_SUCCESS;
    }
#else
    if (serverDescriptor >= 0) {
      throw std::runtime_error(
          "--server-fd requires a POSIX process environment");
    }
#endif

    std::ifstream input{inputPath, std::ios::binary};
    if (!input) {
      throw std::runtime_error(
          "Could not open request file: " +
          inputPath.string());
    }
    std::ofstream output{
        outputPath,
        std::ios::binary | std::ios::trunc};
    if (!output) {
      throw std::runtime_error(
          "Could not open response file: " +
          outputPath.string());
    }

    auto header =
        readRecord<HadronicBatchHeader>(
            input, "hadronic batch header");
    validateHadronicBatchHeader(header);
    constexpr std::uint32_t maximumRequests =
        1'000'000;
    if (header.request_count > maximumRequests) {
      throw std::runtime_error(
          "Hadronic batch request count exceeds safety limit");
    }
    std::vector<HadronicInteractionRequest> requests(
        header.request_count);
    if (!requests.empty()) {
      input.read(
          reinterpret_cast<char*>(requests.data()),
          static_cast<std::streamsize>(
              requests.size() *
              sizeof(requests.front())));
      if (!input) {
        throw std::runtime_error(
            "Could not read complete hadronic request array");
      }
    }
    if (input.peek() !=
        std::char_traits<char>::eof()) {
      throw std::runtime_error(
          "Unexpected trailing bytes in hadronic request file");
    }

    std::vector<EncodedResult> results;
    results.reserve(requests.size());
    std::size_t totalSecondaries = 0;
    for (auto const& request : requests) {
      results.push_back(
          executeRequest(
              model, request, coordinateSystem));
      if (results.back().secondaries.size() >
          std::numeric_limits<std::size_t>::max() -
              totalSecondaries) {
        throw std::overflow_error(
            "Hadronic response secondary count overflow");
      }
      totalSecondaries +=
          results.back().secondaries.size();
    }
    std::vector<HadronicInteractionResultHeader>
        resultHeaders;
    resultHeaders.reserve(results.size());
    std::vector<HadronicSecondaryRecord>
        flattenedSecondaries;
    flattenedSecondaries.reserve(
        totalSecondaries);
    for (auto& result : results) {
      resultHeaders.push_back(result.header);
      flattenedSecondaries.insert(
          flattenedSecondaries.end(),
          result.secondaries.begin(),
          result.secondaries.end());
    }

    writeRecord(
        output, header, "hadronic response header");
    if (!resultHeaders.empty()) {
      output.write(
          reinterpret_cast<char const*>(
              resultHeaders.data()),
          static_cast<std::streamsize>(
              resultHeaders.size() *
              sizeof(resultHeaders.front())));
    }
    if (!flattenedSecondaries.empty()) {
      output.write(
          reinterpret_cast<char const*>(
              flattenedSecondaries.data()),
          static_cast<std::streamsize>(
              flattenedSecondaries.size() *
              sizeof(flattenedSecondaries.front())));
    }
    if (!output) {
      throw std::runtime_error(
          "Could not write bulk hadronic response records");
    }
  } catch (std::exception const& error) {
    std::cerr << "FLUKA batch worker failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
