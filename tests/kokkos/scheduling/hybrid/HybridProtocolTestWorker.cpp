/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Test-only IPC peer. It does not simulate an interaction or replace a generator.
#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>

using namespace corsika::kokkos;

namespace {
  bool receive_bytes(int fd, void* destination, std::size_t bytes) {
    auto* cursor = static_cast<char*>(destination);
    std::size_t offset = 0;
    while (offset < bytes) {
      auto const count = ::recv(fd, cursor + offset, bytes - offset, 0);
      if (count < 0 && errno == EINTR) continue;
      if (count == 0 && offset == 0) return false;
      if (count <= 0) throw std::runtime_error("Test request truncated");
      offset += count;
    }
    return true;
  }

  void send_bytes(int fd, void const* source, std::size_t bytes) {
    auto const* cursor = static_cast<char const*>(source);
    std::size_t offset = 0;
    while (offset < bytes) {
      // Exercise stream framing with writes smaller than a protocol record.
      auto const count = ::send(fd, cursor + offset,
                                std::min(bytes - offset, std::size_t{17}), MSG_NOSIGNAL);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::runtime_error("Test response disconnected");
      offset += count;
    }
  }
} // namespace

int main(int argc, char** argv) {
  try {
    int fd = -1;
    std::size_t worker = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
      std::string const option = argv[i];
      if (option == "--server-fd")
        fd = std::stoi(argv[i + 1]);
      else if (option == "--worker-id")
        worker = std::stoul(argv[i + 1]);
      else
        return 2;
    }
    if (fd < 0) return 2;
    std::string const scenario = std::filesystem::path(argv[0]).filename().string();

    for (;;) {
      HadronicBatchHeader header;
      if (!receive_bytes(fd, &header, sizeof(header))) break;
      validate_hadronic_batch_header(header);
      if (header.requestCount_ > 1000) return 3;
      std::vector<HadronicInteractionRequest> requests(header.requestCount_);
      if (!receive_bytes(fd, requests.data(), requests.size() * sizeof(requests[0])))
        return 4;
      if (worker == 0) std::this_thread::sleep_for(std::chrono::milliseconds(3));
      if (scenario == "disconnect") return 0;
      std::vector<HadronicInteractionResultHeader> results;
      std::vector<HadronicSecondaryRecord> secondaries;
      for (auto const& request : requests) {
        validate_hadronic_interaction_request(request);
        HadronicInteractionResultHeader result;
        result.sequenceId_ = request.sequenceId_;
        result.secondaryCount_ = 2;
        result.randomValuesSupplied_ =
            derive_hadronic_worker_seed(request.randomKey_) & 0xffffU;
        result.finalStateTimeMs_ = 0.25;
        for (std::uint32_t i = 0; i < result.secondaryCount_; ++i) {
          HadronicSecondaryRecord particle;
          particle.pdg_ = i == 0 ? 22 : 11;
          particle.kineticEnergyGeV_ = 0.5 + 0.01 * request.sequenceId_ + i;
          particle.direction_ = {0., 0., 1.};
          secondaries.push_back(particle);
        }
        results.push_back(result);
      }
      if (scenario == "batch") ++header.batchId_;
      if (scenario == "count") ++header.requestCount_;
      if (scenario == "magic") ++header.magic_;
      if (scenario == "endian") ++header.endianMarker_;
      if (scenario == "version") ++results.front().protocolVersion_;
      if (scenario == "order") ++results.front().sequenceId_;
      if (scenario == "failure")
        results.front().status_ = HadronicWorkerStatus::GeneratorFailure;
      if (scenario == "overflow") results.front().secondaryCount_ = 100000001;
      if (scenario == "truncated") {
        send_bytes(fd, &header, sizeof(header) / 2);
        return 0;
      }
      send_bytes(fd, &header, sizeof(header));
      send_bytes(fd, results.data(), results.size() * sizeof(results[0]));
      send_bytes(fd, secondaries.data(),
                 secondaries.size() * sizeof(HadronicSecondaryRecord));
    }
    ::close(fd);
    return 0;
  } catch (...) { return 5; }
}
