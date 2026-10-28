/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/scheduling/HadronicBatchProtocol.hpp>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__linux__)
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace corsika::kokkos {

  struct HadronicInteractionResponse {
    HadronicInteractionResultHeader header_{};
    std::vector<HadronicSecondaryRecord> secondaries_;
  };

  struct HadronicRequestBatch {
    std::uint64_t batchId_{};
    // A fixed worker assignment is part of deterministic execution. The default
    // value asks the pool to use input_batch_index % worker_count.
    std::size_t workerId_{std::numeric_limits<std::size_t>::max()};
    std::vector<HadronicInteractionRequest> requests_;
  };

  struct HadronicCompletedBatch {
    std::uint64_t batchId_{};
    std::size_t workerId_{};
    std::vector<HadronicInteractionResponse> responses_;
  };

  struct HadronicProcessPoolStatistics {
    std::uint64_t batches_{};
    std::uint64_t requests_{};
    std::uint64_t secondaries_{};
    std::uint64_t bytesSent_{};
    std::uint64_t bytesReceived_{};
    double dispatchTimeMs_{};
    double pollWaitTimeMs_{};
    double receiveTimeMs_{};
    double executeWallTimeMs_{};
    std::uint64_t bulkResponseBatches_{};
    std::uint64_t bulkResponsePayloadReads_{};
  };

  /**
   * Persistent process-isolated hadronic final-state worker pool.
   *
   * FLUKA stores mutable state in Fortran COMMON blocks. Each worker is therefore
   * an independent executable connected to the parent with a private Unix-domain
   * socket. The socket uses HadronicBatchProtocol records and never transports C++
   * pointers, stack iterators, or coordinate-system objects.
   *
   * execute() keeps at most one batch in flight per worker. Every batch has a stable
   * worker assignment; unspecified assignments use input_index % worker_count.
   * This deliberately avoids completion-order-dependent dispatch because FLUKA has
   * generator state beyond the injected random stream. Completed batches are returned
   * in the same order as the input vector.
   */
  class HadronicProcessPool {
  public:
    HadronicProcessPool(std::filesystem::path worker_executable,
                        std::size_t worker_count);

    HadronicProcessPool(HadronicProcessPool const&) = delete;
    HadronicProcessPool& operator=(HadronicProcessPool const&) = delete;
    HadronicProcessPool(HadronicProcessPool&&) = delete;
    HadronicProcessPool& operator=(HadronicProcessPool&&) = delete;

    ~HadronicProcessPool();

    std::size_t workerCount() const noexcept;

    HadronicProcessPoolStatistics const& statistics() const noexcept;

    std::vector<HadronicCompletedBatch> execute(
        std::vector<HadronicRequestBatch> const& batches);

  private:
#if defined(__unix__) || defined(__linux__)
    struct Worker {
      pid_t processId_{-1};
      int socketFd_{-1};
      bool busy_{};
      std::size_t inputBatchIndex_{};
    };

    static void sendAll(int const fd, void const* source, std::size_t const bytes);

    static void receiveAll(int const fd, void* destination, std::size_t const bytes);

    template <typename TValue>
    static void sendRecord(int const fd, TValue const& value);

    template <typename TValue>
    static TValue receiveRecord(int const fd);

    void createRuntimeRoot();

    void startWorker(std::size_t const worker_id);

    void dispatch(std::size_t const worker_id, std::size_t const input_batch_index,
                  HadronicRequestBatch const& batch);

    HadronicCompletedBatch receive(std::size_t const worker_id,
                                   HadronicRequestBatch const& expected);

    void stopWorkers() noexcept;
#else
    void stopWorkers() noexcept;
    struct Worker {};
#endif

    std::filesystem::path workerExecutable_;
    std::filesystem::path runtimeRoot_;
    std::vector<Worker> workers_;
    HadronicProcessPoolStatistics statistics_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/HadronicProcessPool.inl>
