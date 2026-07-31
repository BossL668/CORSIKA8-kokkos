/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HadronicBatchProtocol.hpp>

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

namespace corsika {

  struct HadronicInteractionResponse {
    HadronicInteractionResultHeader header{};
    std::vector<HadronicSecondaryRecord> secondaries;
  };

  struct HadronicRequestBatch {
    std::uint64_t batch_id{};
    // A fixed worker assignment is part of deterministic execution. The default
    // value asks the pool to use input_batch_index % worker_count.
    std::size_t worker_id{std::numeric_limits<std::size_t>::max()};
    std::vector<HadronicInteractionRequest> requests;
  };

  struct HadronicCompletedBatch {
    std::uint64_t batch_id{};
    std::size_t worker_id{};
    std::vector<HadronicInteractionResponse> responses;
  };

  struct HadronicProcessPoolStatistics {
    std::uint64_t batches{};
    std::uint64_t requests{};
    std::uint64_t secondaries{};
    std::uint64_t bytes_sent{};
    std::uint64_t bytes_received{};
    double dispatch_time_ms{};
    double poll_wait_time_ms{};
    double receive_time_ms{};
    double execute_wall_time_ms{};
    std::uint64_t bulk_response_batches{};
    std::uint64_t bulk_response_payload_reads{};
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
                        std::size_t worker_count)
        : worker_executable_(
              std::filesystem::absolute(
                  std::move(worker_executable))) {
#if defined(__unix__) || defined(__linux__)
      if (worker_count == 0) {
        throw std::invalid_argument(
            "Hadronic process pool worker count must be positive");
      }
      if (worker_executable_.empty() ||
          !std::filesystem::exists(worker_executable_)) {
        throw std::invalid_argument(
            "Hadronic worker executable does not exist: " +
            worker_executable_.string());
      }
      createRuntimeRoot();
      workers_.reserve(worker_count);
      try {
        for (std::size_t i = 0; i < worker_count; ++i) {
          startWorker(i);
        }
      } catch (...) {
        stopWorkers();
        throw;
      }
#else
      (void)worker_count;
      throw std::runtime_error(
          "HadronicProcessPool requires a POSIX process environment");
#endif
    }

    HadronicProcessPool(HadronicProcessPool const&) = delete;
    HadronicProcessPool& operator=(HadronicProcessPool const&) = delete;
    HadronicProcessPool(HadronicProcessPool&&) = delete;
    HadronicProcessPool& operator=(HadronicProcessPool&&) = delete;

    ~HadronicProcessPool() { stopWorkers(); }

    std::size_t workerCount() const noexcept { return workers_.size(); }

    HadronicProcessPoolStatistics const& statistics() const noexcept {
      return statistics_;
    }

    std::vector<HadronicCompletedBatch> execute(
        std::vector<HadronicRequestBatch> const& batches) {
#if defined(__unix__) || defined(__linux__)
      if (batches.empty()) {
        return {};
      }
      if (workers_.empty()) {
        throw std::logic_error("Hadronic process pool has no live workers");
      }

      constexpr std::size_t maximum_requests = 1'000'000;
      for (auto const& batch : batches) {
        if (batch.requests.empty()) {
          throw std::invalid_argument(
              "Hadronic process pool cannot dispatch an empty batch");
        }
        if (batch.requests.size() > maximum_requests ||
            batch.requests.size() >
                std::numeric_limits<std::uint32_t>::max()) {
          throw std::invalid_argument(
              "Hadronic process pool batch exceeds request safety limit");
        }
        for (auto const& request : batch.requests) {
          validateHadronicInteractionRequest(request);
        }
        if (batch.worker_id !=
                std::numeric_limits<std::size_t>::max() &&
            batch.worker_id >= workers_.size()) {
          throw std::invalid_argument(
              "Hadronic batch has an invalid fixed worker assignment");
        }
      }

      std::vector<HadronicCompletedBatch> completed(batches.size());
      auto const execute_start =
          std::chrono::steady_clock::now();
      std::size_t completed_count = 0;
      std::vector<std::vector<std::size_t>>
          worker_batches(workers_.size());
      for (std::size_t batch_index = 0;
           batch_index < batches.size(); ++batch_index) {
        auto const requested_worker =
            batches[batch_index].worker_id;
        auto const worker_id =
            requested_worker ==
                    std::numeric_limits<std::size_t>::max()
                ? batch_index % workers_.size()
                : requested_worker;
        worker_batches[worker_id].push_back(batch_index);
      }
      std::vector<std::size_t> next_worker_batch(
          workers_.size(), 0);
      for (std::size_t worker_id = 0;
           worker_id < workers_.size(); ++worker_id) {
        if (worker_batches[worker_id].empty()) {
          continue;
        }
        auto const batch_index =
            worker_batches[worker_id][0];
        dispatch(
            worker_id, batch_index,
            batches[batch_index]);
        next_worker_batch[worker_id] = 1;
      }

      while (completed_count < batches.size()) {
        std::vector<pollfd> descriptors;
        std::vector<std::size_t> descriptor_workers;
        descriptors.reserve(workers_.size());
        descriptor_workers.reserve(workers_.size());
        for (std::size_t worker_id = 0; worker_id < workers_.size();
             ++worker_id) {
          if (!workers_[worker_id].busy) {
            continue;
          }
          descriptors.push_back(
              pollfd{workers_[worker_id].socket_fd,
                     static_cast<short>(POLLIN | POLLERR | POLLHUP), 0});
          descriptor_workers.push_back(worker_id);
        }
        if (descriptors.empty()) {
          throw std::runtime_error(
              "Hadronic process pool lost all in-flight workers");
        }

        int poll_result = 0;
        auto const poll_start =
            std::chrono::steady_clock::now();
        do {
          poll_result =
              ::poll(descriptors.data(), descriptors.size(), -1);
        } while (poll_result < 0 && errno == EINTR);
        if (poll_result < 0) {
          throw std::system_error(
              errno, std::generic_category(),
              "poll() failed for hadronic worker pool");
        }
        statistics_.poll_wait_time_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                poll_start)
                .count();

        for (std::size_t descriptor_index = 0;
             descriptor_index < descriptors.size(); ++descriptor_index) {
          auto const events = descriptors[descriptor_index].revents;
          if (events == 0) {
            continue;
          }
          auto const worker_id =
              descriptor_workers[descriptor_index];
          if ((events & (POLLERR | POLLNVAL)) != 0) {
            throw std::runtime_error(
                "Hadronic worker socket reported an error");
          }

          auto const input_index =
              workers_[worker_id].input_batch_index;
          auto result =
              receive(worker_id, batches[input_index]);
          completed[input_index] = std::move(result);
          ++completed_count;

          if (next_worker_batch[worker_id] <
              worker_batches[worker_id].size()) {
            auto const batch_index =
                worker_batches[worker_id]
                              [next_worker_batch[worker_id]++];
            dispatch(
                worker_id, batch_index,
                batches[batch_index]);
          }
        }
      }
      statistics_.execute_wall_time_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() -
              execute_start)
              .count();
      return completed;
#else
      (void)batches;
      throw std::runtime_error(
          "HadronicProcessPool requires a POSIX process environment");
#endif
    }

  private:
#if defined(__unix__) || defined(__linux__)
    struct Worker {
      pid_t process_id{-1};
      int socket_fd{-1};
      bool busy{};
      std::size_t input_batch_index{};
    };

    static void sendAll(int const fd, void const* source,
                        std::size_t const bytes) {
      auto const* cursor = static_cast<unsigned char const*>(source);
      std::size_t remaining = bytes;
      while (remaining > 0) {
        auto const sent =
            ::send(fd, cursor, remaining, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
          continue;
        }
        if (sent <= 0) {
          throw std::system_error(
              sent < 0 ? errno : EPIPE, std::generic_category(),
              "Could not send a complete hadronic worker record");
        }
        cursor += static_cast<std::size_t>(sent);
        remaining -= static_cast<std::size_t>(sent);
      }
    }

    static void receiveAll(int const fd, void* destination,
                           std::size_t const bytes) {
      auto* cursor = static_cast<unsigned char*>(destination);
      std::size_t remaining = bytes;
      while (remaining > 0) {
        auto const received =
            ::recv(fd, cursor, remaining, 0);
        if (received < 0 && errno == EINTR) {
          continue;
        }
        if (received <= 0) {
          throw std::system_error(
              received < 0 ? errno : ECONNRESET,
              std::generic_category(),
              "Could not receive a complete hadronic worker record");
        }
        cursor += static_cast<std::size_t>(received);
        remaining -= static_cast<std::size_t>(received);
      }
    }

    template <typename TValue>
    static void sendRecord(int const fd, TValue const& value) {
      static_assert(std::is_trivially_copyable_v<TValue>);
      sendAll(fd, &value, sizeof(value));
    }

    template <typename TValue>
    static TValue receiveRecord(int const fd) {
      static_assert(std::is_trivially_copyable_v<TValue>);
      TValue value{};
      receiveAll(fd, &value, sizeof(value));
      return value;
    }

    void createRuntimeRoot() {
      auto const base =
          std::filesystem::temp_directory_path();
      auto const process =
          static_cast<unsigned long long>(::getpid());
      auto const clock =
          static_cast<unsigned long long>(
              std::chrono::steady_clock::now()
                  .time_since_epoch()
                  .count());
      for (std::size_t attempt = 0; attempt < 100; ++attempt) {
        auto const candidate =
            base /
            ("corsika8-hadronic-pool-" +
             std::to_string(process) + "-" +
             std::to_string(clock) + "-" +
             std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(
                candidate, error)) {
          runtime_root_ = candidate;
          return;
        }
        if (error &&
            error !=
                std::errc::file_exists) {
          throw std::filesystem::filesystem_error(
              "Could not create hadronic worker runtime directory",
              candidate, error);
        }
      }
      throw std::runtime_error(
          "Could not allocate a unique hadronic worker runtime directory");
    }

    void startWorker(std::size_t const worker_id) {
      auto const working_directory =
          runtime_root_ /
          ("worker-" + std::to_string(worker_id));
      std::filesystem::create_directory(
          working_directory);
      int sockets[2]{-1, -1};
      if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        throw std::system_error(
            errno, std::generic_category(),
            "socketpair() failed for hadronic worker");
      }

      auto const child = ::fork();
      if (child < 0) {
        auto const failure = errno;
        ::close(sockets[0]);
        ::close(sockets[1]);
        throw std::system_error(
            failure, std::generic_category(),
            "fork() failed for hadronic worker");
      }
      if (child == 0) {
        ::close(sockets[0]);
        for (auto const& existing : workers_) {
          if (existing.socket_fd >= 0) {
            ::close(existing.socket_fd);
          }
        }
        if (::chdir(
                working_directory.c_str()) != 0) {
          auto const message =
              std::string{
                  "Could not enter hadronic worker directory: "} +
              std::strerror(errno) + "\n";
          [[maybe_unused]] auto const ignored =
              ::write(
                  STDERR_FILENO, message.data(),
                  message.size());
          ::_exit(126);
        }
        auto const descriptor = std::to_string(sockets[1]);
        auto const worker_id_text = std::to_string(worker_id);
        ::execl(
            worker_executable_.c_str(),
            worker_executable_.filename().c_str(),
            "--server-fd", descriptor.c_str(),
            "--worker-id", worker_id_text.c_str(),
            static_cast<char*>(nullptr));
        auto const message =
            std::string{"Could not exec hadronic worker: "} +
            std::strerror(errno) + "\n";
        [[maybe_unused]] auto const ignored =
            ::write(STDERR_FILENO, message.data(), message.size());
        ::_exit(127);
      }

      ::close(sockets[1]);
      workers_.push_back(Worker{child, sockets[0], false, 0});
    }

    void dispatch(std::size_t const worker_id,
                  std::size_t const input_batch_index,
                  HadronicRequestBatch const& batch) {
      auto& worker = workers_.at(worker_id);
      if (worker.busy) {
        throw std::logic_error(
            "Cannot dispatch to a busy hadronic worker");
      }
      HadronicBatchHeader header;
      header.batch_id = batch.batch_id;
      header.request_count =
          static_cast<std::uint32_t>(batch.requests.size());

      auto const start = std::chrono::steady_clock::now();
      sendRecord(worker.socket_fd, header);
      sendAll(
          worker.socket_fd, batch.requests.data(),
          batch.requests.size() * sizeof(batch.requests.front()));
      statistics_.dispatch_time_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
      statistics_.bytes_sent +=
          sizeof(header) +
          batch.requests.size() * sizeof(batch.requests.front());
      worker.busy = true;
      worker.input_batch_index = input_batch_index;
    }

    HadronicCompletedBatch receive(
        std::size_t const worker_id,
        HadronicRequestBatch const& expected) {
      auto& worker = workers_.at(worker_id);
      if (!worker.busy) {
        throw std::logic_error(
            "Cannot receive from an idle hadronic worker");
      }
      auto const start = std::chrono::steady_clock::now();
      auto const header =
          receiveRecord<HadronicBatchHeader>(worker.socket_fd);
      validateHadronicBatchHeader(header);
      if (header.batch_id != expected.batch_id ||
          header.request_count != expected.requests.size()) {
        throw std::runtime_error(
            "Hadronic worker returned a mismatched batch header");
      }

      HadronicCompletedBatch completed;
      completed.batch_id = header.batch_id;
      completed.worker_id = worker_id;
      std::uint64_t received_bytes = sizeof(header);
      std::vector<HadronicInteractionResultHeader>
          result_headers(header.request_count);
      if (!result_headers.empty()) {
        receiveAll(
            worker.socket_fd, result_headers.data(),
            result_headers.size() *
                sizeof(result_headers.front()));
        ++statistics_.bulk_response_payload_reads;
        received_bytes +=
            result_headers.size() *
            sizeof(result_headers.front());
      }

      constexpr std::size_t maximum_secondary_records =
          100'000'000;
      std::size_t total_secondaries = 0;
      for (std::size_t i = 0;
           i < result_headers.size(); ++i) {
        auto const& result_header =
            result_headers[i];
        if (result_header.protocol_version !=
            HadronicBatchProtocolVersion) {
          throw std::runtime_error(
              "Hadronic worker returned an unsupported result version");
        }
        if (result_header.sequence_id !=
            expected.requests[i].sequence_id) {
          throw std::runtime_error(
              "Hadronic worker changed request order or sequence ID");
        }
        if (result_header.status !=
            HadronicWorkerStatus::Success) {
          throw std::runtime_error(
              "Hadronic worker failed request sequence " +
              std::to_string(result_header.sequence_id) +
              " with status " +
              std::to_string(
                  static_cast<std::uint32_t>(
                      result_header.status)));
        }
        if (result_header.secondary_count >
            maximum_secondary_records -
                total_secondaries) {
          throw std::runtime_error(
              "Hadronic worker response exceeds the secondary safety limit");
        }
        total_secondaries +=
            result_header.secondary_count;
      }

      std::vector<HadronicSecondaryRecord>
          flattened_secondaries(total_secondaries);
      if (!flattened_secondaries.empty()) {
        auto const secondary_bytes =
            flattened_secondaries.size() *
            sizeof(flattened_secondaries.front());
        receiveAll(
            worker.socket_fd,
            flattened_secondaries.data(),
            secondary_bytes);
        ++statistics_.bulk_response_payload_reads;
        received_bytes += secondary_bytes;
      }

      completed.responses.reserve(header.request_count);
      std::size_t secondary_offset = 0;
      for (auto const& result_header :
           result_headers) {
        HadronicInteractionResponse response;
        response.header = result_header;
        auto const secondary_end =
            secondary_offset +
            result_header.secondary_count;
        response.secondaries.insert(
            response.secondaries.end(),
            flattened_secondaries.begin() +
                static_cast<std::ptrdiff_t>(
                    secondary_offset),
            flattened_secondaries.begin() +
                static_cast<std::ptrdiff_t>(
                    secondary_end));
        secondary_offset = secondary_end;
        statistics_.secondaries +=
            response.secondaries.size();
        completed.responses.push_back(std::move(response));
      }
      statistics_.receive_time_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
      statistics_.bytes_received += received_bytes;
      ++statistics_.batches;
      ++statistics_.bulk_response_batches;
      statistics_.requests += expected.requests.size();
      worker.busy = false;
      return completed;
    }

    void stopWorkers() noexcept {
      for (auto& worker : workers_) {
        if (worker.socket_fd >= 0) {
          ::shutdown(worker.socket_fd, SHUT_RDWR);
          ::close(worker.socket_fd);
          worker.socket_fd = -1;
        }
      }
      for (auto& worker : workers_) {
        if (worker.process_id <= 0) {
          continue;
        }
        int status = 0;
        while (::waitpid(worker.process_id, &status, 0) < 0 &&
               errno == EINTR) {}
        worker.process_id = -1;
      }
      workers_.clear();
      if (!runtime_root_.empty()) {
        std::error_code ignored;
        std::filesystem::remove_all(
            runtime_root_, ignored);
        runtime_root_.clear();
      }
    }
#else
    void stopWorkers() noexcept {}
    struct Worker {};
#endif

    std::filesystem::path worker_executable_;
    std::filesystem::path runtime_root_;
    std::vector<Worker> workers_;
    HadronicProcessPoolStatistics statistics_{};
  };

} // namespace corsika
