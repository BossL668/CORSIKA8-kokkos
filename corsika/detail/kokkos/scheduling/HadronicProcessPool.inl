/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  inline HadronicProcessPool::HadronicProcessPool(std::filesystem::path worker_executable,
                                                  std::size_t worker_count)
      : workerExecutable_(std::filesystem::absolute(std::move(worker_executable))) {
#if defined(__unix__) || defined(__linux__)
    if (worker_count == 0) {
      throw std::invalid_argument("Hadronic process pool worker count must be positive");
    }
    if (workerExecutable_.empty() || !std::filesystem::exists(workerExecutable_)) {
      throw std::invalid_argument("Hadronic worker executable does not exist: " +
                                  workerExecutable_.string());
    }
    createRuntimeRoot();
    workers_.reserve(worker_count);
    try {
      for (std::size_t i = 0; i < worker_count; ++i) { startWorker(i); }
    } catch (...) {
      stopWorkers();
      throw;
    }
#else
    (void)worker_count;
    throw std::runtime_error("HadronicProcessPool requires a POSIX process environment");
#endif
  }
  inline HadronicProcessPool::~HadronicProcessPool() { stopWorkers(); }
  inline std::size_t HadronicProcessPool::workerCount() const noexcept {
    return workers_.size();
  }
  inline HadronicProcessPoolStatistics const& HadronicProcessPool::statistics() const
      noexcept {
    return statistics_;
  }
  inline std::vector<HadronicCompletedBatch> HadronicProcessPool::execute(
      std::vector<HadronicRequestBatch> const& batches) {
#if defined(__unix__) || defined(__linux__)
    if (batches.empty()) { return {}; }
    if (workers_.empty()) {
      throw std::logic_error("Hadronic process pool has no live workers");
    }

    constexpr std::size_t maximum_requests = 1'000'000;
    for (auto const& batch : batches) {
      if (batch.requests_.empty()) {
        throw std::invalid_argument(
            "Hadronic process pool cannot dispatch an empty batch");
      }
      if (batch.requests_.size() > maximum_requests ||
          batch.requests_.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Hadronic process pool batch exceeds request safety limit");
      }
      for (auto const& request : batch.requests_) {
        validate_hadronic_interaction_request(request);
      }
      if (batch.workerId_ != std::numeric_limits<std::size_t>::max() &&
          batch.workerId_ >= workers_.size()) {
        throw std::invalid_argument(
            "Hadronic batch has an invalid fixed worker assignment");
      }
    }

    std::vector<HadronicCompletedBatch> completed(batches.size());
    auto const execute_start = std::chrono::steady_clock::now();
    std::size_t completed_count = 0;
    std::vector<std::vector<std::size_t>> worker_batches(workers_.size());
    for (std::size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
      auto const requested_worker = batches[batch_index].workerId_;
      auto const worker_id = requested_worker == std::numeric_limits<std::size_t>::max()
                                 ? batch_index % workers_.size()
                                 : requested_worker;
      worker_batches[worker_id].push_back(batch_index);
    }
    std::vector<std::size_t> next_worker_batch(workers_.size(), 0);
    for (std::size_t worker_id = 0; worker_id < workers_.size(); ++worker_id) {
      if (worker_batches[worker_id].empty()) { continue; }
      auto const batch_index = worker_batches[worker_id][0];
      dispatch(worker_id, batch_index, batches[batch_index]);
      next_worker_batch[worker_id] = 1;
    }

    while (completed_count < batches.size()) {
      std::vector<pollfd> descriptors;
      std::vector<std::size_t> descriptor_workers;
      descriptors.reserve(workers_.size());
      descriptor_workers.reserve(workers_.size());
      for (std::size_t worker_id = 0; worker_id < workers_.size(); ++worker_id) {
        if (!workers_[worker_id].busy_) { continue; }
        descriptors.push_back(pollfd{workers_[worker_id].socketFd_,
                                     static_cast<short>(POLLIN | POLLERR | POLLHUP), 0});
        descriptor_workers.push_back(worker_id);
      }
      if (descriptors.empty()) {
        throw std::runtime_error("Hadronic process pool lost all in-flight workers");
      }

      int poll_result = 0;
      auto const poll_start = std::chrono::steady_clock::now();
      do {
        poll_result = ::poll(descriptors.data(), descriptors.size(), -1);
      } while (poll_result < 0 && errno == EINTR);
      if (poll_result < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "poll() failed for hadronic worker pool");
      }
      statistics_.pollWaitTimeMs_ += std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - poll_start)
                                         .count();

      for (std::size_t descriptor_index = 0; descriptor_index < descriptors.size();
           ++descriptor_index) {
        auto const events = descriptors[descriptor_index].revents;
        if (events == 0) { continue; }
        auto const worker_id = descriptor_workers[descriptor_index];
        if ((events & (POLLERR | POLLNVAL)) != 0) {
          throw std::runtime_error("Hadronic worker socket reported an error");
        }

        auto const input_index = workers_[worker_id].inputBatchIndex_;
        auto result = receive(worker_id, batches[input_index]);
        completed[input_index] = std::move(result);
        ++completed_count;

        if (next_worker_batch[worker_id] < worker_batches[worker_id].size()) {
          auto const batch_index =
              worker_batches[worker_id][next_worker_batch[worker_id]++];
          dispatch(worker_id, batch_index, batches[batch_index]);
        }
      }
    }
    statistics_.executeWallTimeMs_ +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  execute_start)
            .count();
    return completed;
#else
    (void)batches;
    throw std::runtime_error("HadronicProcessPool requires a POSIX process environment");
#endif
  }
#if defined(__unix__) || defined(__linux__)
  inline void HadronicProcessPool::sendAll(int const fd, void const* source,
                                           std::size_t const bytes) {
    auto const* cursor = static_cast<unsigned char const*>(source);
    std::size_t remaining = bytes;
    while (remaining > 0) {
      auto const sent = ::send(fd, cursor, remaining, MSG_NOSIGNAL);
      if (sent < 0 && errno == EINTR) { continue; }
      if (sent <= 0) {
        throw std::system_error(sent < 0 ? errno : EPIPE, std::generic_category(),
                                "Could not send a complete hadronic worker record");
      }
      cursor += static_cast<std::size_t>(sent);
      remaining -= static_cast<std::size_t>(sent);
    }
  }
  inline void HadronicProcessPool::receiveAll(int const fd, void* destination,
                                              std::size_t const bytes) {
    auto* cursor = static_cast<unsigned char*>(destination);
    std::size_t remaining = bytes;
    while (remaining > 0) {
      auto const received = ::recv(fd, cursor, remaining, 0);
      if (received < 0 && errno == EINTR) { continue; }
      if (received <= 0) {
        throw std::system_error(received < 0 ? errno : ECONNRESET,
                                std::generic_category(),
                                "Could not receive a complete hadronic worker record");
      }
      cursor += static_cast<std::size_t>(received);
      remaining -= static_cast<std::size_t>(received);
    }
  }
  template <typename TValue>
  inline void HadronicProcessPool::sendRecord(int const fd, TValue const& value) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    sendAll(fd, &value, sizeof(value));
  }
  template <typename TValue>
  inline TValue HadronicProcessPool::receiveRecord(int const fd) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    TValue value{};
    receiveAll(fd, &value, sizeof(value));
    return value;
  }
  inline void HadronicProcessPool::createRuntimeRoot() {
    auto const base = std::filesystem::temp_directory_path();
    auto const process = static_cast<unsigned long long>(::getpid());
    auto const clock = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    for (std::size_t attempt = 0; attempt < 100; ++attempt) {
      auto const candidate =
          base / ("corsika8-hadronic-pool-" + std::to_string(process) + "-" +
                  std::to_string(clock) + "-" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        runtimeRoot_ = candidate;
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::filesystem::filesystem_error(
            "Could not create hadronic worker runtime directory", candidate, error);
      }
    }
    throw std::runtime_error(
        "Could not allocate a unique hadronic worker runtime directory");
  }
  inline void HadronicProcessPool::startWorker(std::size_t const worker_id) {
    auto const working_directory = runtimeRoot_ / ("worker-" + std::to_string(worker_id));
    std::filesystem::create_directory(working_directory);
    int sockets[2]{-1, -1};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
      throw std::system_error(errno, std::generic_category(),
                              "socketpair() failed for hadronic worker");
    }

    auto const child = ::fork();
    if (child < 0) {
      auto const failure = errno;
      ::close(sockets[0]);
      ::close(sockets[1]);
      throw std::system_error(failure, std::generic_category(),
                              "fork() failed for hadronic worker");
    }
    if (child == 0) {
      ::close(sockets[0]);
      for (auto const& existing : workers_) {
        if (existing.socketFd_ >= 0) { ::close(existing.socketFd_); }
      }
      if (::chdir(working_directory.c_str()) != 0) {
        auto const message = std::string{"Could not enter hadronic worker directory: "} +
                             std::strerror(errno) + "\n";
        [[maybe_unused]] auto const ignored =
            ::write(STDERR_FILENO, message.data(), message.size());
        ::_exit(126);
      }
      auto const descriptor = std::to_string(sockets[1]);
      auto const worker_id_text = std::to_string(worker_id);
      ::execl(workerExecutable_.c_str(), workerExecutable_.filename().c_str(),
              "--server-fd", descriptor.c_str(), "--worker-id", worker_id_text.c_str(),
              static_cast<char*>(nullptr));
      auto const message =
          std::string{"Could not exec hadronic worker: "} + std::strerror(errno) + "\n";
      [[maybe_unused]] auto const ignored =
          ::write(STDERR_FILENO, message.data(), message.size());
      ::_exit(127);
    }

    ::close(sockets[1]);
    workers_.push_back(Worker{child, sockets[0], false, 0});
  }
  inline void HadronicProcessPool::dispatch(std::size_t const worker_id,
                                            std::size_t const input_batch_index,
                                            HadronicRequestBatch const& batch) {
    auto& worker = workers_.at(worker_id);
    if (worker.busy_) {
      throw std::logic_error("Cannot dispatch to a busy hadronic worker");
    }
    HadronicBatchHeader header;
    header.batchId_ = batch.batchId_;
    header.requestCount_ = static_cast<std::uint32_t>(batch.requests_.size());

    auto const start = std::chrono::steady_clock::now();
    sendRecord(worker.socketFd_, header);
    sendAll(worker.socketFd_, batch.requests_.data(),
            batch.requests_.size() * sizeof(batch.requests_.front()));
    statistics_.dispatchTimeMs_ += std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
    statistics_.bytesSent_ +=
        sizeof(header) + batch.requests_.size() * sizeof(batch.requests_.front());
    worker.busy_ = true;
    worker.inputBatchIndex_ = input_batch_index;
  }
  inline HadronicCompletedBatch HadronicProcessPool::receive(
      std::size_t const worker_id, HadronicRequestBatch const& expected) {
    auto& worker = workers_.at(worker_id);
    if (!worker.busy_) {
      throw std::logic_error("Cannot receive from an idle hadronic worker");
    }
    auto const start = std::chrono::steady_clock::now();
    auto const header = receiveRecord<HadronicBatchHeader>(worker.socketFd_);
    validate_hadronic_batch_header(header);
    if (header.batchId_ != expected.batchId_ ||
        header.requestCount_ != expected.requests_.size()) {
      throw std::runtime_error("Hadronic worker returned a mismatched batch header");
    }

    HadronicCompletedBatch completed;
    completed.batchId_ = header.batchId_;
    completed.workerId_ = worker_id;
    std::uint64_t received_bytes = sizeof(header);
    std::vector<HadronicInteractionResultHeader> result_headers(header.requestCount_);
    if (!result_headers.empty()) {
      receiveAll(worker.socketFd_, result_headers.data(),
                 result_headers.size() * sizeof(result_headers.front()));
      ++statistics_.bulkResponsePayloadReads_;
      received_bytes += result_headers.size() * sizeof(result_headers.front());
    }

    constexpr std::size_t maximum_secondary_records = 100'000'000;
    std::size_t total_secondaries = 0;
    for (std::size_t i = 0; i < result_headers.size(); ++i) {
      auto const& result_header = result_headers[i];
      if (result_header.protocolVersion_ != HadronicBatchProtocolVersion) {
        throw std::runtime_error(
            "Hadronic worker returned an unsupported result version");
      }
      if (result_header.sequenceId_ != expected.requests_[i].sequenceId_) {
        throw std::runtime_error("Hadronic worker changed request order or sequence ID");
      }
      if (result_header.status_ != HadronicWorkerStatus::Success) {
        throw std::runtime_error(
            "Hadronic worker failed request sequence " +
            std::to_string(result_header.sequenceId_) + " with status " +
            std::to_string(static_cast<std::uint32_t>(result_header.status_)));
      }
      if (result_header.secondaryCount_ > maximum_secondary_records - total_secondaries) {
        throw std::runtime_error(
            "Hadronic worker response exceeds the secondary safety limit");
      }
      total_secondaries += result_header.secondaryCount_;
    }

    std::vector<HadronicSecondaryRecord> flattened_secondaries(total_secondaries);
    if (!flattened_secondaries.empty()) {
      auto const secondary_bytes =
          flattened_secondaries.size() * sizeof(flattened_secondaries.front());
      receiveAll(worker.socketFd_, flattened_secondaries.data(), secondary_bytes);
      ++statistics_.bulkResponsePayloadReads_;
      received_bytes += secondary_bytes;
    }

    completed.responses_.reserve(header.requestCount_);
    std::size_t secondary_offset = 0;
    for (auto const& result_header : result_headers) {
      HadronicInteractionResponse response;
      response.header_ = result_header;
      auto const secondary_end = secondary_offset + result_header.secondaryCount_;
      response.secondaries_.insert(
          response.secondaries_.end(),
          flattened_secondaries.begin() + static_cast<std::ptrdiff_t>(secondary_offset),
          flattened_secondaries.begin() + static_cast<std::ptrdiff_t>(secondary_end));
      secondary_offset = secondary_end;
      statistics_.secondaries_ += response.secondaries_.size();
      completed.responses_.push_back(std::move(response));
    }
    statistics_.receiveTimeMs_ += std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
    statistics_.bytesReceived_ += received_bytes;
    ++statistics_.batches_;
    ++statistics_.bulkResponseBatches_;
    statistics_.requests_ += expected.requests_.size();
    worker.busy_ = false;
    return completed;
  }
  inline void HadronicProcessPool::stopWorkers() noexcept {
    for (auto& worker : workers_) {
      if (worker.socketFd_ >= 0) {
        ::shutdown(worker.socketFd_, SHUT_RDWR);
        ::close(worker.socketFd_);
        worker.socketFd_ = -1;
      }
    }
    for (auto& worker : workers_) {
      if (worker.processId_ <= 0) { continue; }
      int status = 0;
      while (::waitpid(worker.processId_, &status, 0) < 0 && errno == EINTR) {}
      worker.processId_ = -1;
    }
    workers_.clear();
    if (!runtimeRoot_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(runtimeRoot_, ignored);
      runtimeRoot_.clear();
    }
  }
#else
  inline void HadronicProcessPool::stopWorkers() noexcept {}
#endif
} // namespace corsika::kokkos
