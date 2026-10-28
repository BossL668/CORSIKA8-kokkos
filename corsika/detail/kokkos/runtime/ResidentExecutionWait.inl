/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  void ResidentExecutionWait<TExecutionSpace>::setProgress(
      std::function<bool()> progress) {
    if (blocking_ && progress)
      throw std::logic_error(
          "CUDA blocking waits cannot run a cooperative progress callback");
    progress_ = std::move(progress);
  }

  template <class TExecutionSpace>
  void ResidentExecutionWait<TExecutionSpace>::setBlocking(bool const enabled) {
    if (enabled && progress_)
      throw std::logic_error(
          "CUDA blocking waits cannot run a cooperative progress callback");
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<TExecutionSpace, Kokkos::Cuda>) {
      requireIdle();
      if (blocking_ != enabled) ticket_.reset();
      blocking_ = enabled;
      return;
    }
#endif
    if (enabled)
      throw std::logic_error(
          "Blocking CUDA resident waits require a CUDA execution space");
  }

  template <class TExecutionSpace>
  bool ResidentExecutionWait<TExecutionSpace>::blockingEnabled() const noexcept {
    return blocking_;
  }

  template <class TExecutionSpace>
  std::uint64_t ResidentExecutionWait<TExecutionSpace>::blockingCalls() const noexcept {
    return blockingCalls_;
  }

  template <class TExecutionSpace>
  double ResidentExecutionWait<TExecutionSpace>::blockingHostSeconds() const noexcept {
    return blockingHostSeconds_;
  }

  template <class TExecutionSpace>
  void ResidentExecutionWait<TExecutionSpace>::resetStatistics() {
    requireIdle();
    blockingCalls_ = 0;
    blockingHostSeconds_ = 0.;
  }

  template <class TExecutionSpace>
  void ResidentExecutionWait<TExecutionSpace>::wait(TExecutionSpace const& execution,
                                                    char const* label) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<TExecutionSpace, Kokkos::Cuda>) {
      if (blocking_) {
        auto const start = std::chrono::steady_clock::now();
        auto const stream = execution.cuda_stream();
        try {
          if (!ticket_) ticket_ = std::make_unique<CudaCompletionTicket>(true);
          ticket_->record(stream);
          ticket_->consumeBlocking();
        } catch (...) {
          // Event creation/recording may fail after an enqueued host-output copy.
          // Keep those buffers alive through this stream's exceptional drain.
          if (ticket_ && ticket_->pending())
            ticket_->drainNoThrow();
          else
            cudaStreamSynchronize(stream);
          throw;
        }
        ++blockingCalls_;
        blockingHostSeconds_ +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                .count();
        return;
      }
      if (progress_) {
        if (!ticket_) ticket_ = std::make_unique<CudaCompletionTicket>();
        ticket_->record(execution.cuda_stream());
        try {
          while (!ticket_->ready()) {
            if (!progress_()) std::this_thread::yield();
          }
          ticket_->consume();
        } catch (...) {
          ticket_->drainNoThrow(); // host result buffers may now unwind safely
          throw;
        }
        return;
      }
    }
#endif
    execution.fence(label); // unchanged single-endpoint path
  }

  template <class TExecutionSpace>
  void ResidentExecutionWait<TExecutionSpace>::requireIdle() const {
#ifdef KOKKOS_ENABLE_CUDA
    if (ticket_ && ticket_->pending())
      throw std::logic_error("Cannot reconfigure an in-flight CUDA resident wait");
#endif
  }
  template <class TExecutionSpace>
  void wait_resident_execution(TExecutionSpace const& execution, char const* label,
                               ResidentExecutionWait<TExecutionSpace>* waiter) {
    if (waiter)
      waiter->wait(execution, label);
    else
      execution.fence(label);
  }

} // namespace corsika::kokkos
