/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <Kokkos_Core.hpp>
#ifdef KOKKOS_ENABLE_CUDA
/* CUDA-specific asynchronous control ticket. No global Kokkos fence. */
#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace execution_wait_reference {
  class CudaCompletionTicket {
  public:
    explicit CudaCompletionTicket(bool const blocking_sync = false)
        : blocking_sync_(blocking_sync) {
      check(cudaEventCreateWithFlags(
          &event_,
          cudaEventDisableTiming | (blocking_sync_ ? cudaEventBlockingSync : 0U)));
    }
    ~CudaCompletionTicket() {
      if (event_) cudaEventDestroy(event_);
    }
    CudaCompletionTicket(CudaCompletionTicket const&) = delete;
    CudaCompletionTicket& operator=(CudaCompletionTicket const&) = delete;
    void record(cudaStream_t stream) {
      if (pending_) throw std::logic_error("CUDA control ticket is already in flight");
      check(cudaEventRecord(event_, stream));
      pending_ = true;
    }
    bool ready() const {
      if (!pending_) throw std::logic_error("CUDA control ticket was not submitted");
      auto const status = cudaEventQuery(event_);
      if (status == cudaErrorNotReady) return false;
      check(status);
      return true;
    }
    void consume() {
      if (!ready()) throw std::logic_error("CUDA control data is not ready");
      pending_ = false;
    }
    // Only the explicit blocking ticket may put its CUDA driver to sleep. The
    // ordinary polling ticket and all existing callers retain their semantics.
    void consumeBlocking() {
      if (!blocking_sync_)
        throw std::logic_error("CUDA control ticket does not support blocking waits");
      if (!pending_) throw std::logic_error("CUDA control ticket was not submitted");
      check(cudaEventSynchronize(event_));
      pending_ = false;
    }
    bool pending() const noexcept { return pending_; }
    // Exceptional cleanup only: wait for this event, never the whole device.
    cudaError_t drainNoThrow() noexcept {
      if (!pending_) return cudaSuccess;
      auto const status = cudaEventSynchronize(event_);
      pending_ = false;
      return status;
    }

  private:
    static void check(cudaError_t status) {
      if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA cooperative completion error: ") +
                                 cudaGetErrorString(status));
    }
    cudaEvent_t event_{};
    bool pending_{};
    bool blocking_sync_{};
  };
} // namespace execution_wait_reference

#endif
#pragma once
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#ifdef KOKKOS_ENABLE_CUDA

#endif

namespace execution_wait_reference {
  // Stackful host continuation: the device queue never leaves its execution
  // space. Only the existing control/output waits yield to bounded host work.
  // One host thread owns both the callback and the reusable event. Independent
  // CPU-primary mode instead lets its CUDA driver sleep at the existing waits;
  // no global device scheduling flags or physics kernels are changed.
  template <class ExecutionSpace>
  class ResidentExecutionWait {
  public:
    void setProgress(std::function<bool()> progress) {
      if (blocking_ && progress)
        throw std::logic_error(
            "CUDA blocking waits cannot run a cooperative progress callback");
      progress_ = std::move(progress);
    }
    void setBlocking(bool const enabled) {
      if (enabled && progress_)
        throw std::logic_error(
            "CUDA blocking waits cannot run a cooperative progress callback");
#ifdef KOKKOS_ENABLE_CUDA
      if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
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
    bool blockingEnabled() const noexcept { return blocking_; }
    std::uint64_t blockingCalls() const noexcept { return blocking_calls_; }
    double blockingHostSeconds() const noexcept { return blocking_host_seconds_; }
    // Called only after the previous shower is drained. Preserve mode and event.
    void resetStatistics() {
      requireIdle();
      blocking_calls_ = 0;
      blocking_host_seconds_ = 0.;
    }
    void wait(ExecutionSpace const& execution, char const* label) {
#ifdef KOKKOS_ENABLE_CUDA
      if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
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
          ++blocking_calls_;
          blocking_host_seconds_ +=
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

  private:
    void requireIdle() const {
#ifdef KOKKOS_ENABLE_CUDA
      if (ticket_ && ticket_->pending())
        throw std::logic_error("Cannot reconfigure an in-flight CUDA resident wait");
#endif
    }
    std::function<bool()> progress_;
    bool blocking_{};
    std::uint64_t blocking_calls_{};
    double blocking_host_seconds_{};
#ifdef KOKKOS_ENABLE_CUDA
    std::unique_ptr<CudaCompletionTicket> ticket_;
#endif
  };
  template <class ExecutionSpace>
  void waitResidentExecution(ExecutionSpace const& execution, char const* label,
                             ResidentExecutionWait<ExecutionSpace>* waiter) {
    if (waiter)
      waiter->wait(execution, label);
    else
      execution.fence(label);
  }
} // namespace execution_wait_reference
