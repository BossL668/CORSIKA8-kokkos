/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once
#include <Kokkos_Core.hpp>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#ifdef KOKKOS_ENABLE_CUDA
#include <corsika/accelerator/em/kokkos/CudaCompletionTicket.hpp>
#endif

namespace corsika::accelerator::em::kokkos_detail {
// Stackful host continuation: the device queue never leaves its execution
// space. Only the existing control/output waits yield to bounded host work.
// One coordinator thread owns both the callback and the reusable event.
template<class ExecutionSpace> class ResidentExecutionWait {
 public:
  std::function<bool()> progress;
  void wait(ExecutionSpace const& execution, char const* label) {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      if (progress) {
        if (!ticket_) ticket_ = std::make_unique<CudaCompletionTicket>();
        ticket_->record(execution.cuda_stream());
        try {
          while (!ticket_->ready()) {
            if (!progress()) std::this_thread::yield();
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
#ifdef KOKKOS_ENABLE_CUDA
  std::unique_ptr<CudaCompletionTicket> ticket_;
#endif
};
template<class ExecutionSpace>
void waitResidentExecution(ExecutionSpace const& execution, char const* label,
                          ResidentExecutionWait<ExecutionSpace>* waiter) {
  if (waiter) waiter->wait(execution, label);
  else execution.fence(label);
}
} // namespace corsika::accelerator::em::kokkos_detail
