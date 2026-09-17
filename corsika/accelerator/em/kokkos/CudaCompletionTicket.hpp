/* CUDA-specific asynchronous control ticket. No global Kokkos fence. */
#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace corsika::accelerator::em::kokkos_detail {
class CudaCompletionTicket {
 public:
  explicit CudaCompletionTicket(bool const blocking_sync = false)
      : blocking_sync_(blocking_sync) {
    check(cudaEventCreateWithFlags(
        &event_, cudaEventDisableTiming |
                     (blocking_sync_ ? cudaEventBlockingSync : 0U)));
  }
  ~CudaCompletionTicket() { if (event_) cudaEventDestroy(event_); }
  CudaCompletionTicket(CudaCompletionTicket const&) = delete;
  CudaCompletionTicket& operator=(CudaCompletionTicket const&) = delete;
  void record(cudaStream_t stream) {
    if (pending_) throw std::logic_error("CUDA control ticket is already in flight");
    check(cudaEventRecord(event_, stream)); pending_ = true;
  }
  bool ready() const {
    if (!pending_) throw std::logic_error("CUDA control ticket was not submitted");
    auto const status = cudaEventQuery(event_);
    if (status == cudaErrorNotReady) return false;
    check(status); return true;
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
    if (!pending_)
      throw std::logic_error("CUDA control ticket was not submitted");
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
      throw std::runtime_error(std::string("CUDA cooperative completion error: ") + cudaGetErrorString(status));
  }
  cudaEvent_t event_{};
  bool pending_{};
  bool blocking_sync_{};
};
} // namespace corsika::accelerator::em::kokkos_detail
