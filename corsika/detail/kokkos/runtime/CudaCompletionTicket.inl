/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  inline CudaCompletionTicket::CudaCompletionTicket(bool const blockingSync)
      : blockingSync_(blockingSync) {
    check(cudaEventCreateWithFlags(
        &event_, cudaEventDisableTiming | (blockingSync_ ? cudaEventBlockingSync : 0U)));
  }

  inline CudaCompletionTicket::~CudaCompletionTicket() {
    if (event_) cudaEventDestroy(event_);
  }

  inline void CudaCompletionTicket::record(cudaStream_t stream) {
    if (pending_) throw std::logic_error("CUDA control ticket is already in flight");
    check(cudaEventRecord(event_, stream));
    pending_ = true;
  }

  inline bool CudaCompletionTicket::ready() const {
    if (!pending_) throw std::logic_error("CUDA control ticket was not submitted");
    auto const status = cudaEventQuery(event_);
    if (status == cudaErrorNotReady) return false;
    check(status);
    return true;
  }

  inline void CudaCompletionTicket::consume() {
    if (!ready()) throw std::logic_error("CUDA control data is not ready");
    pending_ = false;
  }

  inline void CudaCompletionTicket::consumeBlocking() {
    if (!blockingSync_)
      throw std::logic_error("CUDA control ticket does not support blocking waits");
    if (!pending_) throw std::logic_error("CUDA control ticket was not submitted");
    check(cudaEventSynchronize(event_));
    pending_ = false;
  }

  inline bool CudaCompletionTicket::pending() const noexcept { return pending_; }

  inline cudaError_t CudaCompletionTicket::drainNoThrow() noexcept {
    if (!pending_) return cudaSuccess;
    auto const status = cudaEventSynchronize(event_);
    pending_ = false;
    return status;
  }

  inline void CudaCompletionTicket::check(cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("CUDA cooperative completion error: ") +
                               cudaGetErrorString(status));
  }

} // namespace corsika::kokkos
