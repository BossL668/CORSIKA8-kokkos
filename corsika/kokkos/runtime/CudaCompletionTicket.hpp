/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace corsika::kokkos {

  /** CUDA event owned by one host thread; buffers outlive pending work. */
  class CudaCompletionTicket {
  public:
    explicit CudaCompletionTicket(bool const blockingSync = false);
    ~CudaCompletionTicket();
    CudaCompletionTicket(CudaCompletionTicket const&) = delete;
    CudaCompletionTicket& operator=(CudaCompletionTicket const&) = delete;
    void record(cudaStream_t stream);
    bool ready() const;
    void consume();
    // Only the explicit blocking ticket may put its CUDA driver to sleep. The
    // ordinary polling ticket and all existing callers retain their semantics.
    void consumeBlocking();
    bool pending() const noexcept;
    // Exceptional cleanup only: wait for this event, never the whole device.
    cudaError_t drainNoThrow() noexcept;

  private:
    static void check(cudaError_t status);
    cudaEvent_t event_{};
    bool pending_{};
    bool blockingSync_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/CudaCompletionTicket.inl>
