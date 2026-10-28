/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  WavefrontExecutionGuard<TExecutionSpace>::WavefrontExecutionGuard(
      WavefrontQueue<TExecutionSpace>& queue, TExecutionSpace const& execution) noexcept
      : queue_(&queue)
      , execution_(&execution) {}

  template <class TExecutionSpace>
  WavefrontExecutionGuard<TExecutionSpace>::~WavefrontExecutionGuard() noexcept {
    if (!active_) return;
    try {
      execution_->fence("unwind resident Kokkos EM wavefront");
      queue_->markExecutionSynchronized();
    } catch (...) {
      // The caller is already unwinding.  A CUDA/HIP/SYCL runtime failure
      // invalidates the current shower and must not replace that exception.
    }
  }

  template <class TExecutionSpace>
  void WavefrontExecutionGuard<TExecutionSpace>::release() noexcept {
    active_ = false;
  }

} // namespace corsika::kokkos
