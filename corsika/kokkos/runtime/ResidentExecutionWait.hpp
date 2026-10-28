/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

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
#include <corsika/kokkos/runtime/CudaCompletionTicket.hpp>
#endif

namespace corsika::kokkos {

  /** Preserve production fence, cooperative polling and blocking-event waits.
   * One host thread owns both callback and event. Reconfigure only when idle.
   */
  template <class TExecutionSpace>
  class ResidentExecutionWait {
  public:
    void setProgress(std::function<bool()> progress);
    void setBlocking(bool const enabled);
    bool blockingEnabled() const noexcept;
    std::uint64_t blockingCalls() const noexcept;
    double blockingHostSeconds() const noexcept;
    // Called only after the previous shower is drained. Preserve mode and event.
    void resetStatistics();
    void wait(TExecutionSpace const& execution, char const* label);

  private:
    void requireIdle() const;
    std::function<bool()> progress_;
    bool blocking_{};
    std::uint64_t blockingCalls_{};
    double blockingHostSeconds_{};
#ifdef KOKKOS_ENABLE_CUDA
    std::unique_ptr<CudaCompletionTicket> ticket_;
#endif
  };
  template <class TExecutionSpace>
  void wait_resident_execution(TExecutionSpace const& execution, char const* label,
                               ResidentExecutionWait<TExecutionSpace>* waiter);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/ResidentExecutionWait.inl>
