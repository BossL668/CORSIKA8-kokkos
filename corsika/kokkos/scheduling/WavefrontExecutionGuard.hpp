/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <corsika/kokkos/scheduling/WavefrontQueue.hpp>

namespace corsika::kokkos {

  /** Drain the queue-owning execution instance on exceptional exit.
   * The borrowed queue and execution must outlive this guard; the caller
   * passes the same instance that owns the queue work. */
  template <class TExecutionSpace>
  class WavefrontExecutionGuard {
  public:
    WavefrontExecutionGuard(WavefrontQueue<TExecutionSpace>& queue,
                            TExecutionSpace const& execution) noexcept;

    WavefrontExecutionGuard(WavefrontExecutionGuard const&) = delete;
    WavefrontExecutionGuard& operator=(WavefrontExecutionGuard const&) = delete;

    ~WavefrontExecutionGuard() noexcept;

    void release() noexcept;

  private:
    WavefrontQueue<TExecutionSpace>* queue_{};
    TExecutionSpace const* execution_{};
    bool active_{true};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/WavefrontExecutionGuard.inl>
