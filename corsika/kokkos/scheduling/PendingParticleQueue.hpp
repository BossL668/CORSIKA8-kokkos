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
#include <corsika/kokkos/data/ParticleState.hpp>
#include <corsika/kokkos/runtime/GeometricCapacity.hpp>
#include <corsika/kokkos/runtime/ResidentMemoryBudget.hpp>

namespace corsika::kokkos {

  /** Source-ordered pending particles; prefix download does not consume them.
   * The caller orders operations on one execution instance and drains it
   * before destroying views or clearing pending work. */
  template <class TExecutionSpace>
  class PendingParticleQueue {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using view_type = Kokkos::View<ParticleState*, memory_space_type>;

    explicit PendingParticleQueue(std::string label = {});

    std::size_t size() const noexcept;
    std::size_t head() const noexcept;
    std::size_t capacity() const noexcept;
    bool empty() const noexcept;
    std::size_t deviceBytes() const noexcept;
    MemoryProjection projectedTailCapacity(std::size_t const appended,
                                           std::size_t const maximum_capacity) const;
    view_type const& view() const noexcept;

    void clear() noexcept;

    void ensureTailCapacity(std::size_t const appended,
                            std::size_t const maximum_capacity,
                            TExecutionSpace const& execution);

    bool tryAppend(view_type const& source, std::size_t const count,
                   std::size_t const maximum_size, TExecutionSpace const& execution,
                   ResidentMemoryBudgetGate const memory_gate = {},
                   char const* const stage = "Kokkos pending-particle queue growth");

    /** Download a stable logical prefix without consuming it. */
    std::vector<ParticleState> downloadPrefix(std::size_t const count,
                                              TExecutionSpace const& execution) const;

    void consume(std::size_t const count);

  private:
    std::string label_{};
    view_type particles_{};
    std::size_t capacity_{};
    std::size_t head_{};
    std::size_t size_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/PendingParticleQueue.inl>
