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
#include <corsika/kokkos/data/ParticleSoA.hpp>
#include <corsika/kokkos/runtime/GeometricCapacity.hpp>
#include <corsika/kokkos/runtime/HostStagingSpace.hpp>
#include <corsika/kokkos/runtime/ResidentMemoryBudget.hpp>
#include <corsika/kokkos/runtime/ResidentExecutionWait.hpp>

namespace corsika::kokkos {

  template <class TExecutionSpace>
  struct QueueScanFunctor {
    using memory_space_type = typename TExecutionSpace::memory_space;
    Kokkos::View<std::uint32_t const*, memory_space_type> flags_;
    Kokkos::View<std::uint64_t*, memory_space_type> offsets_;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index, std::uint64_t& update,
                                           bool const final) const;
  };
  template <class TExecutionSpace>
  struct QueueScatterFunctor {
    ParticleSoARawView input_;
    ParticleSoARawView output_;
    std::uint32_t const* flags_{};
    std::uint64_t const* offsets_{};

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index) const;
  };
  template <class TExecutionSpace>
  struct CountScanFunctor {
    using memory_space_type = typename TExecutionSpace::memory_space;
    Kokkos::View<std::uint32_t const*, memory_space_type> counts_;
    Kokkos::View<std::uint64_t*, memory_space_type> offsets_;

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index, std::uint64_t& update,
                                           bool const final) const;
  };
  /** Double-buffered particle storage and persistent host staging.
   * The bound execution instance owns pending work. Growing storage resets
   * the logical queue; requests within retained capacity preserve it. */
  template <class TExecutionSpace>
  class WavefrontQueue {
  public:
    using execution_space_type = TExecutionSpace;
    using memory_space_type = typename TExecutionSpace::memory_space;
    using flag_view_type = Kokkos::View<std::uint32_t*, memory_space_type>;
    using offset_view_type = Kokkos::View<std::uint64_t*, memory_space_type>;
    using particle_view_type = Kokkos::View<ParticleState*, memory_space_type>;
    using host_staging_space_type =
        corsika::kokkos::host_staging_space_type<TExecutionSpace>;
    using host_staging_view_type = Kokkos::View<ParticleState*, host_staging_space_type>;

    static_assert(std::is_trivially_copyable_v<ParticleState>,
                  "Kokkos wavefront AoS staging requires a trivially-copyable particle");

    WavefrontQueue() = default;

    explicit WavefrontQueue(std::size_t const capacity);

    WavefrontQueue(std::size_t const capacity, TExecutionSpace const& execution);

    /**
     * Grow the device allocation and reset the current logical queue.
     *
     * Resident cascade workspaces call this at a synchronized host wavefront
     * boundary and then upload a new queue. Growing discards any current
     * contents and sets size() to zero. The allocation grows geometrically and
     * may therefore exceed the requested logical checkpoint; physics-side
     * overflow tests continue to use that unchanged request. Capacity never
     * shrinks, so calls at or below the retained capacity leave both the
     * allocation and logical queue unchanged.
     */
    void ensureCapacity(std::size_t const requested);

    void ensureCapacity(std::size_t const requested, TExecutionSpace const& execution);

    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    bool empty() const noexcept;
    std::size_t deviceBytes() const noexcept;
    MemoryProjection projectedCapacity(std::size_t const requested) const;
    std::size_t hostPinnedBytes() const noexcept;
    ParticleSoA<memory_space_type> const& current() const noexcept;
    ParticleSoA<memory_space_type>& next() noexcept;
    ParticleSoA<memory_space_type> const& next() const noexcept;

    void commitNext(std::size_t const next_size);

    /**
     * Swap fronts after the caller has already established execution-space
     * ordering.  Resident cascades fence immediately after materialization
     * and submit all later profile/queue work to the same execution-space
     * instance, so another unconditional device-wide fence at the host-only
     * pointer swap only serializes the next wavefront.
     */
    void commitNextAlreadySynchronized(std::size_t const next_size);

    void clear();

  private:
    void validateNextSize(std::size_t const next_size) const;

  public:
    /** Mark all queue work on the bound execution instance complete. */
    void markExecutionSynchronized() noexcept;

    void upload(std::vector<ParticleState> const& particles);

    void upload(std::vector<ParticleState> const& particles,
                TExecutionSpace const& execution);

    void upload(std::vector<ParticleState> const& particles,
                particle_view_type const& prefix, std::size_t const prefix_head,
                std::size_t const prefix_count, TExecutionSpace const& execution);

    std::vector<ParticleState> download() const;

    std::vector<ParticleState> download(
        TExecutionSpace const& execution,
        ResidentExecutionWait<TExecutionSpace>* const blocking_wait = nullptr) const;

    std::size_t compact(Kokkos::View<std::uint32_t const*, memory_space_type> flags);

    std::uint64_t secondaryOffsets(
        Kokkos::View<std::uint32_t const*, memory_space_type> counts);

    offset_view_type const& offsets() const noexcept;

  private:
    TExecutionSpace execution_{};
    std::size_t capacity_{};
    std::size_t size_{};
    ParticleSoA<memory_space_type> current_{};
    ParticleSoA<memory_space_type> next_{};
    offset_view_type offsets_{};
    particle_view_type aosStaging_{};
    host_staging_view_type hostStaging_{};
    // False means work touching current_, staging, or the pinned host buffer
    // is still ordered on execution_.  Rebinding/reallocation is forbidden
    // until the owning execution instance has been fenced.
    mutable bool executionSynchronized_{true};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/WavefrontQueue.inl>
