/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <corsika/kokkos/data/ParticleSoA.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/runtime/RangePolicy.hpp>
#include <corsika/kokkos/runtime/ResidentMemoryBudget.hpp>

namespace corsika::kokkos {

  namespace wavefront_bucketing_detail {
    struct BucketOrder {
      std::uint64_t bucketKey_{};
      std::uint32_t source_{};
      std::uint32_t reserved_{};
    };
    struct BucketOrderLess {
      KOKKOS_INLINE_FUNCTION bool operator()(BucketOrder const& left,
                                             BucketOrder const& right) const;
    };
  } // namespace wavefront_bucketing_detail

  /** Number of significant bits in the PID x medium x energy bucket key. */
  inline constexpr int WavefrontBucketKeyBits = 50;

  /**
   * Below this size, stable radix sorting costs more than it saves in the
   * resident electromagnetic transport kernels.
   */
  inline constexpr std::size_t MinimumWavefrontRadixSortSize = 256;

  inline constexpr std::size_t MinimumKokkosCudaRadixSortSize =
      MinimumWavefrontRadixSortSize;

  /** Production 50-bit key ordered by PID, signed medium and energy bin. */
  KOKKOS_INLINE_FUNCTION std::uint64_t wavefront_bucket_key(
      ParticleState const& particle);

  /** Retained particle/key views; sorted contents may be overwritten on workspace reuse.
   */
  template <class TExecutionSpace>
  struct WavefrontBucketBatch {
    using memory_space_type = typename TExecutionSpace::memory_space;

    ParticleSoA<memory_space_type> particles_{};
    Kokkos::View<std::uint64_t*, memory_space_type> keys_{};
    std::size_t count_{};
    bool sorted_{};
  };

  /** Reuse grow-only sorting arrays on one execution instance. */
  template <class TExecutionSpace>
  class WavefrontBucketingWorkspace {
  public:
    using memory_space_type = typename TExecutionSpace::memory_space;
    using order_type = wavefront_bucketing_detail::BucketOrder;

    static constexpr bool cudaRadixSortAvailable() noexcept;

    static constexpr bool usesCudaRadixSort(std::size_t const count) noexcept;

    void ensureCapacity(std::size_t const requested,
                        TExecutionSpace const& execution = {});

    std::size_t capacity() const noexcept;
    std::size_t portableCapacity() const noexcept;
    std::size_t radixCapacity() const noexcept;
    std::size_t deviceBytes() const noexcept;

    /**
     * Predict the retained and allocation-overlap bytes of ensureCapacity().
     *
     * The projection follows the exact allocation order below.  In
     * particular, each newly constructed Kokkos::View coexists briefly with
     * the member it replaces.  Counting only the final retained size is not
     * sufficient to prevent an allocator OOM during geometric growth.
     */
    MemoryProjection projectedCapacity(std::size_t const requested,
                                       TExecutionSpace const& execution = {}) const;
    // A reserved arena must also cover the portable small-front path used
    // below the CUDA radix threshold, even when its largest front uses CUB.
    void ensureCountRange(std::size_t const maximumCount,
                          TExecutionSpace const& execution);

    MemoryProjection projectedCountRange(std::size_t const maximumCount,
                                         TExecutionSpace const& execution) const;

    ParticleSoA<memory_space_type> const& output() const noexcept;
    Kokkos::View<std::uint64_t*, memory_space_type> const& keys() const noexcept;
    Kokkos::View<order_type*, memory_space_type> const& order() const noexcept;
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::View<std::uint64_t*, memory_space_type> const& inputKeys() const noexcept;
    Kokkos::View<std::uint32_t*, memory_space_type> const& inputIndices() const noexcept;
    Kokkos::View<std::uint32_t*, memory_space_type> const& outputIndices() const noexcept;
    Kokkos::View<unsigned char*, memory_space_type> const& radixTemporary() const
        noexcept;
    std::size_t radixTemporaryBytes() const noexcept;
#endif

  private:
    static std::size_t grownCapacity(std::size_t const current,
                                     std::size_t const requested);

    void ensureOutputCapacity(std::size_t const requested,
                              TExecutionSpace const& execution);

    void ensurePortableCapacity(std::size_t const requested,
                                TExecutionSpace const& execution);

#if defined(KOKKOS_ENABLE_CUDA)
    void ensureRadixCapacity(std::size_t const requested,
                             TExecutionSpace const& execution);
#endif

    ParticleSoA<memory_space_type> output_{};
    Kokkos::View<std::uint64_t*, memory_space_type> keys_{};
    Kokkos::View<order_type*, memory_space_type> order_{};
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::View<std::uint64_t*, memory_space_type> inputKeys_{};
    Kokkos::View<std::uint32_t*, memory_space_type> inputIndices_{};
    Kokkos::View<std::uint32_t*, memory_space_type> outputIndices_{};
    Kokkos::View<unsigned char*, memory_space_type> radixTemporary_{};
    std::size_t radixTemporaryBytes_{};
    std::size_t radixCapacity_{};
#endif
    std::size_t outputCapacity_{};
    std::size_t portableCapacity_{};
  };

  /** Group a front with the production stable ordering and no-sort threshold.
   * The input must not alias workspace output. Owners must remain alive and
   * unchanged until submitted work completes. Equal keys retain source order;
   * below the threshold the returned particle views alias the input.
   */
  template <class TExecutionSpace>
  WavefrontBucketBatch<TExecutionSpace> bucket_wavefront(
      ParticleSoA<typename TExecutionSpace::memory_space> const& input,
      std::size_t const count, WavefrontBucketingWorkspace<TExecutionSpace>& workspace,
      TExecutionSpace const& execution = {}, std::size_t const openmpChunkSize = 0);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/scheduling/WavefrontBucketing.inl>
