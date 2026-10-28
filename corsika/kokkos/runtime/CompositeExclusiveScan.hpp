/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#if __has_include(<cub/iterator/transform_input_iterator.cuh>)
#include <cub/iterator/transform_input_iterator.cuh>
#else
// CCCL 3 (CUDA 13) removed CUB's legacy transform input iterator.
#include <thrust/iterator/transform_iterator.h>
#endif
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace corsika::kokkos {

  /** Select the execution-specific implementation of a composite scan. */
  enum class CompositeScanImplementation {
    Automatic,
    Portable,
  };

  /**
   * One input item for the common packed uint32 count case.
   *
   * Callers with an existing POD count record do not need to copy it into
   * this type: exclusiveScan accepts any device-accessible one-dimensional
   * View together with a loader that returns CompositeScanValue<Columns>.
   */
  template <std::size_t NColumns>
  struct CompositeUint32Counts {
    static_assert(NColumns > 0);
    std::uint32_t values_[NColumns]{};
  };

  /** 64-bit prefix or total for one multi-output source record. */
  template <std::size_t NColumns>
  struct CompositeScanValue {
    static_assert(NColumns > 0);
    std::uint64_t values_[NColumns]{};
  };

  /** Convert the common packed uint32 POD to the 64-bit scan accumulator. */
  template <std::size_t NColumns>
  struct CompositeUint32Loader {
    KOKKOS_INLINE_FUNCTION CompositeScanValue<NColumns> operator()(
        CompositeUint32Counts<NColumns> const& input) const;
  };

  namespace composite_scan {

    template <std::size_t NColumns>
    struct Sum {
      KOKKOS_INLINE_FUNCTION CompositeScanValue<NColumns> operator()(
          CompositeScanValue<NColumns> const& left,
          CompositeScanValue<NColumns> const& right) const;
    };

    template <class TExecutionSpace, std::size_t NColumns, class TInputView,
              class TLoader>
    struct PortableScanFunctor {
      using value_type = CompositeScanValue<NColumns>;

      TInputView input_;
      Kokkos::View<value_type*, typename TExecutionSpace::memory_space> prefixes_;
      TLoader loader_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index, value_type& update,
                                             bool const final) const;
    };

#if defined(KOKKOS_ENABLE_CUDA)
    inline void check_cuda(cudaError_t const status, char const* const operation);

    template <std::size_t NColumns>
    inline std::size_t cub_temporary_bytes(std::size_t const count,
                                           cudaStream_t const stream);
#endif

  } // namespace composite_scan
  /**
   * Persistent temporary storage for a source-ordered composite scan.
   *
   * The output consists of one exclusive 64-bit prefix per input item and a
   * device-resident final total. Kokkos-CUDA uses CUB on the supplied Kokkos
   * execution-space stream; all other execution spaces use a portable
   * Kokkos::parallel_scan with exactly the same loader, prefix layout and
   * integer sums. Callers must wait on the supplied execution space before
   * reading results on the host or reusing storage from another stream.
   *
   * One workspace must not service overlapping scans because CUB reuses its
   * temporary byte array. It is intentionally grow-only so steady-state
   * wavefronts introduce no cudaMalloc/cudaFree operations.
   */

  template <class TExecutionSpace, std::size_t NColumns>
  class CompositeExclusiveScan {
  public:
    using execution_space_type = TExecutionSpace;
    using memory_space_type = typename execution_space_type::memory_space;
    using value_type = CompositeScanValue<NColumns>;
    using prefix_view_type = Kokkos::View<value_type*, memory_space_type>;
    using total_view_type = Kokkos::View<value_type, memory_space_type>;

    static_assert(NColumns == 7 || NColumns == 8,
                  "resident EM scans use exactly seven or eight columns");

    static constexpr bool cubAvailable() noexcept;

    std::size_t cubCapacity() const noexcept;

    std::size_t cubTemporaryBytes() const noexcept;

    bool usedCubForLastScan() const noexcept;

    /**
     * Enqueue source-ordered prefixes and a device total. Input storage must be
     * contiguous for the CUB path and remain alive until the scan completes.
     * The loader returns nonnegative counts whose sums fit in uint64_t.
     * Empty input writes a zero total without changing prefix entries.
     * @throws std::length_error if input/output extents or the CUB limit are exceeded.
     */
    template <class TInputView, class TLoader>
    void exclusiveScan(TInputView const& input, std::size_t const count,
                       prefix_view_type const& prefixes, total_view_type const& total,
                       TLoader loader, execution_space_type const& execution = {},
                       CompositeScanImplementation const implementation =
                           CompositeScanImplementation::Automatic);

  private:
#if defined(KOKKOS_ENABLE_CUDA)
    static std::size_t grownCapacity(std::size_t const current,
                                     std::size_t const requested);

    void ensureCubCapacity(std::size_t const requested,
                           execution_space_type const& execution);

    Kokkos::View<unsigned char*, memory_space_type> cubTemporary_{};
    std::size_t cubTemporaryBytes_{};
    std::size_t cubCapacity_{};
#endif
    bool lastScanUsedCub_{};
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/CompositeExclusiveScan.inl>
