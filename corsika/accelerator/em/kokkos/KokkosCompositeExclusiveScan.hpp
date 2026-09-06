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

namespace corsika::accelerator::em::kokkos_detail {

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
  template <std::size_t Columns>
  struct CompositeUint32Counts {
    static_assert(Columns > 0);
    std::uint32_t values[Columns]{};
  };

  /** 64-bit prefix or total for one multi-output source record. */
  template <std::size_t Columns>
  struct CompositeScanValue {
    static_assert(Columns > 0);
    std::uint64_t values[Columns]{};
  };

  /** Convert the common packed uint32 POD to the 64-bit scan accumulator. */
  template <std::size_t Columns>
  struct CompositeUint32Loader {
    KOKKOS_INLINE_FUNCTION CompositeScanValue<Columns> operator()(
        CompositeUint32Counts<Columns> const& input) const {
      CompositeScanValue<Columns> result{};
      for (std::size_t column = 0; column < Columns; ++column)
        result.values[column] = input.values[column];
      return result;
    }
  };

  namespace composite_scan_detail {

    template <std::size_t Columns>
    struct Sum {
      KOKKOS_INLINE_FUNCTION CompositeScanValue<Columns> operator()(
          CompositeScanValue<Columns> const& left,
          CompositeScanValue<Columns> const& right) const {
        CompositeScanValue<Columns> result{};
        for (std::size_t column = 0; column < Columns; ++column)
          result.values[column] =
              left.values[column] + right.values[column];
        return result;
      }
    };

    template <class ExecutionSpace, std::size_t Columns, class InputView,
              class Loader>
    struct PortableScanFunctor {
      using value_type = CompositeScanValue<Columns>;

      InputView input;
      Kokkos::View<value_type*, typename ExecutionSpace::memory_space>
          prefixes;
      Loader loader;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
        value = {};
      }

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const {
        destination = Sum<Columns>{}(destination, source);
      }

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index,
                                             value_type& update,
                                             bool const final) const {
        if (final) prefixes(index) = update;
        update = Sum<Columns>{}(update, loader(input(index)));
      }
    };

#if defined(KOKKOS_ENABLE_CUDA)
    inline void checkCuda(cudaError_t const status,
                          char const* const operation) {
      if (status == cudaSuccess) return;
      throw std::runtime_error(std::string(operation) + " failed: " +
                               cudaGetErrorString(status));
    }

    template <std::size_t Columns>
    inline std::size_t cubTemporaryBytes(std::size_t const count,
                                         cudaStream_t const stream) {
      if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Kokkos-CUDA composite scan exceeds the CUB item limit");
      std::size_t bytes{};
      checkCuda(
          cub::DeviceScan::ExclusiveScan(
              nullptr, bytes,
              static_cast<CompositeScanValue<Columns> const*>(nullptr),
              static_cast<CompositeScanValue<Columns>*>(nullptr),
              Sum<Columns>{}, CompositeScanValue<Columns>{},
              static_cast<int>(count), stream),
          "query Kokkos-CUDA composite-scan temporary storage");
      return bytes;
    }
#endif

  } // namespace composite_scan_detail

  /**
   * Persistent temporary storage for a source-ordered composite scan.
   *
   * The output consists of one exclusive 64-bit prefix per input item and a
   * device-resident final total. Kokkos-CUDA uses CUB on the supplied Kokkos
   * execution-space stream; all other execution spaces use a portable
   * Kokkos::parallel_scan with exactly the same loader, prefix layout and
   * addition order. No host synchronization is performed by this class.
   *
   * One workspace must not service overlapping scans because CUB reuses its
   * temporary byte array. It is intentionally grow-only so steady-state
   * wavefronts introduce no cudaMalloc/cudaFree operations.
   */
  template <class ExecutionSpace, std::size_t Columns>
  class KokkosCompositeExclusiveScanWorkspace {
  public:
    using execution_space = ExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using value_type = CompositeScanValue<Columns>;
    using PrefixView = Kokkos::View<value_type*, memory_space>;
    using TotalView = Kokkos::View<value_type, memory_space>;

    static_assert(Columns == 7 || Columns == 8,
                  "resident EM scans use exactly seven or eight columns");

    static constexpr bool cubAvailable() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
      return std::is_same_v<execution_space, Kokkos::Cuda>;
#else
      return false;
#endif
    }

    std::size_t cubCapacity() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
      return cub_capacity_;
#else
      return 0;
#endif
    }

    std::size_t cubTemporaryBytes() const noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
      return cub_temporary_bytes_;
#else
      return 0;
#endif
    }

    bool usedCubForLastScan() const noexcept { return last_scan_used_cub_; }

    template <class InputView, class Loader>
    void exclusiveScan(
        InputView const& input, std::size_t const count,
        PrefixView const& prefixes, TotalView const& total, Loader loader,
        execution_space const& execution = {},
        CompositeScanImplementation const implementation =
            CompositeScanImplementation::Automatic) {
      static_assert(InputView::rank == 1,
                    "composite scan input must be one-dimensional");
      static_assert(
          std::is_same_v<typename InputView::memory_space, memory_space>,
          "composite scan input must use the execution memory space");

      if (count > input.extent(0))
        throw std::length_error(
            "Kokkos composite scan exceeds the input extent");
      if (count > prefixes.extent(0))
        throw std::length_error(
            "Kokkos composite scan exceeds the prefix-output extent");

      last_scan_used_cub_ = false;
#if !defined(KOKKOS_ENABLE_CUDA)
      (void)implementation;
#endif
      if (count == 0) {
        Kokkos::deep_copy(execution, total, value_type{});
        return;
      }

#if defined(KOKKOS_ENABLE_CUDA)
      if constexpr (cubAvailable()) {
        if (implementation == CompositeScanImplementation::Automatic) {
          ensureCubCapacity(count, execution);
          using InputPointer = decltype(input.data());
#if __has_include(<cub/iterator/transform_input_iterator.cuh>)
          using TransformIterator = cub::TransformInputIterator<
              value_type, Loader, InputPointer>;
#else
          // Preserve value-returning loads and the same 64-bit prefix sum.
          using TransformIterator = thrust::transform_iterator<
              Loader, InputPointer, value_type, value_type>;
#endif
          TransformIterator transformed(input.data(), loader);
          auto bytes = cub_temporary_bytes_;
          composite_scan_detail::checkCuda(
              cub::DeviceScan::ExclusiveScan(
                  cub_temporary_.data(), bytes, transformed, prefixes.data(),
                  composite_scan_detail::Sum<Columns>{}, value_type{},
                  static_cast<int>(count), execution.cuda_stream()),
              "run Kokkos-CUDA composite exclusive scan");

          auto const total_kernel = KOKKOS_LAMBDA(std::size_t const) {
            total() = composite_scan_detail::Sum<Columns>{}(
                prefixes(count - 1), loader(input(count - 1)));
          };
          static_assert(sizeof(total_kernel) < 512);
          Kokkos::parallel_for(
              "c8_kokkos_composite_scan_total",
              Kokkos::RangePolicy<execution_space>(execution, 0, 1),
              total_kernel);
          last_scan_used_cub_ = true;
          return;
        }
      }
#endif

      Kokkos::parallel_scan(
          "c8_kokkos_composite_exclusive_scan",
          Kokkos::RangePolicy<execution_space>(execution, 0, count),
          composite_scan_detail::PortableScanFunctor<
              execution_space, Columns, InputView, Loader>{input, prefixes,
                                                           loader},
          total);
    }

  private:
#if defined(KOKKOS_ENABLE_CUDA)
    static std::size_t grownCapacity(std::size_t const current,
                                     std::size_t const requested) {
      if (current == 0) return requested;
      if (current > std::numeric_limits<std::size_t>::max() / 2)
        return requested;
      return std::max(requested, current * 2);
    }

    void ensureCubCapacity(std::size_t const requested,
                           execution_space const& execution) {
      if constexpr (cubAvailable()) {
        if (requested <= cub_capacity_) return;
        auto const grown = grownCapacity(cub_capacity_, requested);
        auto const required =
            composite_scan_detail::cubTemporaryBytes<Columns>(
                grown, execution.cuda_stream());
        if (required > cub_temporary_.extent(0)) {
          cub_temporary_ = decltype(cub_temporary_)(
              Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                                 "c8_kokkos_composite_scan_temporary"),
              required);
        }
        cub_temporary_bytes_ = required;
        cub_capacity_ = grown;
      }
    }

    Kokkos::View<unsigned char*, memory_space> cub_temporary_{};
    std::size_t cub_temporary_bytes_{};
    std::size_t cub_capacity_{};
#endif
    bool last_scan_used_cub_{};
  };

} // namespace corsika::accelerator::em::kokkos_detail
