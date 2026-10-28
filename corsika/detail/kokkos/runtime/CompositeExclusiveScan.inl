/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <std::size_t NColumns>
  KOKKOS_INLINE_FUNCTION CompositeScanValue<NColumns> CompositeUint32Loader<NColumns>::
  operator()(CompositeUint32Counts<NColumns> const& input) const {
    CompositeScanValue<NColumns> result{};
    for (std::size_t column = 0; column < NColumns; ++column)
      result.values_[column] = input.values_[column];
    return result;
  }
  namespace composite_scan {
    template <std::size_t NColumns>
    KOKKOS_INLINE_FUNCTION CompositeScanValue<NColumns> Sum<NColumns>::operator()(
        CompositeScanValue<NColumns> const& left,
        CompositeScanValue<NColumns> const& right) const {
      CompositeScanValue<NColumns> result{};
      for (std::size_t column = 0; column < NColumns; ++column)
        result.values_[column] = left.values_[column] + right.values_[column];
      return result;
    }

    template <class TExecutionSpace, std::size_t NColumns, class TInputView,
              class TLoader>
    KOKKOS_INLINE_FUNCTION void
    PortableScanFunctor<TExecutionSpace, NColumns, TInputView, TLoader>::init(
        value_type& value) const {
      value = {};
    }

    template <class TExecutionSpace, std::size_t NColumns, class TInputView,
              class TLoader>
    KOKKOS_INLINE_FUNCTION void
    PortableScanFunctor<TExecutionSpace, NColumns, TInputView, TLoader>::join(
        value_type& destination, value_type const& source) const {
      destination = Sum<NColumns>{}(destination, source);
    }

    template <class TExecutionSpace, std::size_t NColumns, class TInputView,
              class TLoader>
    KOKKOS_INLINE_FUNCTION void
    PortableScanFunctor<TExecutionSpace, NColumns, TInputView, TLoader>::operator()(
        std::size_t const index, value_type& update, bool const final) const {
      if (final) prefixes_(index) = update;
      update = Sum<NColumns>{}(update, loader_(input_(index)));
    }

#if defined(KOKKOS_ENABLE_CUDA)
    inline void check_cuda(cudaError_t const status, char const* const operation) {
      if (status == cudaSuccess) return;
      throw std::runtime_error(std::string(operation) +
                               " failed: " + cudaGetErrorString(status));
    }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
    template <std::size_t NColumns>
    inline std::size_t cub_temporary_bytes(std::size_t const count,
                                           cudaStream_t const stream) {
      if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("Kokkos-CUDA composite scan exceeds the CUB item limit");
      std::size_t bytes{};
      check_cuda(
          cub::DeviceScan::ExclusiveScan(
              nullptr, bytes, static_cast<CompositeScanValue<NColumns> const*>(nullptr),
              static_cast<CompositeScanValue<NColumns>*>(nullptr), Sum<NColumns>{},
              CompositeScanValue<NColumns>{}, static_cast<int>(count), stream),
          "query Kokkos-CUDA composite-scan temporary storage");
      return bytes;
    }
#endif
  } // namespace composite_scan
  template <class TExecutionSpace, std::size_t NColumns>
  constexpr bool
  CompositeExclusiveScan<TExecutionSpace, NColumns>::cubAvailable() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    return std::is_same_v<execution_space_type, Kokkos::Cuda>;
#else
    return false;
#endif
  }

  template <class TExecutionSpace, std::size_t NColumns>
  std::size_t CompositeExclusiveScan<TExecutionSpace, NColumns>::cubCapacity() const
      noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    return cubCapacity_;
#else
    return 0;
#endif
  }

  template <class TExecutionSpace, std::size_t NColumns>
  std::size_t CompositeExclusiveScan<TExecutionSpace, NColumns>::cubTemporaryBytes() const
      noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    return cubTemporaryBytes_;
#else
    return 0;
#endif
  }

  template <class TExecutionSpace, std::size_t NColumns>
  bool CompositeExclusiveScan<TExecutionSpace, NColumns>::usedCubForLastScan() const
      noexcept {
    return lastScanUsedCub_;
  }

  template <class TExecutionSpace, std::size_t NColumns>
  template <class TInputView, class TLoader>
  void CompositeExclusiveScan<TExecutionSpace, NColumns>::exclusiveScan(
      TInputView const& input, std::size_t const count, prefix_view_type const& prefixes,
      total_view_type const& total, TLoader loader, execution_space_type const& execution,
      CompositeScanImplementation const implementation) {
    static_assert(TInputView::rank == 1, "composite scan input must be one-dimensional");
    static_assert(std::is_same_v<typename TInputView::memory_space, memory_space_type>,
                  "composite scan input must use the execution memory space");

    if (count > input.extent(0))
      throw std::length_error("Kokkos composite scan exceeds the input extent");
    if (count > prefixes.extent(0))
      throw std::length_error("Kokkos composite scan exceeds the prefix-output extent");

    lastScanUsedCub_ = false;
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
        using input_pointer_type = decltype(input.data());
#if __has_include(<cub/iterator/transform_input_iterator.cuh>)
        using transform_iterator_type =
            cub::TransformInputIterator<value_type, TLoader, input_pointer_type>;
#else
        // Preserve value-returning loads and the same 64-bit prefix sum.
        using transform_iterator_type =
            thrust::transform_iterator<TLoader, input_pointer_type, value_type,
                                       value_type>;
#endif
        transform_iterator_type transformed(input.data(), loader);
        auto bytes = cubTemporaryBytes_;
        composite_scan::check_cuda(
            cub::DeviceScan::ExclusiveScan(
                cubTemporary_.data(), bytes, transformed, prefixes.data(),
                composite_scan::Sum<NColumns>{}, value_type{}, static_cast<int>(count),
                execution.cuda_stream()),
            "run Kokkos-CUDA composite exclusive scan");

        auto const total_kernel = KOKKOS_LAMBDA(std::size_t const) {
          total() = composite_scan::Sum<NColumns>{}(prefixes(count - 1),
                                                    loader(input(count - 1)));
        };
        static_assert(sizeof(total_kernel) < 512);
        Kokkos::parallel_for("c8_kokkos_composite_scan_total",
                             Kokkos::RangePolicy<execution_space_type>(execution, 0, 1),
                             total_kernel);
        lastScanUsedCub_ = true;
        return;
      }
    }
#endif

    Kokkos::parallel_scan(
        "c8_kokkos_composite_exclusive_scan",
        Kokkos::RangePolicy<execution_space_type>(execution, 0, count),
        composite_scan::PortableScanFunctor<execution_space_type, NColumns, TInputView,
                                            TLoader>{input, prefixes, loader},
        total);
  }

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace, std::size_t NColumns>
  std::size_t CompositeExclusiveScan<TExecutionSpace, NColumns>::grownCapacity(
      std::size_t const current, std::size_t const requested) {
    if (current == 0) return requested;
    if (current > std::numeric_limits<std::size_t>::max() / 2) return requested;
    return std::max(requested, current * 2);
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  template <class TExecutionSpace, std::size_t NColumns>
  void CompositeExclusiveScan<TExecutionSpace, NColumns>::ensureCubCapacity(
      std::size_t const requested, execution_space_type const& execution) {
    if constexpr (cubAvailable()) {
      if (requested <= cubCapacity_) return;
      auto const grown = grownCapacity(cubCapacity_, requested);
      auto const required =
          composite_scan::cub_temporary_bytes<NColumns>(grown, execution.cuda_stream());
      if (required > cubTemporary_.extent(0)) {
        cubTemporary_ = decltype(cubTemporary_)(
            Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                               "c8_kokkos_composite_scan_temporary"),
            required);
      }
      cubTemporaryBytes_ = required;
      cubCapacity_ = grown;
    }
  }
#endif

} // namespace corsika::kokkos
