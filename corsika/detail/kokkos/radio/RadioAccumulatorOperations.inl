/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio::accumulator_detail {
  KOKKOS_INLINE_FUNCTION unsigned long long RadioLocalCounterOperations::add(
      unsigned long long* address, unsigned long long value) {
    auto const previous = *address;
    *address += value;
    return previous;
  }

  KOKKOS_INLINE_FUNCTION long long RadioOpenMpThreadLocalOperations::add(
      long long* address, long long value) {
    // Each OpenMP worker owns a separate waveform row.  Preserve the same
    // two's-complement addition and previous-value overflow contract as the
    // atomic path without paying for a locked instruction on every sample.
    auto* const storage = reinterpret_cast<unsigned long long*>(address);
    auto const previous = *storage;
    *storage = previous + static_cast<unsigned long long>(value);
    return static_cast<long long>(previous);
  }

  KOKKOS_INLINE_FUNCTION unsigned long long RadioOpenMpThreadLocalOperations::add(
      unsigned long long* address, unsigned long long value) {
    auto const previous = *address;
    *address = previous + value;
    return previous;
  }

  KOKKOS_INLINE_FUNCTION double RadioOpenMpThreadLocalOperations::add(double* address,
                                                                      double value) {
    auto const previous = *address;
    *address = previous + value;
    return previous;
  }

  KOKKOS_INLINE_FUNCTION void RadioOpenMpThreadLocalOperations::maximum(double* address,
                                                                        double value) {
    if (value > *address) *address = value;
  }

  template <class TThreadView, class TOutputView>
  KOKKOS_INLINE_FUNCTION void MergeThreadWaveformKernel<TThreadView, TOutputView>::
  operator()(std::size_t const index, unsigned long long& local_overflows) const {
    long long sum{};
    for (std::size_t slot = 0; slot < threadSlots_; ++slot) {
      auto const increment = privateWaveform_(slot, index);
      auto const previous = RadioOpenMpThreadLocalOperations::add(&sum, increment);
      if ((increment > 0 && previous > SignedIntegerMaximum - increment) ||
          (increment < 0 && previous < SignedIntegerMinimum - increment))
        ++local_overflows;
    }
    waveform_(index) = sum;
  }
} // namespace corsika::kokkos::radio::accumulator_detail
