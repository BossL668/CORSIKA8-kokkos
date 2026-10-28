/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/RadioAtomicOperations.hpp>
#include <corsika/kokkos/radio/RadioProjectionData.hpp>
#include <cstddef>

namespace corsika::kokkos::radio::accumulator_detail {
  struct RadioLocalCounterOperations : RadioAtomicOperations {
    using RadioAtomicOperations::add;

    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value);
  };
  struct RadioOpenMpThreadLocalOperations {
    KOKKOS_INLINE_FUNCTION static long long add(long long* address, long long value);

    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value);

    KOKKOS_INLINE_FUNCTION static double add(double* address, double value);

    KOKKOS_INLINE_FUNCTION static void maximum(double* address, double value);
  };
  template <class TThreadView, class TOutputView>
  struct MergeThreadWaveformKernel {
    TThreadView privateWaveform_;
    TOutputView waveform_;
    std::size_t threadSlots_;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const index,
                                           unsigned long long& local_overflows) const;
  };
} // namespace corsika::kokkos::radio::accumulator_detail

#include <corsika/detail/kokkos/radio/RadioAccumulatorOperations.inl>
