/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  KOKKOS_INLINE_FUNCTION long long ProfileAtomicOperations::add(long long* address,
                                                                long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
    // CUDA has a native unsigned 64-bit atomic add, while the signed
    // overload selected through Kokkos/desul is implemented as a CAS retry
    // loop.  Two's-complement signed addition has the same bit result as
    // unsigned addition, and the returned pre-add bit pattern preserves the
    // overflow check in checkedProfileAtomicAdd().  This deliberately
    // mirrors the established native-CUDA profile projection path.
    auto const previous = ::atomicAdd(reinterpret_cast<unsigned long long*>(address),
                                      static_cast<unsigned long long>(value));
    return static_cast<long long>(previous);
#else
    return Kokkos::atomic_fetch_add(address, value);
#endif
  }
  KOKKOS_INLINE_FUNCTION unsigned long long ProfileAtomicOperations::add(
      unsigned long long* address, unsigned long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
    return ::atomicAdd(address, value);
#else
    return Kokkos::atomic_fetch_add(address, value);
#endif
  }
  KOKKOS_INLINE_FUNCTION void ProfileAtomicOperations::maximum(
      unsigned long long* address, unsigned long long value) {
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDA_ARCH__)
    ::atomicMax(address, value);
#else
    Kokkos::atomic_max(address, value);
#endif
  }

} // namespace corsika::kokkos
