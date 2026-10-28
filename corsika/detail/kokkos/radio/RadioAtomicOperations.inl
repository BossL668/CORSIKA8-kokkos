/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  KOKKOS_INLINE_FUNCTION long long RadioAtomicOperations::add(long long* address,
                                                              long long value) {
    // Radio fixed-point waveforms execute this fetch-add billions of times.
    // Kokkos/desul implements the returning 64-bit overload with a CAS loop
    // on CUDA, whereas the production native-CUDA path emits one hardware
    // ATOMG.ADD.64.  Use the backend intrinsic in device compilation and
    // retain Kokkos' portable implementation for OpenMP.  Two's-complement
    // modular addition has the same bit result for signed and unsigned
    // storage, and the caller performs its existing overflow check on the
    // returned bits.
    auto* const storage = reinterpret_cast<unsigned long long*>(address);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    auto const previous = atomicAdd(storage, static_cast<unsigned long long>(value));
#else
    auto const previous =
        Kokkos::atomic_fetch_add(storage, static_cast<unsigned long long>(value));
#endif
    return static_cast<long long>(previous);
  }
  KOKKOS_INLINE_FUNCTION unsigned long long RadioAtomicOperations::add(
      unsigned long long* address, unsigned long long value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return atomicAdd(address, value);
#else
    return Kokkos::atomic_fetch_add(address, value);
#endif
  }
  KOKKOS_INLINE_FUNCTION double RadioAtomicOperations::add(double* address,
                                                           double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return atomicAdd(address, value);
#else
    return Kokkos::atomic_fetch_add(address, value);
#endif
  }
  KOKKOS_INLINE_FUNCTION void RadioAtomicOperations::maximum(double* address,
                                                             double value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // Diagnostic maxima are finite and non-negative, so IEEE-754 bit order
    // is monotonic and matches the native CUDA implementation.
    auto* const bits = reinterpret_cast<unsigned long long*>(address);
    atomicMax(bits, __double_as_longlong(value));
#else
    Kokkos::atomic_max(address, value);
#endif
  }
} // namespace corsika::kokkos::radio
