/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

namespace corsika::kokkos {

  /** Production atomic backend, including the CUDA signed-sum bit convention. */
  struct ProfileAtomicOperations {
    KOKKOS_INLINE_FUNCTION static long long add(long long* address, long long value);
    KOKKOS_INLINE_FUNCTION static unsigned long long add(unsigned long long* address,
                                                         unsigned long long value);
    KOKKOS_INLINE_FUNCTION static void maximum(unsigned long long* address,
                                               unsigned long long value);
  };

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/ProfileAtomicOperations.inl>
