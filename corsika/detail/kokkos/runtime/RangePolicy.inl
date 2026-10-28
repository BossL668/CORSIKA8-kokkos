/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  template <class TExecutionSpace>
  Kokkos::RangePolicy<TExecutionSpace> make_range_policy(
      TExecutionSpace const& execution, std::size_t const count,
      std::size_t const openmp_chunk_size) {
    Kokkos::RangePolicy<TExecutionSpace> policy(execution, 0, count);
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::is_same_v<TExecutionSpace, Kokkos::OpenMP>) {
      if (openmp_chunk_size != 0) policy.set_chunk_size(openmp_chunk_size);
    } else {
      (void)openmp_chunk_size;
    }
#else
    (void)openmp_chunk_size;
#endif
    return policy;
  }

} // namespace corsika::kokkos
