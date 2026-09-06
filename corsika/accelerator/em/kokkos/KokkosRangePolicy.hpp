/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>

namespace corsika::accelerator::em::kokkos_detail {

  /**
   * Construct a production RangePolicy while keeping OpenMP scheduling
   * tuning out of GPU execution spaces.
   *
   * Kokkos' chunk size controls how a host range is partitioned among the
   * OpenMP workers.  Applying the same setting to CUDA/HIP/SYCL changes their
   * launch policy and can regress a device-specific optimum, so the value is
   * intentionally ignored unless the selected execution space is OpenMP.
   */
  template <class ExecutionSpace>
  Kokkos::RangePolicy<ExecutionSpace> makeKokkosRangePolicy(
      ExecutionSpace const& execution, std::size_t const count,
      std::size_t const openmp_chunk_size = 0) {
    Kokkos::RangePolicy<ExecutionSpace> policy(execution, 0, count);
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
      if (openmp_chunk_size != 0)
        policy.set_chunk_size(openmp_chunk_size);
    } else {
      (void)openmp_chunk_size;
    }
#else
    (void)openmp_chunk_size;
#endif
    return policy;
  }

} // namespace corsika::accelerator::em::kokkos_detail
