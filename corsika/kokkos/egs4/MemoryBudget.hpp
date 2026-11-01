/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace corsika::kokkos::egs4::application {
  // Capacity planning only: no empty buffer is allocated to inflate GPU usage.
  // The queue still allocates actual live histories and fails without dropping
  // particles if its bounded capacity is exceeded. External allocations after
  // this snapshot can still make CUDA allocation fail.
  struct GpuMemoryBudget {
    double fraction_{};
    std::size_t totalBytes_{}, freeBytes_{}, requestedBytes_{}, workingBytes_{};
    std::size_t reserveBytes_{}, bytesPerHistory_{};
    int queueCapacity_{};
  };
  inline GpuMemoryBudget plan_gpu_memory(double fraction, std::size_t total,
                                         std::size_t free, std::size_t bytes_per_history,
                                         std::size_t reserve = 512ULL * 1024 * 1024);
} // namespace corsika::kokkos::egs4::application

#include <corsika/detail/kokkos/egs4/MemoryBudget.inl>
