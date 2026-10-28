/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace corsika::kokkos {
  // SharedHostPinnedSpace follows the package's default device. In a dual
  // package OpenMP must still use ordinary RAM, not CUDA-pinned staging memory.
  template <class TExecutionSpace>
  using host_staging_space_type = std::conditional_t<
      std::is_same_v<typename TExecutionSpace::memory_space, Kokkos::HostSpace>,
      Kokkos::HostSpace, Kokkos::SharedHostPinnedSpace>;

} // namespace corsika::kokkos
