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

namespace corsika::kokkos {

  /** Construct the production range, applying chunk size only to OpenMP. */
  template <class TExecutionSpace>
  Kokkos::RangePolicy<TExecutionSpace> make_range_policy(
      TExecutionSpace const& execution, std::size_t const count,
      std::size_t const openmp_chunk_size = 0);

  /** Construct a bounded range for independently evaluated physics records.
   *
   * HIP uses the compile-time block and residency limits. Other execution spaces
   * preserve the ordinary range policy, including the OpenMP chunk size.
   * The limits affect launch configuration, not particle ordering or arithmetic.
   */
  template <unsigned int MaximumThreads, unsigned int MinimumBlocks,
            class TExecutionSpace>
  auto make_bounded_range_policy(TExecutionSpace const& execution,
                                 std::size_t const count,
                                 std::size_t const openmp_chunk_size = 0);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/RangePolicy.inl>
