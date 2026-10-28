/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <corsika/kokkos/runtime/TuningCache.hpp>
#include <stdexcept>

std::uint64_t runtime_other_translation_unit(
    corsika::kokkos::BackendRuntime const& runtime) {
  auto result = runtime.runPrimitiveProbe(17, 1);
  if (!result.scanValid_ || result.selected_ != 11 || result.checksum_ != 96)
    throw std::runtime_error("Independent-TU primitive probe disagrees");
  return result.checksum_;
}

std::size_t cache_other_translation_unit() {
  return corsika::kokkos::TuningParameters{}.deviceQueues_;
}
