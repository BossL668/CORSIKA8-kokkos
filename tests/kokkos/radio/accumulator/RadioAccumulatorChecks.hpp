/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <catch2/catch_all.hpp>
#include <corsika/kokkos/radio/RadioAccumulator.hpp>
#include <corsika/kokkos/runtime/Runtime.hpp>
#include <cmath>
#include <type_traits>

namespace radio_accumulator_test {
  namespace cp = corsika::kokkos;
  namespace cr = cp::radio;
  using execution_type = Kokkos::DefaultExecutionSpace;
  using accumulator_type = cr::RadioAccumulator<execution_type>;
  using record_type = cp::LeptonTransportRecord;
  inline void ensure_runtime() { static cp::Runtime runtime; }
  inline bool close(double a, double b, double floor = 1.e-28) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= floor + 2.e-12 * std::max(std::abs(a), std::abs(b));
  }
} // namespace radio_accumulator_test

#if defined(CORSIKA_RADIO_EXPECT_CUDA)
static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>);
#elif defined(CORSIKA_RADIO_EXPECT_OPENMP)
static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>);
#endif
