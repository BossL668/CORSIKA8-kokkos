/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  /** Evaluate log(1 + value), retaining the sign of a zero argument. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline double logarithm_one_plus(double value);

  /** Evaluate exp(value) - 1, retaining the sign of a zero argument. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline double exponential_minus_one(double value);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/utility/ElementaryFunctions.inl>
