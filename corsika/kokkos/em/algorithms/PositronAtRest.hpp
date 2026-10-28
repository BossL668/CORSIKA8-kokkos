/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/utility/PositronAtRest.hpp>
namespace corsika::kokkos {
  /** Compatibility entry for the shared production direction calculation. */
  CORSIKA_HOST_DEVICE inline void positron_at_rest_direction(double u, double azimuthU,
                                                             double* direction);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/algorithms/PositronAtRest.inl>
