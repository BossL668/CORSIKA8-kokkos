/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika::kokkos {
  CORSIKA_HOST_DEVICE inline void positron_at_rest_direction(double u, double azimuthU,
                                                             double* direction) {
    ::corsika::positron_at_rest_direction(u, azimuthU, direction);
  }
} // namespace corsika::kokkos
