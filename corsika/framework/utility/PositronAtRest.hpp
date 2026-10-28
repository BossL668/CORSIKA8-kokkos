/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/framework/core/HostDevice.hpp>
#include <cmath>
namespace corsika {
  // Match original C8's energy-cut policy. This controls only extra photons
  // from a positron removed at the cut; in-flight annihilation is unchanged.
  inline constexpr bool EnableStoppedPositronAnnihilation = false;

  /** Isotropic back-to-back two-photon axis for a free, unpolarized pair at rest.
   * The caller supplies uniforms and accounts for the cut-endpoint kinetic
   * deposit and the target electron's rest-mass input. No positronium branch.
   */
  CORSIKA_HOST_DEVICE inline void positron_at_rest_direction(double u, double azimuthU,
                                                             double* direction);
} // namespace corsika
#include <corsika/detail/framework/utility/PositronAtRest.inl>
