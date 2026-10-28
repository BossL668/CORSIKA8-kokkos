/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
namespace corsika {
  CORSIKA_HOST_DEVICE inline void positron_at_rest_direction(double u, double azimuthU,
                                                             double* direction) {
    auto const cosine = 2. * u - 1.;
    auto const sine = ::sqrt((1. - cosine) * (1. + cosine));
    auto const phi = 6.283185307179586476925286766559 * azimuthU;
    direction[0] = sine * ::cos(phi);
    direction[1] = sine * ::sin(phi);
    direction[2] = cosine;
  }
} // namespace corsika
