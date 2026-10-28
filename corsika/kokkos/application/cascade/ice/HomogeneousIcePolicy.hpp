/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/media/MediumProperties.hpp>
#include <stdexcept>

namespace corsika::kokkos::application::ice {
  /** Validate the uniform ice scene and return its common refractive index.
   * The application constructs one convex ice volume. Exported calculators must
   * come from that same material; this check does not manufacture any tables.
   */
  template <typename TSnapshot>
  double require_homogeneous_ice(TSnapshot const& snapshot);
} // namespace corsika::kokkos::application::ice

#include <corsika/detail/kokkos/application/cascade/ice/HomogeneousIcePolicy.inl>
