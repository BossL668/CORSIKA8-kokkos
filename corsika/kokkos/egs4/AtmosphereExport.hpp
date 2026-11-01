/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <string>
#include <array>
#include <corsika/kokkos/egs4/AtmosphereData.hpp>
#include <corsika/kokkos/environment/EnvironmentDataView.hpp>
#include <corsika/kokkos/environment/TransportGeometryConfig.hpp>
#include <corsika/media/MediumProperties.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace corsika::kokkos::egs4 {
  /** Export the actual five nested C8 air nodes for C7 EGS4 stepping.
   * Accepts four radial exponentials and one homogeneous top layer, one
   * constant magnetic field and AIR-NTP composition. Rejects exclusions,
   * additional volumes and non-air media. No replacement atmosphere is built.
   */
  AtmosphereData export_atmosphere(EnvironmentDataView const& environment,
                                   TransportGeometryConfig const& geometry,
                                   double earthRadiusM);
} // namespace corsika::kokkos::egs4

#include <corsika/detail/kokkos/egs4/AtmosphereExport.inl>
