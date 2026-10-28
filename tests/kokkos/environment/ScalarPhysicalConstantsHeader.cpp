/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/ScalarPhysicalConstants.hpp>

static_assert(corsika::kokkos::scalar_constants::Version == 1);
static_assert(corsika::kokkos::scalar_constants::SpeedOfLightMPerS == 299792458.);
static_assert(corsika::kokkos::scalar_constants::ElementaryChargeC == 1.6021766208e-19);
static_assert(corsika::kokkos::scalar_constants::VacuumPermittivityFPerM ==
              8.8541878128e-12);
static_assert(corsika::kokkos::scalar_constants::MagneticRigidityGeVPerTeslaMeter ==
              .2997925146834389);
