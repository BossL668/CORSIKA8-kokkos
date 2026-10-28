/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/data/ParticleConversion.hpp>

// No preceding includes: verify the public header declares all of its dependencies.
static_assert(sizeof(corsika::kokkos::ParticleState::energyGeV_) == sizeof(double));
