/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTopology.hpp>

// Instantiate every member, not merely a pointer to an incomplete template.
template class corsika::kokkos::EnvironmentTopology<corsika::IEmpty>;
