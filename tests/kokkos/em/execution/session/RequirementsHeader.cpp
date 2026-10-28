/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/proposal/NativeRequirements.hpp>

void requirements_header_smoke(corsika::kokkos::ProposalNativeTableSet const& table,
                               corsika::kokkos::PhysicsRequirements const& requirements) {
  corsika::kokkos::validate_proposal_native_requirements(table, requirements);
}
