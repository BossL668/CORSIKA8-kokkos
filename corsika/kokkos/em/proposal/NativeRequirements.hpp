/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/PhysicsRequirements.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>

namespace corsika::kokkos {

  void validate_proposal_native_requirements(ProposalNativeTableSet const&,
                                             PhysicsRequirements const&);

} // namespace corsika::kokkos
