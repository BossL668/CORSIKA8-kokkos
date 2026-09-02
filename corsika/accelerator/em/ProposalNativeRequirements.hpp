/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/accelerator/em/AcceleratedPhysicsRequirements.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTable.hpp>

namespace corsika::accelerator::em {

  void validateProposalNativeRequirements(
      gpu::em::tables::ProposalNativeTableSet const&,
      AcceleratedPhysicsRequirements const&);

} // namespace corsika::accelerator::em
