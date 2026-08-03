/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/gpu/em/tables/MediumConfig.hpp>
#include <corsika/gpu/em/tables/RateTable.hpp>

#include <PROPOSAL/medium/Medium.h>

#include <vector>

namespace corsika::gpu::em::tables {

  PROPOSAL::Medium makeProposalMedium(MediumConfig const& config);

  std::vector<MediumComponent> makeRateTableMediumComponents(
      MediumConfig const& config, PROPOSAL::Medium const& medium);

} // namespace corsika::gpu::em::tables
