/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <vector>

#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>
#include <corsika/modules/proposal/NativeCalculatorView.hpp>

namespace corsika::gpu::em::tables {

  /**
   * Copy the immutable interpolation state from calculators already owned by
   * CORSIKA. No private PROPOSAL cache files are parsed and no physics function
   * is sampled by this operation.
   */
  ProposalNativeTableSet exportProposalNativeTables(
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous);

} // namespace corsika::gpu::em::tables
