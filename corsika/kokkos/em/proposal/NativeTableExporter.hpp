/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace corsika::proposal {
  class NativeInteractionCalculatorView;
  class NativeContinuousCalculatorView;
} // namespace corsika::proposal

namespace corsika::kokkos {

  /**
   * Copy immutable interpolation state from calculators already owned by CORSIKA.
   *
   * This host-only bridge requires the version-locked read-only PROPOSAL export API.
   * It neither parses private cache files nor samples a replacement physics table.
   * The existing single-composition and calculator-selection rules are retained.
   */
  [[nodiscard]] ProposalNativeTableSet export_proposal_native_tables(
      std::vector<proposal::NativeInteractionCalculatorView> const& interactions,
      std::vector<proposal::NativeContinuousCalculatorView> const& continuous);

  /**
   * Resolve the production PID/process/parametrization contract.
   * Unsupported tuples throw rather than acquiring a different physics model.
   */
  [[nodiscard]] NativeKinematicModel classify_proposal_native_kinematic_model(
      std::int32_t processId, std::string const& parametrization, std::int32_t pdgId);

} // namespace corsika::kokkos
