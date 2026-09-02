/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

#include <corsika/gpu/em/tables/ProposalNativeTable.hpp>

namespace corsika::gpu::em::tables {

  enum class ProposalNativeQueryKind : std::uint32_t {
    Rate = 0,
    TotalRate = 1,
    CumulativeRate = 2,
    LossFraction = 3,
    ContinuousDedx = 4,
    ContinuousRange = 5,
    ContinuousEnergy = 6,
    ContinuousEnergyAfterLoss = 7,
  };

  struct ProposalNativeQuery {
    ProposalNativeQueryKind kind{ProposalNativeQueryKind::Rate};
    std::int32_t pdg_id{};
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    double energy_MeV{};
    double argument{};
  };

  /** Validation-only explicit-threshold selector query. */
  struct ProposalNativeSelectionQuery {
    std::int32_t pdg_id{};
    double energy_MeV{};
    double threshold{};
    std::int32_t boundary_process_id{};
    std::uint64_t boundary_component_hash{};
  };

  struct ProposalNativeSelectionResult {
    NativeQueryStatus status{NativeQueryStatus::InvalidView};
    std::int32_t process_id{};
    std::uint64_t component_hash{};
    double total_rate{};
    double boundary_cumulative_rate{};
    double residual_quantile{};
    std::uint32_t selected{};
  };

} // namespace corsika::gpu::em::tables
