/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ContinuousPhysicsQueries.hpp>
#include <corsika/kokkos/em/tables/DiscretePhysicsQueries.hpp>

namespace corsika::kokkos {

  /** Stable production operation identifiers. */
  enum class TableQueryKind : std::uint32_t {
    Rate = 0,
    TotalRate = 1,
    LossFraction = 2,
    ContinuousDedx = 3,
    ContinuousRange = 4,
    ContinuousEnergy = 5,
    EnergyAfterContinuousLoss = 6,
  };
  /** Value-only production request. energyMeV_ holds range (g/cm^2) for
   * ContinuousEnergy, otherwise energy (MeV). quantile_ holds grammage (g/cm^2)
   * for EnergyAfterContinuousLoss, otherwise the dimensionless loss quantile.
   * These historical slot meanings are preserved, not normalized by dispatch. */
  struct TableQuery {
    TableQueryKind kind_{TableQueryKind::Rate};
    std::int32_t pdgId_{};
    std::int32_t processId_{};
    std::uint64_t componentHash_{};
    double energyMeV_{};
    double quantile_{};
  };
  static_assert(std::is_trivially_copyable_v<TableQuery>);
  /** Route one request to the unchanged query, preserving status and iteration counts.
   * The borrowed view supplies only the native tables and cuts read by dispatch. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult execute_table_query(
      ContinuousPhysicsView const& view, TableQuery const& query);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/tables/TableQuery.inl>
