/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/ProcessId.hpp>
#include <corsika/kokkos/em/models/PhotonPairKinematics.hpp>
#include <corsika/kokkos/em/tables/TableQueryResult.hpp>

#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  /** Original ordered process/component selection and its conditional quantile. */
  struct RateColumnSelectionResult {
    TableLookupStatus status_{TableLookupStatus::InvalidTableView};
    std::int32_t processId_{};
    std::uint64_t componentHash_{};
    double totalRate_{};
    double residualQuantile_{};
    std::uint32_t selected_{};
  };
  static_assert(std::is_trivially_copyable_v<RateColumnSelectionResult>);

  /** Query a physical column rate in cm^2/g at total energy in MeV.
   * The borrowed view arrays must remain accessible in the querying memory space.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_discrete_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV);

  /** Query the separate total-rate table used for sampling an interaction distance.
   * This is not the ordered column sum used by select_rate_column_by_uniform.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_discrete_total_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV);

  /** Select using a supplied uniform in [0,1), without generating random numbers.
   * Retains PROPOSAL column order, strict subtraction comparison and the residual
   * quantile from the same draw. A successful zero-rate result may select no column.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline RateColumnSelectionResult
  select_rate_column_by_uniform(ProposalNativeDeviceView const& view, std::int32_t pdg_id,
                                double energy_MeV, double uniform);

  /** Invert the original native CDF without changing its solver or endpoints.
   * Ordinary identities return the consumed energy fraction. The two query-only
   * photon-pair identities return the independent secondary split: normalized
   * electron coordinate for PhotonPairFinalStateProcessId, and muon energy
   * fraction for PhotonMuonPairFinalStateProcessId. Counts and failures are kept.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_discrete_loss_fraction(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV, double quantile);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/tables/DiscretePhysicsQueries.inl>
