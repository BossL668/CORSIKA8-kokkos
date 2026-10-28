/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeQueries.hpp>

#include <cstdint>

namespace corsika::kokkos {

  /** Original lookup status values, shared with subsequent interaction queries. */
  enum class TableLookupStatus : std::uint32_t {
    Success = 0,
    ParticleNotFound = 1,
    ColumnNotFound = 2,
    RateEnergyOutOfRange = 3,
    InverseCdfUnavailable = 4,
    LossEnergyOutOfRange = 5,
    LossQuantileOutOfRange = 6,
    NonFiniteInput = 7,
    InvalidQueryKind = 8,
    InvalidTableView = 9,
    ContinuousParticleNotFound = 10,
    ContinuousEnergyOutOfRange = 11,
    ContinuousRangeOutOfRange = 12,
    TransportCutReached = 13,
  };

  /** Non-throwing result; reserved_ packs the original inverse-solver counts. */
  struct TableQueryResult {
    TableLookupStatus status_{TableLookupStatus::InvalidTableView};
    std::uint32_t reserved_{};
    double value_{};
  };

  /** Saturate each count to 16 bits; Newton occupies the low half. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline std::uint32_t pack_native_inverse_iterations(
      std::uint32_t newton, std::uint32_t bisection);
  /** Read the saturated Newton count from reserved_. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline std::uint32_t native_newton_iterations(
      TableQueryResult const& result);
  /** Read the saturated bisection count from reserved_. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline std::uint32_t native_bisection_iterations(
      TableQueryResult const& result);

  namespace table_query_detail {
    CORSIKA_HOST_DEVICE inline bool finite_value(double value);
    CORSIKA_HOST_DEVICE inline TableQueryResult from_proposal_native(
        NativeQueryResult const& result, bool continuous = false);
  } // namespace table_query_detail

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/tables/TableQueryResult.inl>
