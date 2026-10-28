/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace corsika::kokkos::test {

  // Real PROPOSAL tables with deliberately small interpolation grids:
  // dN/dX 32 x 32, dE/dX/range/total-rate 64 nodes, Emax=1e8 MeV,
  // Air, official five projectile factories, absolute cut 0.4 MeV.
  // These grids test export and algorithm preservation, not shower accuracy.
  ProposalNativeTableSet make_native_tables(std::string const& cacheDirectory);
  ProposalNativeTableSet make_native_tables();
  std::string native_table_cache_directory();

  struct NativeTableExportAudit {
    ProposalNativeTableSet table_;
    sha256_digest_type referenceHash_{};
    bool allRecordsEqual_{};
    bool ordinaryStateUnchanged_{};
    bool emptyViewsRejected_{};
    bool defaultViewsEmpty_{};
    std::size_t processCount_{};
    std::vector<std::int32_t> projectileIds_;
  };

  NativeTableExportAudit audit_native_tables(std::string const& cacheDirectory);
} // namespace corsika::kokkos::test
