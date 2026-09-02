/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosProposalNativeTableTestDriver.hpp"

#include <Kokkos_Core.hpp>

#include <corsika/accelerator/em/kokkos/KokkosProposalNativeTable.hpp>

namespace corsika::accelerator::em::testing {

  KokkosProposalNativeQueryOutput runKokkosProposalNativeQueries(
      gpu::em::tables::ProposalNativeTableSet const& source,
      std::vector<gpu::em::tables::ProposalNativeQuery> const& queries) {
    kokkos_detail::KokkosProposalNativeTable<Kokkos::DefaultExecutionSpace>
        table;
    table.initialize(source);
    return {table.queryForValidation(queries), table.sourceContentHash(),
            table.deviceBytes()};
  }

} // namespace corsika::accelerator::em::testing
