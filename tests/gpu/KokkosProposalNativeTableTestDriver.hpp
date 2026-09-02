/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <corsika/gpu/em/tables/ProposalNativeQueries.hpp>
#include <corsika/gpu/em/tables/ProposalNativeTable.hpp>

namespace corsika::accelerator::em::testing {

  struct KokkosProposalNativeQueryOutput {
    std::vector<gpu::em::tables::NativeQueryResult> results;
    gpu::em::tables::Sha256Digest source_content_hash{};
    std::size_t device_bytes{};
  };

  /**
   * This bridge is compiled by the selected Kokkos device compiler.  Keeping
   * it separate lets the PROPOSAL calculator construction remain an ordinary
   * host C++ translation unit; nvcc therefore never has to parse the complete
   * CORSIKA geometry/unit header graph merely to launch the validation kernel.
   */
  KokkosProposalNativeQueryOutput runKokkosProposalNativeQueries(
      gpu::em::tables::ProposalNativeTableSet const& source,
      std::vector<gpu::em::tables::ProposalNativeQuery> const& queries);

} // namespace corsika::accelerator::em::testing
