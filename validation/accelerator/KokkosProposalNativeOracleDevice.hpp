/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <corsika/accelerator/em/common/tables/ProposalNativeQueries.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeTable.hpp>

namespace corsika::accelerator::em::testing {

  /**
   * Host-compiler-safe validation facade around the selected Kokkos device
   * table.  Its implementation is compiled by nvcc/hipcc when required, so
   * the live PROPOSAL oracle translation unit does not need to parse Kokkos
   * device headers.
   */
  class KokkosProposalNativeOracleDevice {
  public:
    KokkosProposalNativeOracleDevice();
    ~KokkosProposalNativeOracleDevice();

    KokkosProposalNativeOracleDevice(
        KokkosProposalNativeOracleDevice&&) noexcept;
    KokkosProposalNativeOracleDevice& operator=(
        KokkosProposalNativeOracleDevice&&) noexcept;
    KokkosProposalNativeOracleDevice(
        KokkosProposalNativeOracleDevice const&) = delete;
    KokkosProposalNativeOracleDevice& operator=(
        KokkosProposalNativeOracleDevice const&) = delete;

    void initialize(
        gpu::em::tables::ProposalNativeTableSet const&, int device = 0,
        std::size_t maximum_bytes = static_cast<std::size_t>(-1));
    std::size_t deviceBytes() const noexcept;
    std::vector<gpu::em::tables::NativeQueryResult> queryForValidation(
        std::vector<gpu::em::tables::ProposalNativeQuery> const&) const;
    std::vector<gpu::em::tables::ProposalNativeSelectionResult>
    selectForValidation(
        std::vector<gpu::em::tables::ProposalNativeSelectionQuery> const&)
        const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::accelerator::em::testing
