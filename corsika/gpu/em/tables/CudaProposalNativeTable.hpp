/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <corsika/gpu/em/tables/ProposalNativeQueries.hpp>

namespace corsika::gpu::em::tables {

  /**
   * Owns one immutable device copy of PROPOSAL's native interpolation state.
   * All coefficient arrays are uploaded exactly once.  No PROPOSAL object,
   * virtual function, Boost archive or host pointer is visible to CUDA.
   */
  class CudaProposalNativeTable {
  public:
    CudaProposalNativeTable();
    ~CudaProposalNativeTable();

    CudaProposalNativeTable(CudaProposalNativeTable&&) noexcept;
    CudaProposalNativeTable& operator=(CudaProposalNativeTable&&) noexcept;

    CudaProposalNativeTable(CudaProposalNativeTable const&) = delete;
    CudaProposalNativeTable& operator=(CudaProposalNativeTable const&) = delete;

    void initialize(
        ProposalNativeTableSet const& source, int device,
        std::size_t maximum_device_bytes =
            std::numeric_limits<std::size_t>::max());
    void reset() noexcept;

    bool initialized() const noexcept;
    std::size_t deviceBytes() const noexcept;
    Sha256Digest const& sourceContentHash() const;
    ProposalNativeDeviceView deviceView() const;

    std::vector<NativeQueryResult> queryForValidation(
        std::vector<ProposalNativeQuery> const& queries) const;
    std::vector<ProposalNativeSelectionResult> selectForValidation(
        std::vector<ProposalNativeSelectionQuery> const& queries) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::gpu::em::tables
