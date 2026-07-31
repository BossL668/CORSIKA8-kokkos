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

#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em::tables {

  /**
   * Owns one immutable device copy of a validated v4 rate table.
   *
   * deviceView() is a small pointer-only object intended to be copied into
   * transport kernels. queryForValidation() performs allocations and copies
   * deliberately and must not be used in the production wavefront hot path.
   */
  class CudaRateTable {
  public:
    CudaRateTable();
    ~CudaRateTable();

    CudaRateTable(CudaRateTable&&) noexcept;
    CudaRateTable& operator=(CudaRateTable&&) noexcept;

    CudaRateTable(CudaRateTable const&) = delete;
    CudaRateTable& operator=(CudaRateTable const&) = delete;

    void initialize(
        RateTableSet const& source, int device,
        std::size_t maximum_device_bytes =
            std::numeric_limits<std::size_t>::max());
    void reset() noexcept;
    bool initialized() const noexcept;
    std::size_t deviceBytes() const noexcept;
    Sha256Digest const& sourceContentHash() const;
    FlatRateTableView deviceView() const;

    std::vector<TableQueryResult> queryForValidation(
        std::vector<TableQuery> const& queries) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::gpu::em::tables
