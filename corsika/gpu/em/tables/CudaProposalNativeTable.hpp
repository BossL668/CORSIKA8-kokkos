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

  /** Validation-only explicit-threshold selector query.  Accepting an
   * absolute cumulative-rate threshold (rather than a random uniform) lets
   * the oracle probe the exact floating-point boundary and its two adjacent
   * representable values without perturbing the production RNG path. */
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
