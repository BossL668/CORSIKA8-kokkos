/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ProposalNativeData.hpp>
#include <corsika/kokkos/utility/Sha256.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace corsika::kokkos {

  /** Owning host-only canonical export; values retain PROPOSAL's MeV-based units. */
  struct ProposalNativeTableSet {
    std::uint32_t formatVersion_{ProposalNativeTableFormatVersion};
    std::string proposalVersion_;
    std::string cubicInterpolationVersion_;
    NativePhysicalConstants constants_{};
    double stochasticCutMeV_{};
    sha256_digest_type contentHash_{};
    // Diagnostic state is deliberately excluded from contentHash_: a cold
    // build and a warm PROPOSAL cache load must export identical physics.
    std::uint64_t proposalCacheTableCount_{};
    std::uint64_t proposalCacheHitCount_{};
    bool proposalCacheAllHit_{};
    std::vector<NativeInteractionIdentity> interactionIdentities_;
    std::vector<NativeDndxColumn> dndxColumns_;
    std::vector<NativeTotalRateColumn> totalRateColumns_;
    // Index indirection that walks dndxColumns_ in Interaction::Rates order.
    // Entries remain grouped by PID and have one-to-one coverage.
    std::vector<std::uint32_t> selectionColumnIndices_;
    std::vector<NativeDedxColumn> dedxColumns_;
    // Index indirection that walks dedxColumns_ in scalar PROPOSAL's exact
    // cross-list/component summation order. Entries remain grouped by PID.
    std::vector<std::uint32_t> dedxAccumulationIndices_;
    std::vector<NativeUtilityColumn> utilityColumns_;
    // All bicubic arrays use the same column-major offset and element count.
    std::vector<double> bicubicValues_;
    std::vector<double> bicubicDerivativeEnergy_;
    std::vector<double> bicubicDerivativeLoss_;
    std::vector<double> bicubicMixedDerivative_;
    // Sixteen row-major coefficients per cell.  Cells are ordered with the
    // energy interval varying fastest: energy + (rows - 1) * loss.
    std::vector<double> bicubicPolynomialCoefficients_;
    // Every 1-D interval is represented exactly by node values and first
    // derivatives copied from the cardinal cubic B-spline.
    std::vector<double> cubicValues_;
    std::vector<double> cubicNodeDerivatives_;
  };

  /**
   * Borrow the table arrays for numerical queries without copying or validation.
   * The table must remain alive and its storage must not be changed while used.
   */
  [[nodiscard]] ProposalNativeDeviceView make_proposal_native_host_view(
      ProposalNativeTableSet const& table);

  /** Return the production device-array byte estimate, excluding host diagnostics. */
  [[nodiscard]] std::size_t proposal_native_table_bytes(
      ProposalNativeTableSet const& table);

  /** Validate the original format, dependency identities, ordering and coefficient
   * layout. */
  void validate_proposal_native_table(ProposalNativeTableSet const& table);

  /**
   * Hash the canonical physics content in the production encoding order.
   * Cache-hit diagnostics and contentHash_ itself are excluded from the digest.
   */
  [[nodiscard]] sha256_digest_type calculate_proposal_native_hash(
      ProposalNativeTableSet const& table);

} // namespace corsika::kokkos
