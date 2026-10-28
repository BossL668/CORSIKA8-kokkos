/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  // These identities are disjoint from process-specific final-state draws.
  inline constexpr std::uint64_t EmThinningFirstDrawId = 0x100;
  inline constexpr std::uint64_t EmThinningSecondDrawId = 0x101;

  /** Value configuration of production one-to-two electromagnetic thinning.
   * thresholdGeV_ applies to the parent's total energy, not to a secondary cut.
   * eraseZeroWeight_=0 retains both child slots, including zero-weight children.
   */
  struct EmThinningConfig {
    double thresholdGeV_{};
    double maximumWeight_{};
    std::uint32_t enabled_{};
    std::uint32_t eraseZeroWeight_{1};
  };
  enum class EmThinningStatus : std::uint32_t {
    NotApplied = 0,
    Hillas = 1,
    Statistical = 2,
    InvalidInput = 3,
  };
  /** Retained-child mask and weights; bit zero/one corresponds to child zero/one.
   * InvalidInput must be handled by the caller. A selected zero-weight child is
   * erased only in the production erase-zero-weight mode.
   */
  struct EmThinningResult {
    EmThinningStatus status_{EmThinningStatus::InvalidInput};
    std::uint32_t keepMask_{};
    double firstWeight_{};
    double secondWeight_{};
  };
  static_assert(std::is_standard_layout_v<EmThinningConfig>);
  static_assert(std::is_trivially_copyable_v<EmThinningConfig>);
  static_assert(std::is_standard_layout_v<EmThinningResult>);
  static_assert(std::is_trivially_copyable_v<EmThinningResult>);

  /** Apply production thinning to already generated, eligible two-child EM data.
   * The caller checks particle eligibility and multiplicity before calling.
   * All energies are total GeV; weights are dimensionless. Both supplied uniforms
   * must be strictly inside (0,1), including when thinning is disabled.
   * first_uniform selects the Hillas child or the first statistical decision;
   * second_uniform supplies the second statistical decision. This function never
   * draws random numbers or mutates particles; it returns weights and keep bits.
   * Preserve production thresholds, zero-weight handling and invalid-input rules.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline EmThinningResult apply_em_thinning(
      EmThinningConfig const& config, double parent_energy_GeV, double parent_weight,
      double first_energy_GeV, double second_energy_GeV, double first_uniform,
      double second_uniform);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/EmThinning.inl>
