/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {
  /** Diagnostic record of a generated charged-lepton final state.
   * photonEnergyFraction_ retains its production name: it stores the brems photon
   * fraction, annihilation split, ionization transfer or direct-pair transfer,
   * according to processId_. Offsets/counts address the caller's retained children.
   * Keeping a direct-pair field does not enable a device sampler: the two missing
   * conditional-rho tables continue to require CPU completion in production.
   * weightedMassConventionCorrectionGeV_ retains the weighted energy correction.
   */
  // Units match ParticleState (GeV, metres, seconds); weights are dimensionless.
  // Preserve field order, widths and defaults. Native layout is not a wire format.
  struct LeptonFinalStateRecord {
    std::uint64_t inputIndex_{};
    std::uint64_t parentHistoryId_{};
    std::uint64_t secondaryOffset_{};
    std::uint32_t secondaryCount_{};
    std::int32_t processId_{};
    double photonEnergyFraction_{};
    double finalStateUniform_{};
    double azimuthUniform_{};
    double auxiliaryUniform_{};
    double lpmSurvivalProbability_{};
    double lpmUniform_{};
    std::uint64_t finalStateDrawId_{};
    std::uint64_t azimuthDrawId_{};
    std::uint64_t auxiliaryDrawId_{};
    std::uint64_t lpmDrawId_{};
    std::uint32_t thinningStatus_{};
    std::uint32_t thinningKeepMask_{0x3U};
    double thinningFirstUniform_{};
    double thinningSecondUniform_{};
    std::uint64_t thinningFirstDrawId_{};
    std::uint64_t thinningSecondDrawId_{};
    double weightedMassConventionCorrectionGeV_{};
  };
  static_assert(std::is_standard_layout_v<LeptonFinalStateRecord>);
  static_assert(std::is_trivially_copyable_v<LeptonFinalStateRecord>);
} // namespace corsika::kokkos
