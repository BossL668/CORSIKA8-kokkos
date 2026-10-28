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
  /** Diagnostic record of a generated photon final state.
   * processId_ identifies the process-specific meaning of energySplitFraction_:
   * electron or positive-muon fraction for pair production, transferred fraction
   * for Compton scattering, electron kinetic fraction for photoelectric absorption.
   * secondaryOffset_/secondaryCount_ address retained children in the caller's
   * output storage. Sampling draws and thinning metadata retain production order.
   * weightedMassConventionCorrectionGeV_ is a weighted energy correction, not a
   * new physical energy-loss process. This record performs no sampling.
   */
  // Units match ParticleState (GeV, metres, seconds); weights are dimensionless.
  // Preserve field order, widths and defaults. Native layout is not a wire format.
  struct PhotonFinalStateRecord {
    std::uint64_t inputIndex_{};
    std::uint64_t parentHistoryId_{};
    std::uint64_t secondaryOffset_{};
    std::uint32_t secondaryCount_{};
    std::int32_t processId_{};
    double energySplitFraction_{};
    double splitUniform_{};
    double azimuthUniform_{};
    double electronPolarUniform_{};
    double positronPolarUniform_{};
    double lpmSurvivalProbability_{};
    double lpmUniform_{};
    std::uint64_t splitDrawId_{};
    std::uint64_t azimuthDrawId_{};
    std::uint64_t electronPolarDrawId_{};
    std::uint64_t positronPolarDrawId_{};
    std::uint64_t lpmDrawId_{};
    std::uint32_t thinningStatus_{};
    std::uint32_t thinningKeepMask_{0x3U};
    double thinningFirstUniform_{};
    double thinningSecondUniform_{};
    std::uint64_t thinningFirstDrawId_{};
    std::uint64_t thinningSecondDrawId_{};
    double weightedMassConventionCorrectionGeV_{};
  };
  static_assert(std::is_standard_layout_v<PhotonFinalStateRecord>);
  static_assert(std::is_trivially_copyable_v<PhotonFinalStateRecord>);
} // namespace corsika::kokkos
