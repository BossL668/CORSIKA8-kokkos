/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <type_traits>
namespace corsika::kokkos {
  /** Production profile input, not a radio track. Deposits are unweighted GeV;
   * depths are g/cm^2 and weight_ is carried separately. The reserved_ bits retain
   * the original multiple-scattering diagnostics and nativeMuonPairCut_ the
   * tagged signed muon-cut convention. Construction performs no accumulation.
   */
  struct ProjectedStepRecord {
    std::uint64_t historyId_{};
    std::int32_t pid_{};
    std::int32_t processId_{};
    std::int32_t transportLimit_{};
    std::int32_t reserved_{};
    double startGrammageGPerCm2_{};
    double endGrammageGPerCm2_{};
    double endEnergyGeV_{};
    double depositedEnergyGeV_{};
    double weight_{};
    double cutDepositedEnergyGeV_{};
    std::uint32_t observationSurfaceReachedBeforeCut_{};
    std::uint32_t nativeMuonPairCut_{};
  };
  static_assert(std::is_standard_layout_v<ProjectedStepRecord>);
  static_assert(std::is_trivially_copyable_v<ProjectedStepRecord>);
} // namespace corsika::kokkos
