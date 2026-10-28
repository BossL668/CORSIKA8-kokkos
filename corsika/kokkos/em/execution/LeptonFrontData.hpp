/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {
  /** Production scan columns and one host checkpoint. Field widths and order
   * are unchanged. Legacy counters do not enable device rejection sampling. */
  enum ResidentLeptonFallbackStage : std::uint32_t {
    LeptonNoFallback = 0,
    LeptonSelectionFallback = 1,
    LeptonTransportFallback = 2,
    LeptonVertexFallback = 3,
    LeptonFinalStateFallback = 4,
  };
  struct ResidentLeptonSourceCounts {
    std::uint32_t child_{};
    std::uint32_t next_{};
    std::uint32_t photon_{};
    std::uint32_t fallbackStage_{LeptonNoFallback};
    std::uint32_t observation_{};
    std::uint32_t decay_{};
    std::uint32_t step_{};
    std::uint32_t record_{};
  };
  enum ResidentLeptonOffset : std::size_t {
    LeptonChildOffset = 0,
    LeptonNextOffset = 1,
    LeptonPhotonOffset = 2,
    LeptonSelectionFallbackOffset = 3,
    LeptonTransportFallbackOffset = 4,
    LeptonVertexFallbackOffset = 5,
    LeptonFinalStateFallbackOffset = 6,
    LeptonObservationOffset = 7,
    LeptonDecayOffset = 8,
    LeptonStepOffset = 9,
    LeptonRecordOffset = 10,
    LeptonOffsetCount = 11,
  };
  struct ResidentLeptonScanValue {
    std::uint64_t values_[LeptonOffsetCount]{};
  };
  struct ResidentLeptonFrontControl {
    ResidentLeptonScanValue totals_{};
    std::uint64_t interactionCount_{};
    std::uint64_t vertexInteractionCount_{};
    std::uint32_t materializationError_{};
  };
  enum ResidentLeptonTransportStatistic : std::size_t {
    LeptonTransportInteractionVertices = 0,
    LeptonTransportMoliereTrials,
    LeptonTransportMoliereDeflections,
    LeptonTransportMoliereZeroDeflections,
    LeptonTransportMoliereNewtonIterations,
    LeptonTransportMoliereMaximumNewtonIterations,
    LeptonTransportLimitBase,
    LeptonTransportStatisticCount = LeptonTransportLimitBase + 8,
  };
  enum ResidentLeptonInteractionStatistic : std::size_t {
    LeptonInteractionLpmSuppressions = 0,
    LeptonInteractionGpuFinalStates,
    LeptonInteractionPhysicalSecondaries,
    LeptonInteractionBremsFinalStates,
    LeptonInteractionAnnihilationFinalStates,
    LeptonInteractionIonizationFinalStates,
    LeptonInteractionElectronPairFinalStates,
    LeptonInteractionBremsLpmTrials,
    LeptonInteractionBremsLpmSuppressions,
    LeptonInteractionElectronPairLpmTrials,
    LeptonInteractionElectronPairLpmSuppressions,
    LeptonInteractionElectronPairRejectionTrials,
    LeptonInteractionElectronPairZeroWeightSamples,
    LeptonInteractionElectronPairRejectionFallbacks,
    LeptonInteractionElectronPairEnvelopeViolations,
    LeptonInteractionThinningHillasVertices,
    LeptonInteractionThinningStatisticalVertices,
    LeptonInteractionThinningParticlesDiscarded,
    LeptonInteractionStatisticCount,
  };
  constexpr std::size_t LeptonCallTransportStatisticBase = 0;
  constexpr std::size_t LeptonCallInteractionStatisticBase =
      LeptonCallTransportStatisticBase + LeptonTransportStatisticCount;
  constexpr std::size_t LeptonCallStatisticCount =
      LeptonCallInteractionStatisticBase + LeptonInteractionStatisticCount + 1;
  // Append receiver-only diagnostics; keep every production column at its old index.
  constexpr std::size_t LeptonCallMaterialBoundaries = LeptonCallStatisticCount - 1;
  static_assert(std::is_standard_layout_v<ResidentLeptonScanValue>);
  static_assert(std::is_trivially_copyable_v<ResidentLeptonScanValue>);
  static_assert(sizeof(ResidentLeptonScanValue) ==
                LeptonOffsetCount * sizeof(std::uint64_t));
  static_assert(std::is_standard_layout_v<ResidentLeptonFrontControl>);
  static_assert(std::is_trivially_copyable_v<ResidentLeptonFrontControl>);
  static_assert(offsetof(ResidentLeptonFrontControl, totals_) == 0);
  static_assert(offsetof(ResidentLeptonFrontControl, interactionCount_) ==
                sizeof(ResidentLeptonScanValue));
  static_assert(offsetof(ResidentLeptonFrontControl, vertexInteractionCount_) ==
                sizeof(ResidentLeptonScanValue) + sizeof(std::uint64_t));
  static_assert(offsetof(ResidentLeptonFrontControl, materializationError_) ==
                sizeof(ResidentLeptonScanValue) + 2 * sizeof(std::uint64_t));

  /**
   * Exclusive scan used to compact interaction candidates without changing
   * their source order.  The expensive vertex/final-state stage consumes the
   * resulting source-index list; all other transported leptons bypass it.
   */

} // namespace corsika::kokkos
