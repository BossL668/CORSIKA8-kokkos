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
  /** Production source counts, stable scan columns and one host checkpoint.
   * Integer widths, column order and control-record layout are unchanged. */
  enum ResidentPhotonFallbackStage : std::uint32_t {
    PhotonNoFallback = 0,
    PhotonSelectionFallback = 1,
    PhotonTransportFallback = 2,
    PhotonFinalStateFallback = 3,
  };
  struct ResidentPhotonSourceCounts {
    std::uint32_t child_{};
    std::uint32_t next_{};
    std::uint32_t charged_{};
    std::uint32_t fallbackStage_{PhotonNoFallback};
    std::uint32_t observation_{};
    std::uint32_t step_{};
    std::uint32_t record_{};
  };
  enum ResidentPhotonOffset : std::size_t {
    PhotonChildOffset = 0,
    PhotonNextOffset = 1,
    PhotonChargedOffset = 2,
    PhotonSelectionFallbackOffset = 3,
    PhotonTransportFallbackOffset = 4,
    PhotonFinalStateFallbackOffset = 5,
    PhotonObservationOffset = 6,
    PhotonStepOffset = 7,
    PhotonRecordOffset = 8,
    PhotonOffsetCount = 9,
  };
  struct ResidentPhotonScanValue {
    std::uint64_t values_[PhotonOffsetCount]{};
  };
  struct ResidentPhotonFrontControl {
    ResidentPhotonScanValue totals_{};
    std::uint64_t interactionCount_{};
    std::uint32_t materializationError_{};
  };
  enum ResidentPhotonTransportStatistic : std::size_t {
    PhotonTransportInteractionVertices = 0,
    PhotonTransportLayerBoundaries,
    PhotonTransportParticleCuts,
    PhotonSelectionNativeNewtonIterations,
    PhotonSelectionNativeBisectionIterations,
    PhotonSelectionNativeInverseFailures,
    PhotonTransportStatisticCount,
  };
  enum ResidentPhotonInteractionStatistic : std::size_t {
    PhotonInteractionLpmSuppressions = 0,
    PhotonInteractionGpuFinalStates,
    PhotonInteractionPhysicalSecondaries,
    PhotonInteractionPairFinalStates,
    PhotonInteractionMuonPairFinalStates,
    PhotonInteractionComptonFinalStates,
    PhotonInteractionPhotoelectricFinalStates,
    PhotonInteractionLpmTrials,
    PhotonInteractionThinningHillasVertices,
    PhotonInteractionThinningStatisticalVertices,
    PhotonInteractionThinningParticlesDiscarded,
    PhotonInteractionStatisticCount,
  };
  inline constexpr std::size_t PhotonCallTransportStatisticOffset = 0;
  inline constexpr std::size_t PhotonCallInteractionStatisticOffset =
      PhotonTransportStatisticCount;
  inline constexpr std::size_t PhotonCallStatisticCount =
      PhotonTransportStatisticCount + PhotonInteractionStatisticCount;
  static_assert(std::is_standard_layout_v<ResidentPhotonScanValue>);
  static_assert(std::is_trivially_copyable_v<ResidentPhotonScanValue>);
  static_assert(sizeof(ResidentPhotonScanValue) ==
                PhotonOffsetCount * sizeof(std::uint64_t));
  static_assert(std::is_standard_layout_v<ResidentPhotonFrontControl>);
  static_assert(std::is_trivially_copyable_v<ResidentPhotonFrontControl>);
  static_assert(offsetof(ResidentPhotonFrontControl, totals_) == 0);
  static_assert(offsetof(ResidentPhotonFrontControl, interactionCount_) ==
                sizeof(ResidentPhotonScanValue));
  static_assert(offsetof(ResidentPhotonFrontControl, materializationError_) ==
                sizeof(ResidentPhotonScanValue) + sizeof(std::uint64_t));

} // namespace corsika::kokkos
