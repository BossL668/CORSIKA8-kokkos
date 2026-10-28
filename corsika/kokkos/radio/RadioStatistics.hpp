/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace corsika::kokkos::radio {
  /** Production execution diagnostics; recording these does not modify the shower. */
  struct RadioStatistics {
    std::uint64_t leptonTracks_{};
    std::uint64_t trackObserverPairs_{};
    std::uint64_t fusedTrackObserverPairs_{};
    std::uint64_t coreasContributions_{};
    std::uint64_t zhsContributions_{};
    std::uint64_t zhsSubtracks_{};
    std::uint64_t fixedPointOverflows_{};
    bool trackDiagnosticsEnabled_{};
    double weightedSegmentCount_{};
    double trackLengthM_{};
    double weightedTrackLengthM_{};
    double electronWeightedTrackLengthM_{};
    double positronWeightedTrackLengthM_{};
    double signedChargeWeightedTrackLengthM_{};
    double energyWeightedTrackLengthGeVM_{};
    double maximumSegmentLengthM_{};
    double weightedDirectionChangeRad_{};
    double weightedDirectionChangeSquaredRad2_{};
    double weightedBetaDeficitTrackLengthM_{};
    double weightedTimeResidualS_{};
    double maximumDirectionChangeRad_{};
    std::array<double, 3> signedChargeWeightedDirectionChange_{};
    std::array<double, 15> weightedTrackLengthByKineticEnergyM_{};
    std::size_t deviceBytes_{};
    std::uint64_t hostToDeviceBytes_{};
    std::uint64_t hostTracksEnqueued_{};
    std::uint64_t hostTrackBatches_{};
    std::uint64_t deviceToHostBytes_{};
    double deviceTimeMs_{};
    double kernelTimeMs_{};
    double transferTimeMs_{};
    std::uint64_t inputSlotWaits_{};
    double inputSlotHostWaitTimeMs_{};
    bool trackPrecomputeEnabled_{};
    std::uint32_t trackTileSize_{};
    std::uint32_t observerTileSize_{};
    std::uint64_t trackPrecomputeBatches_{};
    std::uint64_t trackPrecomputedRecords_{};
    std::uint64_t directProjectionBatches_{};
    std::uint64_t directProjectionRecords_{};
    std::uint64_t projectionTiles_{};
    std::size_t trackWorkspaceBytes_{};
    std::size_t maximumTrackBatch_{};
    double trackPrecomputeDeviceTimeMs_{};
    double projectionDeviceTimeMs_{};
  };
} // namespace corsika::kokkos::radio
