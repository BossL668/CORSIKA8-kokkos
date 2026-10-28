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
#include <string>
#include <vector>

namespace corsika::kokkos {
  struct TransferTimingStatistics {
    bool deviceEventTimingEnabled_{};
    std::uint64_t operations_{};
    std::uint64_t hostToDeviceOperations_{};
    std::uint64_t deviceToHostOperations_{};
    std::uint64_t deviceToDeviceOperations_{};
    double hostApiTimeMs_{};
    double deviceCopyTimeMs_{};
    double hostWaitUpperBoundMs_{};
  };

  struct SynchronizationTimingStatistics {
    std::uint64_t physicalPipelineWaits_{};
    std::uint64_t profileInputWaits_{};
    double physicalPipelineWaitTimeMs_{};
    double profileInputWaitTimeMs_{};
  };

  struct LeptonPipelineTimingStatistics {
    bool enabled_{};
    std::uint64_t wavefronts_{};
    double selectionMs_{};
    double transportMs_{};
    double transportPhysicsMs_{};
    double moliereMs_{};
    double transportControlMs_{};
    double transportCompactionMs_{};
    double interactionExtractionMs_{};
    double vertexSelectionMs_{};
    double finalStateMs_{};
    double finalStateClassificationMs_{};
    double finalStateScanMs_{};
    double finalStateSummaryMs_{};
    double finalStateWriteMs_{};
    double endpointCompactionMs_{};
    double postEndpointMs_{};
  };
} // namespace corsika::kokkos
