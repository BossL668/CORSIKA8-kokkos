/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace corsika::kokkos::application {

  /**
   * Application-owned acceleration options.
   *
   * This type deliberately contains no CUDA runtime objects.  Keeping the CLI
   * contract independent of the optional backend lets a CPU-only build expose
   * the same help text and reject an explicitly requested CUDA backend with the
   * same fail-closed diagnostic as the production build.
   */
  struct BackendOptions {
    std::string emBackend_{"proposal"};
    double egs4Stepfc_{1.};
    int gpuDevice_{0};
    std::size_t gpuMinBatch_{4096};
    std::size_t gpuResidentBatchLimit_{};
    double gpuMemoryFraction_{0.70};
    std::string gpuPhysicsSource_{"proposal-native"};
    std::filesystem::path gpuAuxCacheDir_;
    bool gpuDeterministic_{true};
    bool gpuDetailedStageTiming_{false};
    bool gpuFullStepRecords_{false};
    bool gpuResidentCrossSpecies_{true};

    int kokkosNumThreads_{0};
    std::string kokkosExecution_;
    std::string kokkosCooperativePolicy_{"legacy"};
    int kokkosDevice_{0};
    std::filesystem::path kokkosTuningCache_;
    bool kokkosRequireTuning_{false};

    std::string radioBackend_{"cpu"};
    // Set by applications supplying a host radio adapter for transported tracks.
    // This is a library capability, not a command-line option.
    bool hostRadioAdapter_{false};
    // Internal single-event handoff from CPU prefix processes. Public radio
    // selection remains cpu/kokkos; these files contain transported segments.
    std::vector<std::string> radioTrackInputs_;
    double gpuRadioFieldLimit_{1.};
    bool gpuRadioTrackDiagnostics_{false};
    double radioSamplingRateGHz_{1.};
    double radioWindowDurationNs_{400.};
    double radioPretriggerNs_{10.};

    std::string cudaReplayTrace_;
    std::filesystem::path cudaReplayTapeOut_;

    int hadronicPlanWorkers_{4};
    double hadronicPlanTargetMs_{5.};
    std::size_t hadronicPlanMaxBatch_{256};
    std::string hadronicBackend_{"scalar"};
    int hadronicWorkers_{4};
    std::size_t hadronicMinBatch_{64};
    double hadronicTargetBatchMs_{5.};
    std::size_t hadronicMaxBatch_{256};
    double hadronicInitialCostMs_{0.1};
    bool cpuDetailedStepTiming_{false};
    std::filesystem::path hadronicWorkerExecutable_;
  };

} // namespace corsika::kokkos::application
