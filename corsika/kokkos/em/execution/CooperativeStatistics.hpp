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
  struct CooperativeStatistics {
    bool enabled_{};
    bool independentDrivers_{};
    bool independentSubshowers_{};
    bool adaptivePolicy_{};
    // Bounded host-only integer profile replicas, never full per-thread arenas.
    std::size_t hostProfileShards_{}, hostProfileShardBytes_{};
    // [endpoint CUDA/OpenMP][kind photon/lepton]; host-call costs, not kernel time.
    std::array<std::array<double, 2>, 2> adaptiveRecordsPerMs_{};
    std::array<std::array<std::uint64_t, 2>, 2> adaptiveObservations_{};
    std::array<std::array<std::size_t, 2>, 2> adaptiveWaveLimit_{};
    // Completed physical steps, unlike repeatedly counted job inputs.
    std::array<std::array<std::uint64_t, 2>, 2> adaptiveTransportRecords_{};
    std::array<std::array<std::uint64_t, 2>, 2> adaptiveResidentWavefronts_{};
    std::array<std::array<std::uint64_t, 2>, 2> adaptiveFullObservations_{};
    // [endpoint][kind][completed, minimum, workspace, history, wave limit, other].
    std::array<std::array<std::array<std::uint64_t, 6>, 2>, 2>
        adaptiveCompletionReasons_{};
    std::array<std::array<std::size_t, 2>, 2> adaptiveInputLimit_{};
    // floor(log2(inputs)), final bin >=2^20; bounded, no per-job trace growth.
    std::array<std::array<std::uint64_t, 21>, 2> adaptiveJobInputHistogram_{};
    std::array<std::uint64_t, 2> adaptiveSpeciesCoalesces_{};
    double adaptiveMigrationMs_{};
    std::uint64_t subshowerCudaAutonomousContinuations_{};
    /** Submissions during coordinator scalar-yield; not kernel overlap time. */
    std::uint64_t subshowerCudaForegroundPackets_{};
    std::uint64_t subshowerCudaForegroundContinuations_{};
    double cudaCompletionBufferDelayMs_{};
    std::uint64_t cudaCompletionMailboxCapacity_{};
    std::uint64_t cudaCompletionRetentionBudgetBytes_{};
    std::uint64_t cudaCompletionPeakBytes_{};
    std::uint64_t maximumCudaPacketCalls_{};
    // completed, no-progress, memory, call-limit, handoff, time, disabled.
    std::array<std::uint64_t, 7> cudaContinuationStops_{};
    std::uint64_t subshowerCudaSubmissions_{}, subshowerCudaCommits_{};
    std::uint64_t subshowerOpenmpEpochs_{}, maximumHostEpochsPerCudaJob_{};
    std::uint64_t subshowerTailMigrations_{}, subshowerHostQueuePeakBytes_{};
    std::uint64_t subshowerRequeuedSpillParticles_{},
        subshowerCrossEndpointSpillParticles_{};
    std::uint64_t subshowerProactiveRefills_{}, subshowerInflightWaitingMigrations_{};
    std::uint64_t subshowerInflightWaitingParticles_{}, subshowerCudaEpochReductions_{};
    std::size_t subshowerGpuLeptonWaveLimit_{1024};
    double subshowerInitialHostShare_{}, subshowerMaximumCudaEpochMs_{};
    double coordinatorIdleWaitMs_{}, cudaResultServiceDelayMs_{};
    std::uint64_t independentJointCalls_{};
    double cudaDriverWallMs_{}, jointWallMs_{}, endpointWindowOverlapMs_{};
    // CPU-primary helper only: event-scoped blocking waits, never global CUDA
    // scheduling flags. Durations overlap OpenMP work and are not kernel time.
    bool auxiliaryBlockingWaitEnabled_{};
    std::uint64_t auxiliaryBlockingWaitCalls_{};
    double auxiliaryBlockingWaitMs_{};
    double cudaFinishedBeforeHostMs_{}, hostFinishedBeforeCudaMs_{};
    std::uint64_t cudaInputParticles_{}, openmpInputParticles_{};
    std::uint64_t openmpSlices_{}, slicesWhileCudaPending_{}, oversizedSlices_{};
    std::uint64_t migrationBytes_{}, photonCalls_{}, leptonCalls_{};
    double openmpWallMs_{}, openmpWhileCudaPendingMs_{}, maximumSliceMs_{};
    // A submission window is NOT a measured kernel-overlap interval.
    std::size_t openmpBatchTarget_{256}, openmpWorkspaceBytes_{};
  };
} // namespace corsika::kokkos
