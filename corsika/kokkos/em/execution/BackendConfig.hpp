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
#include <filesystem>
#include <corsika/kokkos/em/algorithms/EmThinning.hpp>
#include <corsika/kokkos/output/ProfileProjectionConfig.hpp>
#include <corsika/kokkos/radio/RadioConfig.hpp>

namespace corsika::kokkos {
  enum class PhysicsSource : std::uint32_t {
    ProposalNative = 1,
  };

  struct BackendConfig {

    int device_{0};
    std::size_t minBatchSize_{4096};
    /**
     * Optional backend-neutral logical checkpoint for a resident input batch.
     *
     * Zero keeps the backend's automatic capacity policy.  A non-zero value
     * is primarily useful for cross-backend decision-tape validation: using
     * the same logical checkpoint prevents otherwise equivalent native-CUDA
     * and Kokkos-CUDA runs from assigning different history-ID ranges merely
     * because their workspace implementations advertise different physical
     * capacities.  Backends must reject a value they cannot hold; they must
     * never silently clamp it.
     */
    std::size_t residentBatchLimit_{};
    double memoryFraction_{0.70};
    double emTransportCutMeV_{};
    // User-facing muon kinetic-energy/ParticleCut threshold.  Native
    // PROPOSAL utilities are shared by electrons and muons, so their scalar
    // transport endpoints must remain independently configurable.
    double muonTransportCutMeV_{};
    bool deterministic_{true};
    /**
     * Record CUDA-event timings at the boundaries of the fused lepton
     * pipeline. This is disabled in production because the extra event
     * records are intended for profiling, not normal shower transport.
     */
    bool detailedStageTiming_{};
    /**
     * Retain the process/component/v decisions made at physical interaction
     * vertices for cross-backend replay diagnostics.  This deliberately
     * defaults to false: enabling it adds device-to-host traffic at every
     * interaction-bearing wavefront and is not a production transport mode.
     */
    bool diagnosticInteractionRecords_{};
    std::uint64_t randomSeed_{};
    std::uint64_t showerId_{};
    PhysicsSource physicsSource_{PhysicsSource::ProposalNative};
    std::filesystem::path auxiliaryCacheDirectory_{};
    EmThinningConfig thinning_{};
    bool residentCrossSpecies_{};
    ProfileProjectionConfig profileProjection_{};
    radio::RadioConfig radio_{};
  };
} // namespace corsika::kokkos
