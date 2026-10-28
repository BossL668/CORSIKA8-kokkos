/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/LeptonFront.hpp>
#include <corsika/kokkos/em/execution/LeptonStatistics.hpp>
#include <corsika/kokkos/em/execution/LeptonCascadeResult.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
#include <corsika/kokkos/radio/RadioAccumulator.hpp>
#include <corsika/kokkos/scheduling/PendingParticleQueue.hpp>
#include <corsika/kokkos/scheduling/WavefrontExecutionGuard.hpp>
#include <corsika/kokkos/runtime/ResidentExecutionWait.hpp>
#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace corsika::kokkos {

  namespace lepton_cascade_detail {
    template <class TDestination, class THostView>
    void append_lepton_host_copy(TDestination& destination, THostView const& host,
                                 std::size_t const count);
  }
  /** Enqueue the production profile, diagnostic and radio accumulation.
   * The front must already be materialized and its control checked. Keep all
   * owners alive until execution finishes, and use the same execution instance.
   */
  template <class TExecutionSpace>
  void enqueue_lepton_accumulation(
      LeptonFrontWorkspace<TExecutionSpace> const& workspace,
      std::size_t const currentCount, ResidentLeptonFrontControl const& control,
      ProfileProjectionView const profileProjection,
      ProfileAccumulatorStorage<TExecutionSpace>* const profileAccumulator,
      radio::RadioAccumulator<TExecutionSpace>* const radioAccumulator,
      TExecutionSpace const& execution, std::size_t const openmpChunkSize);
  /** Advance production lepton fronts to completion or a lossless checkpoint.
   * All borrowed model/context owners must outlive the call. The caller must
   * continue CPU fallbacks, decay candidates, spill and remaining particles.
   * Preserve source order, history ranges, profile ownership and radio endpoints.
   * Geometry uses the common C8 environment tree; no spherical-layer shortcut.
   */
  template <class TExecutionSpace>
  LeptonCascadeResult run_lepton_cascade(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      double const electronMassGeV, bool const airMoliereFastPath,
      std::vector<ParticleState> const& particles,
      std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
      std::uint64_t const secondaryHistoryIdLimitExclusive,
      std::size_t const minimumResidentBatchSize,
      std::optional<FirstInteractionRecord>& firstInteraction,
      bool const projectSteps = false, ProfileProjectionView const profileProjection = {},
      ProfileAccumulatorStorage<TExecutionSpace>* const profileAccumulator = nullptr,
      radio::RadioAccumulator<TExecutionSpace>* const radioAccumulator = nullptr,
      LeptonFrontWorkspace<TExecutionSpace>* const workspace = nullptr,
      PendingParticleQueue<TExecutionSpace>* const pendingLeptons = nullptr,
      PendingParticleQueue<TExecutionSpace>* const pendingPhotons = nullptr,
      std::size_t const pendingInputCount = 0,
      std::size_t const maximumPendingParticles = std::numeric_limits<std::size_t>::max(),
      TExecutionSpace const& execution = {},
      bool const captureDiagnosticInteractions = false,
      std::size_t const openmpChunkSize = 0,
      ResidentMemoryBudgetGate const memoryGate = {},
      std::size_t const reservedSourceLimit = 0,
      ResidentExecutionWait<TExecutionSpace>* const cooperativeWait = nullptr);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/LeptonCascade.inl>
