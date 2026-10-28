/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/PhotonFront.hpp>
#include <corsika/kokkos/em/execution/PhotonCascadeResult.hpp>
#include <corsika/kokkos/output/ProfileAccumulatorStorage.hpp>
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

  namespace photon_cascade_detail {
    struct ResidentPhotonTransportStatisticsValue {
      std::uint64_t values_[PhotonTransportStatisticCount]{};
    };
    struct ResidentPhotonInteractionStatisticsValue {
      std::uint64_t values_[PhotonInteractionStatisticCount]{};
    };
    template <class TExecutionSpace>
    struct ResidentPhotonTransportStatisticsFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = ResidentPhotonTransportStatisticsValue;

      Kokkos::View<InteractionSelectionOutcome const*, memory_space_type> selections_;
      Kokkos::View<PhotonTransportRecord const*, memory_space_type> packedSteps_;
      std::size_t stepCount_{};
      Kokkos::View<std::uint64_t*, memory_space_type> callStatistics_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void final(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                             value_type& value) const;
    };
    template <class TExecutionSpace>
    struct ResidentPhotonTransportStatisticsProfileFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = ResidentPhotonTransportStatisticsValue;

      ResidentPhotonTransportStatisticsFunctor<TExecutionSpace> statistics_;
      ProfileProjectionView projection_;
      ProfileAccumulatorView accumulator_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void final(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                             value_type& value) const;
    };
    template <class TExecutionSpace>
    struct ResidentPhotonInteractionStatisticsFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = ResidentPhotonInteractionStatisticsValue;

      Kokkos::View<PhotonFinalStateClassification const*, memory_space_type> finalStates_;
      Kokkos::View<std::uint64_t*, memory_space_type> callStatistics_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void final(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const interactionIndex,
                                             value_type& value) const;
    };
    template <class TExecutionSpace>
    struct ResidentPhotonInteractionStatisticsProfileFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = ResidentPhotonInteractionStatisticsValue;

      ResidentPhotonInteractionStatisticsFunctor<TExecutionSpace> statistics_;
      ProfileProjectionView projection_;
      ProfileAccumulatorView accumulator_;
      Kokkos::View<PhotonTransportRecord*, memory_space_type> steps_;
      std::size_t stepCount_{};
      Kokkos::View<PhotonFinalStateRecord*, memory_space_type> records_;
      std::size_t recordCount_{};

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void final(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const interactionIndex,
                                             value_type& value) const;
    };
  } // namespace photon_cascade_detail
  /** Advance production photon fronts until completion or an explicit checkpoint.
   * All borrowed owners, physics/model arrays and the execution instance must remain
   * valid for the call. The caller handles returned CPU fallbacks and spill particles.
   * Preserve production limits, pending-queue order, history IDs and profile ownership.
   * This operates on the common environment-tree view; it assumes no atmospheric model.
   */
  template <class TExecutionSpace>
  PhotonCascadeResult run_photon_cascade(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      std::vector<ParticleState> const& particles,
      std::uint64_t const firstSecondaryHistoryId, std::size_t const maximumWavefronts,
      std::size_t const minimumResidentBatchSize,
      std::optional<FirstInteractionRecord>& firstInteraction,
      bool const projectSteps = false, ProfileProjectionView const profileProjection = {},
      ProfileAccumulatorStorage<TExecutionSpace>* const profileAccumulator = nullptr,
      PhotonFrontWorkspace<TExecutionSpace>* const workspace = nullptr,
      PendingParticleQueue<TExecutionSpace>* const pendingPhotons = nullptr,
      PendingParticleQueue<TExecutionSpace>* const pendingLeptons = nullptr,
      std::size_t const pendingInputCount = 0,
      std::size_t const maximumPendingParticles = std::numeric_limits<std::size_t>::max(),
      TExecutionSpace const& execution = {},
      bool const captureDiagnosticInteractions = false,
      std::size_t const openmpChunkSize = 0,
      ResidentMemoryBudgetGate const memoryGate = {},
      ResidentExecutionWait<TExecutionSpace>* const cooperativeWait = nullptr);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/PhotonCascade.inl>
