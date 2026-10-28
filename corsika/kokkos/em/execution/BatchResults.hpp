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
#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <corsika/kokkos/em/FirstInteractionRecord.hpp>
#include <corsika/kokkos/em/FallbackRequest.hpp>
#include <corsika/kokkos/em/PhotonTransportRecord.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/PhotonFinalStateRecord.hpp>
#include <corsika/kokkos/em/LeptonFinalStateRecord.hpp>
#include <corsika/kokkos/em/PhotonPairLpmSuppressionRecord.hpp>
#include <corsika/kokkos/em/BremsLpmSuppressionRecord.hpp>
#include <corsika/kokkos/output/ObservationRecord.hpp>
#include <corsika/kokkos/output/RadioTrackRecord.hpp>

namespace corsika::kokkos {
  struct BatchResult {
    std::uint64_t wavefrontIndex_{};
    std::size_t inputParticles_{};
    std::size_t outputParticles_{};
    bool queueOverflow_{};
    std::vector<StepRecord> stepRecords_{};
    std::vector<FallbackRequest> fallbackEvents_{};
    std::vector<RadioTrackRecord> radioTracks_{};
    std::vector<ObservationRecord> observations_{};
  };

  struct InteractionBatchResult {
    std::size_t inputParticles_{};
    std::vector<InteractionRecord> interactions_{};
    std::vector<FallbackRequest> fallbackEvents_{};
  };

  struct LeptonVertexSelectionBatchResult {
    std::size_t inputCandidates_{};
    std::vector<InteractionRecord> interactions_{};
    std::vector<InteractionRecord> continuations_{};
    std::vector<FallbackRequest> fallbackEvents_{};
  };

  struct PhotonTransportBatchResult {
    std::size_t inputInteractions_{};
    std::vector<PhotonTransportRecord> records_{};
    std::vector<FallbackRequest> fallbackEvents_{};
  };

  struct LeptonTransportBatchResult {
    std::size_t inputInteractions_{};
    std::vector<LeptonTransportRecord> records_{};
    std::vector<FallbackRequest> fallbackEvents_{};
  };

  struct LeptonFinalStateBatchResult {
    std::size_t inputInteractions_{};
    std::size_t gpuInteractions_{};
    std::size_t bremsInteractions_{};
    std::size_t annihilationInteractions_{};
    std::size_t ionizationInteractions_{};
    std::size_t electronPairInteractions_{};
    std::size_t bremsLpmTrials_{};
    std::size_t bremsLpmSuppressions_{};
    std::size_t electronPairLpmTrials_{};
    std::size_t electronPairLpmSuppressions_{};
    std::size_t electronPairRejectionTrials_{};
    std::size_t electronPairZeroWeightSamples_{};
    std::size_t electronPairRejectionFallbacks_{};
    std::size_t electronPairEnvelopeViolations_{};
    std::vector<LeptonFinalStateRecord> finalStateRecords_{};
    std::vector<ParticleState> secondaries_{};
    std::vector<BremsLpmSuppressionRecord> lpmSuppressed_{};
    std::vector<InteractionRecord> continuations_{};
    std::vector<FallbackRequest> fallbackEvents_{};
  };

  struct PhotonFinalStateBatchResult {
    std::size_t inputInteractions_{};
    std::size_t gpuInteractions_{};
    std::size_t photonPairInteractions_{};
    std::size_t photonMuonPairInteractions_{};
    std::size_t comptonInteractions_{};
    std::size_t photoelectricInteractions_{};
    std::vector<PhotonFinalStateRecord> finalStateRecords_{};
    std::vector<ParticleState> secondaries_{};
    std::vector<PhotonPairLpmSuppressionRecord> lpmSuppressed_{};
    std::vector<InteractionRecord> continuations_{};
    std::vector<FallbackRequest> fallbackEvents_{};
  };

  struct PhotonWavefrontBatchResult {
    std::size_t inputParticles_{};
    std::vector<PhotonTransportRecord> transportRecords_{};
    std::vector<ParticleState> nextPhotons_{};
    std::vector<ObservationRecord> observations_{};
    std::vector<FallbackRequest> selectionFallbackEvents_{};
    std::vector<FallbackRequest> transportFallbackEvents_{};
    PhotonFinalStateBatchResult finalStates_{};
  };
} // namespace corsika::kokkos
