/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/LeptonFrontWorkspace.hpp>
#include <corsika/kokkos/em/execution/PhysicsContext.hpp>
#include <corsika/kokkos/em/algorithms/LeptonTransportStep.hpp>
#include <corsika/kokkos/em/algorithms/MoliereStep.hpp>
#include <corsika/kokkos/output/algorithms/ProfileProjection.hpp>
#include <corsika/kokkos/runtime/RangePolicy.hpp>

namespace corsika::kokkos {

  namespace lepton_front_detail {
    template <class TExecutionSpace>
    struct ResidentLeptonScanFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = ResidentLeptonScanValue;

      Kokkos::View<ResidentLeptonSourceCounts const*, memory_space_type> counts_;
      Kokkos::View<std::uint64_t * [LeptonOffsetCount], memory_space_type> offsets_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source, value_type& update,
                                             bool const final) const;
    };
    template <class TExecutionSpace>
    struct ResidentLeptonInteractionScanFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = std::uint64_t;

      Kokkos::View<std::uint32_t const*, memory_space_type> flags_;
      Kokkos::View<std::size_t*, memory_space_type> sources_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source, value_type& update,
                                             bool const final) const;
    };
    struct ResidentLeptonFinalStateWriter {
      ParticleSoARawView next_{};
      RawView1D<LeptonFinalStateRecord> records_{};
      RawView1D<ParticleState> photons_{};
      RawView1D<FallbackRequest> fallbacks_{};
      RawView1D<FirstInteractionRecord> firstSnapshots_{};
      std::uint32_t* firstInteractionCandidates_{};
      std::uint32_t* error_{};
      std::uint64_t nextOffset_{};
      std::uint64_t photonOffset_{};
      std::uint64_t fallbackOffset_{};
      std::uint64_t recordOffset_{};
      std::uint32_t captureFirstInteraction_{};
      std::uint32_t nextWritten_{};
      std::uint32_t photonWritten_{};

      KOKKOS_INLINE_FUNCTION void fallback(FallbackRequest const& value);

      KOKKOS_INLINE_FUNCTION void continuation(InteractionRecord const& value);

      KOKKOS_INLINE_FUNCTION void suppression(ParticleState const& parent,
                                              InteractionRecord const&,
                                              LeptonFinalStateParameters const&);

      KOKKOS_INLINE_FUNCTION void firstInteraction(ParticleState const& parent,
                                                   std::int32_t processId,
                                                   ParticleState const& first,
                                                   ParticleState const* second = nullptr,
                                                   ParticleState const* third = nullptr);

      KOKKOS_INLINE_FUNCTION void secondary(std::uint32_t, ParticleState const& particle);

      KOKKOS_INLINE_FUNCTION LeptonFinalStateRecord& record();

      KOKKOS_INLINE_FUNCTION void finishRecord(std::uint32_t);

      KOKKOS_INLINE_FUNCTION void fail(std::uint32_t value);
    };
    KOKKOS_INLINE_FUNCTION std::uint64_t lepton_fallback_output_offset(
        std::uint32_t const stage, std::uint64_t const selectionOffset,
        std::uint64_t const transportOffset, std::uint64_t const vertexOffset,
        std::uint64_t const finalStateOffset, ResidentLeptonScanValue const& totals);
    KOKKOS_INLINE_FUNCTION void classify_lepton_transport_limit(
        LeptonTransportRecord const& step, ResidentLeptonSourceCounts& sourceCounts);
    KOKKOS_INLINE_FUNCTION void classify_lepton_interaction_source(
        InteractionPhysicsView const& physics, BremsLpmSnapshot const& bremsLpm,
        EmThinningConfig const& thinning, std::uint64_t const randomSeed,
        std::uint64_t const showerId, LeptonTransportRecord const& step,
        LeptonVertexSelectionOutcome& vertex, LeptonFinalStateClassification& finalState,
        ResidentLeptonSourceCounts& sourceCounts);
  } // namespace lepton_front_detail
  /** Enqueue one production charged-lepton front without allocation or host waits.
   * The caller reserves source/output and both queue buffers, resets control and
   * first-candidate counters, and keeps all borrowed context owners alive.
   * capacity is the logical next-lepton limit; photonCapacity bounds photon output.
   * History IDs must cover the retained-child scan and the supplied limit must not
   * precede nextHistoryId. After completion inspect control before committing output.
   * Four fallback stages remain separate and ordered. CPU handlers, decay completion,
   * profile/radio accumulation and multi-round scheduling belong to the caller.
   * The fast Moliere specialization requires at most four material components;
   * otherwise use the general branch. Geometry always uses the environment tree.
   */
  template <class TExecutionSpace>
  void enqueue_lepton_front(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      double const electronMassGeV, bool const airMoliereFastPath,
      ParticleSoARawView const current,
      LeptonFrontWorkspace<TExecutionSpace>& activeWorkspace,
      std::size_t const currentCount, std::size_t const capacity,
      std::size_t const photonCapacity, std::uint64_t const nextHistoryId,
      std::uint64_t const secondaryHistoryIdLimitExclusive, bool const projectSteps,
      bool const captureFirstInteraction, TExecutionSpace const& execution,
      std::size_t const openmpChunkSize);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/LeptonFront.inl>
