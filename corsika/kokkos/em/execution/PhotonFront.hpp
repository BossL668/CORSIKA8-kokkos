/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/execution/PhotonFrontWorkspace.hpp>
#include <corsika/kokkos/em/execution/PhysicsContext.hpp>
#include <corsika/kokkos/em/algorithms/PhotonTransportStep.hpp>
#include <corsika/kokkos/output/algorithms/ProfileProjection.hpp>
#include <corsika/kokkos/runtime/RangePolicy.hpp>

namespace corsika::kokkos {

  namespace photon_front_detail {
    KOKKOS_INLINE_FUNCTION std::uint64_t photon_fallback_output_offset(
        std::uint32_t const stage, std::uint64_t const selectionOffset,
        std::uint64_t const transportOffset, std::uint64_t const finalStateOffset,
        ResidentPhotonScanValue const& totals);
    template <class TExecutionSpace>
    struct ResidentPhotonScanFunctor {
      using memory_space_type = typename TExecutionSpace::memory_space;
      using value_type = ResidentPhotonScanValue;

      Kokkos::View<ResidentPhotonSourceCounts const*, memory_space_type> counts_;
      Kokkos::View<std::uint64_t * [PhotonOffsetCount], memory_space_type> offsets_;

      KOKKOS_INLINE_FUNCTION void init(value_type& value) const;

      KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                       value_type const& source) const;

      KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source, value_type& update,
                                             bool const final) const;
    };
    template <class TExecutionSpace>
    struct ResidentPhotonCompactionScanFunctor {
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
    struct ResidentPhotonFinalStateWriter {
      ParticleSoARawView next_{};
      RawView1D<PhotonFinalStateRecord> records_{};
      RawView1D<ParticleState> charged_{};
      RawView1D<FallbackRequest> fallbacks_{};
      RawView1D<FirstInteractionRecord> firstSnapshots_{};
      std::uint32_t* firstInteractionCandidates_{};
      std::uint32_t* error_{};
      std::uint64_t nextOffset_{};
      std::uint64_t chargedOffset_{};
      std::uint64_t fallbackOffset_{};
      std::uint64_t recordOffset_{};
      std::uint32_t captureFirstInteraction_{};
      std::uint32_t nextWritten_{};
      std::uint32_t chargedWritten_{};

      KOKKOS_INLINE_FUNCTION void fallback(FallbackRequest const& value);

      KOKKOS_INLINE_FUNCTION void continuation(InteractionRecord const& value);

      KOKKOS_INLINE_FUNCTION void suppression(ParticleState const& parent,
                                              InteractionRecord const&,
                                              PhotonFinalStateParameters const&);

      KOKKOS_INLINE_FUNCTION void firstInteraction(ParticleState const& parent,
                                                   std::int32_t processId,
                                                   ParticleState const& first,
                                                   ParticleState const* second);

      KOKKOS_INLINE_FUNCTION void secondary(std::uint32_t, ParticleState const& particle);

      KOKKOS_INLINE_FUNCTION PhotonFinalStateRecord& record();

      KOKKOS_INLINE_FUNCTION void finishRecord(std::uint32_t);

      KOKKOS_INLINE_FUNCTION void fail(std::uint32_t value);
    };
  } // namespace photon_front_detail
  /** Enqueue one production photon front on the supplied execution instance.
   * The caller allocates both queues and all workspace/candidate arrays, resets
   * error and first-candidate counters, and supplies valid context/table lifetimes.
   * capacity covers next photons and twice that many charged children. History
   * IDs must have room for the complete retained-child scan. No results are
   * downloaded here; read frontControl_ after completion before committing queues.
   * Only environment-tree geometry is used. This is not a whole shower driver.
   */
  template <class TExecutionSpace>
  void enqueue_photon_front(
      typename PhysicsContext<TExecutionSpace>::device_view_type const physicsContext,
      ParticleSoARawView const current, ParticleSoARawView const next,
      PhotonFrontWorkspace<TExecutionSpace>& workspace, std::size_t const currentCount,
      std::size_t const capacity, std::uint64_t const nextHistoryId,
      bool const projectSteps, bool const captureFirstInteraction,
      TExecutionSpace const& execution, std::size_t const openmpChunkSize);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/execution/PhotonFront.inl>
