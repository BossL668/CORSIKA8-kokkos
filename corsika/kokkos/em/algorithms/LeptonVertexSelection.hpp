/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/algorithms/InteractionSelection.hpp>
#include <corsika/kokkos/em/algorithms/LeptonContinuousStep.hpp>

namespace corsika::kokkos {

  /** Single-candidate result; the caller owns continuation/interaction/fallback queues.
   */
  struct LeptonVertexSelectionOutcome {
    InteractionRecord record_{};
    FallbackRequest fallback_{};
    std::uint32_t interactionFlag_{};
    std::uint32_t continuationFlag_{};
    std::uint32_t fallbackFlag_{};
    std::uint32_t nativeNewtonIterations_{};
    std::uint32_t nativeBisectionIterations_{};
    std::uint32_t nativeInverseFailures_{};
  };
  static_assert(std::is_trivially_copyable_v<LeptonVertexSelectionOutcome>);
  /**
   * Select a charged-lepton reaction at the supplied post-transport energy.
   * Keeps the outer acceptance draw, independent native channel draw and residual
   * loss quantile in production order. Rejection increments step only on the
   * original continuation branch; overflow is a failure, not a new draw.
   * No final state, CPU handler or position update is executed here.
   * \return Exactly one original branch result (interaction, continuation or fallback).
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline LeptonVertexSelectionOutcome
  select_lepton_vertex(InteractionPhysicsView const& table,
                       InteractionRecord const& candidate,
                       std::uint64_t const random_seed, std::uint64_t const shower_id);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/LeptonVertexSelection.inl>
