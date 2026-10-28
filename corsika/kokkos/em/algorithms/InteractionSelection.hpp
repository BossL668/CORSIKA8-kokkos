/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/algorithms/InteractionPhysicsView.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>
#include <corsika/kokkos/em/ProcessCapabilities.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <corsika/kokkos/em/random/RandomDomains.hpp>
#include <corsika/kokkos/em/tables/TableQuery.hpp>
#include <cmath>

namespace corsika::kokkos {

  /** One particle's production selection outcome, before any queue compaction. */
  struct InteractionSelectionOutcome {
    InteractionRecord interaction_{};
    FallbackRequest fallback_{};
    std::uint32_t fallbackFlag_{};
    std::uint32_t nativeNewtonIterations_{};
    std::uint32_t nativeBisectionIterations_{};
    std::uint32_t nativeInverseFailures_{};
  };
  static_assert(std::is_trivially_copyable_v<InteractionSelectionOutcome>);
  /**
   * Sample interaction grammage and preserve production random provenance.
   * Charged particles defer channel selection until after continuous loss. Photon
   * CPU completions may be recorded as pending, but must first reach their vertex.
   * This function advances neither particle position/time nor any execution queue.
   * \param input_index Stable caller index, not an RNG input.
   * \return Interaction/fallback records and the original native inverse counters.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline InteractionSelectionOutcome
  select_discrete_interaction(InteractionPhysicsView const& physics,
                              ParticleState const& particle,
                              std::uint64_t const input_index,
                              std::uint64_t const random_seed,
                              std::uint64_t const shower_id);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/InteractionSelection.inl>
