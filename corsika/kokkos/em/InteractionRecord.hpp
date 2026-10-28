/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/data/ParticleState.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace corsika::kokkos {

  /** Existing interaction-stage labels; declaring them performs no selection. */
  enum class InteractionStatus : std::int32_t {
    Selected = 0,
    NoDiscreteInteraction = 1,
    /**
     * A charged particle reached the sampled interaction grammage after
     * continuous energy loss. Process/component/v must be sampled again at
     * the vertex energy before final-state generation.
     */
    RequiresReselection = 2,
    /**
     * Only the total rate, interaction grammage and process threshold have
     * been sampled. Charged-particle process/component/v selection is
     * deferred until after continuous loss at the vertex.
     */
    DistanceSampled = 3,
    /**
     * The state is already below the configured kinetic-energy cut. The
     * transport kernel emits a zero-length terminal record and deposits the
     * remaining kinetic energy without querying an out-of-domain rate.
     */
    ParticleCut = 4,
    // A stopped positron: kinetic energy has been deposited by transport;
    // two rest-energy photons must still be materialized, without rate/CDF
    // queries and without thinning the already weighted parent again.
    AtRestAnnihilation = 5,
  };

  /** Value-only selected/sampled interaction and its original provenance.
   * Random-domain identifiers and uniforms are copied data, not RNG execution.
   * Units, field order and defaults retain the production record contract. */
  struct InteractionRecord {
    ParticleState particle_{};
    // A photon can select a CPU-only loss/endpoint before transport, but must
    // still compete its sampled flight against geometry and cuts. Negative
    // means no completion is pending; otherwise this is a FallbackReason.
    std::int32_t deferredPhotonFallbackReason_{-1};
    std::int32_t deferredPhotonTableStatus_{};
    std::uint32_t interactionVertexReached_{};
    std::uint64_t inputIndex_{};
    std::int32_t processId_{};
    InteractionStatus status_{InteractionStatus::Selected};
    std::uint64_t componentHash_{};
    double totalRateCm2PerG_{};
    double vertexTotalRateCm2PerG_{};
    double interactionGrammageGPerCm2_{};
    // Local density at the selected interaction vertex. It is intentionally
    // stored in the record because LPM suppression is evaluated after the
    // final-state energy split has been sampled.
    double massDensityGPerCm3_{};
    // Rest mass resolved from the versioned continuous-energy table. Keeping
    // it with the selected interaction lets a shared charged-lepton
    // final-state kernel distinguish e± from μ± without hard-coded host
    // dispatch or an additional device-table lookup.
    double particleMassGeV_{};
    // μ± decay competes with interaction, continuous and geometric limits.
    // The sampled lab-frame distance is retained across the selection and
    // transport kernels so a CPU decay module can be forced at the exact GPU
    // vertex without resampling the already selected lifetime.
    double decayDistanceM_{std::numeric_limits<double>::infinity()};
    double decayUniform_{};
    std::uint64_t decayDrawId_{};
    double energyFraction_{};
    double distanceUniform_{};
    double processUniform_{};
    double lossQuantile_{};
    std::uint64_t distanceDrawId_{};
    std::uint64_t processDrawId_{};
    std::uint64_t lossDrawId_{};
    // Proposal-native reproduces the second, independent scalar PROPOSAL draw
    // and derives lossQuantile_ from the selected rate interval. Legacy
    // c8emrt leaves these fields zero and keeps its existing loss draw.
    double proposalSelectionUniform_{};
    std::uint32_t processRandomProcessId_{};
    std::uint32_t proposalSelectionRandomProcessId_{};
    std::uint64_t proposalSelectionDrawId_{};
  };
  static_assert(std::is_standard_layout_v<InteractionRecord>);
  static_assert(std::is_trivially_copyable_v<InteractionRecord>);

} // namespace corsika::kokkos
