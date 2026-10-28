/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <corsika/kokkos/em/tables/ContinuousPhysicsQueries.hpp>
#include <cstdint>
#include <type_traits>

namespace corsika::kokkos {

  struct TableQuery;

  /** Stable production reason identifiers, including historical inactive reasons.
   * These identifiers do not enable a sampler, a CPU handler or a retry policy. */
  enum class FallbackReason : std::int32_t {
    UnsupportedParticle = 0,
    UnsupportedMedium = 1,
    UnsupportedGeometry = 2,
    ParticleTableMissing = 3,
    ProcessComponentMissing = 4,
    RateEnergyOutOfRange = 5,
    InverseCdfUnavailable = 6,
    LossEnergyOutOfRange = 7,
    LossQuantileOutOfRange = 8,
    InvalidTableQuery = 9,
    DeviceMemorySpill = 10,
    ZeroTotalRate = 11,
    CpuOnlyProcess = 12,
    GpuProcessNotImplemented = 13,
    InvalidFinalState = 14,
    LpmParametersUnavailable = 15,
    InvalidMassDensity = 16,
    ContinuousEnergyOutOfRange = 17,
    ContinuousRangeOutOfRange = 18,
    TransportCutReached = 19,
    MoliereParametersUnavailable = 20,
    MoliereSamplingFailed = 21,
    MagneticTransportFailed = 22,
    MagneticBoundaryFailed = 23,
    AtmosphereGrammageFailed = 24,
    AtmosphereInverseGrammageFailed = 25,
    AtmosphereVertexLookupFailed = 26,
    // The bounded device rejection sampler found a production point above
    // its precomputed Epair envelope.  The interaction type, target and v
    // remain valid, so CPU PROPOSAL may generate only this specified final
    // state.  Keep this separate from InvalidFinalState: the latter is a
    // numerical/physics defect and must not be silently retried.
    EpairRejectionEnvelopeExceeded = 27,
    // The device selected the process/component with the original PROPOSAL
    // uniform, but the conditional interval is too ill-conditioned for the
    // last few host/device interpolation ULPs to guarantee the requested v
    // tolerance.  CPU PROPOSAL replays that same uniform, verifies the
    // selected identity, and completes the final state after the wavefront.
    NativeSelectionReplay = 28,
    // The selected interaction and v are valid; finish rho on the CPU,
    // never run a numerical inverse integral in a device exhaustion branch.
    EpairRejectionTrialsExhausted = 29,
  };

  /** Value-only fallback request; numeric diagnostics do not execute a fallback. */
  struct FallbackRequest {
    ParticleState particle_{};
    // A selected final state is legal only after transport reached its vertex.
    // Process/component hashes alone do not establish that position and time
    // have advanced. Generic pre-transport failures keep this flag zero.
    std::uint32_t interactionVertexReached_{};
    std::uint64_t inputIndex_{};
    std::int32_t processId_{};
    FallbackReason reason_{FallbackReason::InvalidTableQuery};
    // Optional transport diagnostics. These fields are populated only for
    // generic numerical/geometry failures; physics-process fallbacks leave
    // them zero. Keeping the first failing state self-describing avoids
    // silently hiding a rare device transport defect behind a scalar retry.
    std::int32_t diagnosticStatus_{};
    std::uint32_t diagnosticReserved_{};
    double diagnosticValue0_{};
    double diagnosticValue1_{};
    double diagnosticValue2_{};
    std::uint64_t componentHash_{};
    std::uint64_t mediumHash_{};
    std::uint64_t interactionHash_{};
    double energyFraction_{};
    double selectionUniform_{};
    double lossQuantile_{};
    double finalStateUniform_{};
    // Counter-key provenance. selectionUniform_/random_* identify the inner
    // PROPOSAL SampleLoss draw; outer_acceptance_* identify CORSIKA's
    // independent interaction-process acceptance draw.
    double outerAcceptanceUniform_{};
    std::uint32_t randomProcessId_{};
    std::uint32_t outerAcceptanceRandomProcessId_{};
    std::uint64_t randomDrawId_{};
    std::uint64_t outerAcceptanceDrawId_{};
    std::uint64_t finalStateDrawId_{};
  };
  static_assert(std::is_standard_layout_v<FallbackRequest>);
  static_assert(std::is_trivially_copyable_v<FallbackRequest>);

  /** Map the original query status to its unchanged fallback reason. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline FallbackReason proposal_fallback_reason(
      TableLookupStatus status);

  /** Copy the selected process/component and random provenance without draws.
   * The original native-selection discriminator controls which recorded uniform
   * and draw identifiers are copied. No particle or input record is modified. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline FallbackRequest make_process_fallback_event(
      InteractionRecord const& record, FallbackReason reason);

  /** Whether the already selected loss must be completed with CPU PROPOSAL. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool proposal_fallback_requires_selected_loss(
      FallbackReason reason);
  /** Whether a selected reaction must first reach its interaction vertex. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool
  proposal_fallback_requires_interaction_vertex(FallbackReason reason);
  /** Stable host diagnostic name; inactive historical reasons remain identifiable. */
  [[nodiscard]] inline char const* proposal_fallback_reason_name(
      FallbackReason reason) noexcept;
  /** Capture a failed query without claiming transport has reached a vertex.
   * Selected-loss/random provenance not supplied here remains zero-initialized. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline FallbackRequest make_table_fallback_event(
      ParticleState const& particle, TableQuery const& query,
      TableQueryResult const& result, std::uint64_t random_draw_id,
      std::uint64_t input_index = 0);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/FallbackRequest.inl>
