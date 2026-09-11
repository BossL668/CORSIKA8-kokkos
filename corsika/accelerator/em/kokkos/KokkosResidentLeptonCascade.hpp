/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/kokkos/ResidentExecutionWait.hpp>
#include <corsika/accelerator/em/kokkos/KokkosMemorySpace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/LeptonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/LeptonTransportStep.hpp>
#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>
#include <corsika/accelerator/em/detail/MoliereStep.hpp>
#include <corsika/accelerator/em/detail/ProfileProjectionStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontBucketing.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosRangePolicy.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  enum ResidentLeptonFallbackStage : std::uint32_t {
    LeptonNoFallback = 0,
    LeptonSelectionFallback = 1,
    LeptonTransportFallback = 2,
    LeptonVertexFallback = 3,
    LeptonFinalStateFallback = 4,
  };

  struct ResidentLeptonSourceCounts {
    std::uint32_t child{};
    std::uint32_t next{};
    std::uint32_t photon{};
    std::uint32_t fallback_stage{LeptonNoFallback};
    std::uint32_t observation{};
    std::uint32_t decay{};
    std::uint32_t step{};
    std::uint32_t record{};
  };

  enum ResidentLeptonOffset : std::size_t {
    LeptonChildOffset = 0,
    LeptonNextOffset = 1,
    LeptonPhotonOffset = 2,
    LeptonSelectionFallbackOffset = 3,
    LeptonTransportFallbackOffset = 4,
    LeptonVertexFallbackOffset = 5,
    LeptonFinalStateFallbackOffset = 6,
    LeptonObservationOffset = 7,
    LeptonDecayOffset = 8,
    LeptonStepOffset = 9,
    LeptonRecordOffset = 10,
    LeptonOffsetCount = 11,
  };

  struct ResidentLeptonScanValue {
    std::uint64_t values[LeptonOffsetCount]{};
  };

  KOKKOS_INLINE_FUNCTION std::uint64_t leptonFallbackOutputOffset(
      std::uint32_t const stage, std::uint64_t const selection_offset,
      std::uint64_t const transport_offset,
      std::uint64_t const vertex_offset,
      std::uint64_t const final_state_offset,
      ResidentLeptonScanValue const& totals) {
    if (stage == LeptonSelectionFallback) return selection_offset;
    if (stage == LeptonTransportFallback)
      return totals.values[LeptonSelectionFallbackOffset] + transport_offset;
    if (stage == LeptonVertexFallback)
      return totals.values[LeptonSelectionFallbackOffset] +
             totals.values[LeptonTransportFallbackOffset] + vertex_offset;
    if (stage == LeptonFinalStateFallback)
      return totals.values[LeptonSelectionFallbackOffset] +
             totals.values[LeptonTransportFallbackOffset] +
             totals.values[LeptonVertexFallbackOffset] + final_state_offset;
    return ~std::uint64_t{0};
  }

  template <class ExecutionSpace>
  struct ResidentLeptonScanFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentLeptonScanValue;

    Kokkos::View<ResidentLeptonSourceCounts const*, Memory> counts;
    Kokkos::View<std::uint64_t*[LeptonOffsetCount], Memory> offsets;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const { value = {}; }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      for (std::size_t column = 0; column < LeptonOffsetCount; ++column)
        destination.values[column] += source.values[column];
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& update,
                                           bool const final) const {
      if (final) {
        for (std::size_t column = 0; column < LeptonOffsetCount; ++column)
          offsets(source, column) = update.values[column];
      }
      auto const& source_counts = counts(source);
      update.values[LeptonChildOffset] += source_counts.child;
      update.values[LeptonNextOffset] += source_counts.next;
      update.values[LeptonPhotonOffset] += source_counts.photon;
      update.values[LeptonSelectionFallbackOffset] +=
          source_counts.fallback_stage == LeptonSelectionFallback;
      update.values[LeptonTransportFallbackOffset] +=
          source_counts.fallback_stage == LeptonTransportFallback;
      update.values[LeptonVertexFallbackOffset] +=
          source_counts.fallback_stage == LeptonVertexFallback;
      update.values[LeptonFinalStateFallbackOffset] +=
          source_counts.fallback_stage == LeptonFinalStateFallback;
      update.values[LeptonObservationOffset] += source_counts.observation;
      update.values[LeptonDecayOffset] += source_counts.decay;
      update.values[LeptonStepOffset] += source_counts.step;
      update.values[LeptonRecordOffset] += source_counts.record;
    }
  };

  /** Device-to-host control record emitted once per resident wavefront. */
  struct ResidentLeptonFrontControl {
    ResidentLeptonScanValue totals{};
    std::uint64_t interaction_count{};
    std::uint64_t vertex_interaction_count{};
    std::uint32_t materialization_error{};
  };

  static_assert(std::is_standard_layout_v<ResidentLeptonScanValue>);
  static_assert(std::is_trivially_copyable_v<ResidentLeptonScanValue>);
  static_assert(
      sizeof(ResidentLeptonScanValue) ==
      LeptonOffsetCount * sizeof(std::uint64_t));
  static_assert(std::is_standard_layout_v<ResidentLeptonFrontControl>);
  static_assert(std::is_trivially_copyable_v<ResidentLeptonFrontControl>);
  static_assert(offsetof(ResidentLeptonFrontControl, totals) == 0);
  static_assert(
      offsetof(ResidentLeptonFrontControl, interaction_count) ==
      sizeof(ResidentLeptonScanValue));
  static_assert(
      offsetof(ResidentLeptonFrontControl, vertex_interaction_count) ==
      sizeof(ResidentLeptonScanValue) + sizeof(std::uint64_t));
  static_assert(
      offsetof(ResidentLeptonFrontControl, materialization_error) ==
      sizeof(ResidentLeptonScanValue) + 2 * sizeof(std::uint64_t));

  /**
   * Exclusive scan used to compact interaction candidates without changing
   * their source order.  The expensive vertex/final-state stage consumes the
   * resulting source-index list; all other transported leptons bypass it.
   */
  template <class ExecutionSpace>
  struct ResidentLeptonInteractionScanFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = std::uint64_t;

    Kokkos::View<std::uint32_t const*, Memory> flags;
    Kokkos::View<std::size_t*, Memory> sources;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const { value = 0; }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      destination += source;
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& update,
                                           bool const final) const {
      auto const selected = flags(source) != 0;
      if (final && selected)
        sources(static_cast<std::size_t>(update)) = source;
      update += selected;
    }
  };

  enum ResidentLeptonTransportStatistic : std::size_t {
    LeptonTransportInteractionVertices = 0,
    LeptonTransportMoliereTrials,
    LeptonTransportMoliereDeflections,
    LeptonTransportMoliereZeroDeflections,
    LeptonTransportMoliereNewtonIterations,
    LeptonTransportMoliereMaximumNewtonIterations,
    LeptonTransportLimitBase,
    LeptonTransportStatisticCount = LeptonTransportLimitBase + 8,
  };

  struct ResidentLeptonTransportStatisticsValue {
    std::uint64_t values[LeptonTransportStatisticCount]{};
  };

  constexpr std::size_t LeptonCallTransportStatisticBase = 0;

  template <class ExecutionSpace>
  struct ResidentLeptonTransportStatisticsFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentLeptonTransportStatisticsValue;

    Kokkos::View<gpu::em::LeptonTransportRecord const*, Memory> transports;
    Kokkos::View<ResidentLeptonSourceCounts const*, Memory> counts;
    Kokkos::View<std::uint64_t*, Memory> call_statistics;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
      for (std::size_t index = 0; index < LeptonTransportStatisticCount;
           ++index)
        value.values[index] = 0;
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      for (std::size_t index = 0; index < LeptonTransportStatisticCount;
           ++index) {
        if (index == LeptonTransportMoliereMaximumNewtonIterations)
          destination.values[index] =
              destination.values[index] > source.values[index]
                  ? destination.values[index]
                  : source.values[index];
        else
          destination.values[index] += source.values[index];
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& value) const {
      using namespace gpu::em;
      if (counts(source).step != 0) {
        auto const& step = transports(source);
        value.values[LeptonTransportInteractionVertices] +=
            step.limit == LeptonTransportLimit::InteractionCandidate;
        auto const limit_index = static_cast<std::size_t>(step.limit);
        if (limit_index < 8)
          ++value.values[LeptonTransportLimitBase + limit_index];
        // Native CUDA records one Moliere trial for every completed lepton
        // transport step for which multiple scattering is enabled, including
        // zero-grammage boundary/observation steps.  The Kokkos resident path
        // invokes the same stage for every counts(source).step record, so use
        // that exact event definition instead of conditioning the metadata on
        // traversed grammage.
        ++value.values[LeptonTransportMoliereTrials];
        value.values[LeptonTransportMoliereDeflections] +=
            step.multiple_scattering_applied != 0;
        value.values[LeptonTransportMoliereZeroDeflections] +=
            step.multiple_scattering_status ==
            static_cast<std::uint16_t>(MoliereStatus::NoDeflection);
        value.values[LeptonTransportMoliereNewtonIterations] +=
            step.multiple_scattering_iterations;
        auto const iterations =
            static_cast<std::uint64_t>(step.multiple_scattering_iterations);
        if (iterations >
            value.values[LeptonTransportMoliereMaximumNewtonIterations])
          value.values[LeptonTransportMoliereMaximumNewtonIterations] =
              iterations;
      }
    }

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const {
      for (std::size_t index = 0; index < LeptonTransportStatisticCount;
           ++index) {
        auto& accumulated =
            call_statistics(LeptonCallTransportStatisticBase + index);
        if (index == LeptonTransportMoliereMaximumNewtonIterations)
          accumulated = accumulated > value.values[index]
                            ? accumulated
                            : value.values[index];
        else
          accumulated += value.values[index];
      }
    }
  };

  /**
   * Fuse source-wide transport statistics with the stable packed profile-step
   * ledger without changing either index domain.
   */
  template <class ExecutionSpace>
  struct ResidentLeptonTransportStatisticsProfileFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentLeptonTransportStatisticsValue;

    ResidentLeptonTransportStatisticsFunctor<ExecutionSpace> statistics;
    gpu::em::detail::DeviceProfileProjection projection;
    gpu::em::detail::DeviceProfileAccumulator accumulator;
    Kokkos::View<gpu::em::LeptonTransportRecord*, Memory> steps;
    std::size_t step_count{};

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
      statistics.init(value);
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      statistics.join(destination, source);
    }

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const {
      statistics.final(value);
      // The reduction has already computed these two integer diagnostics for
      // exactly the same stable packed step ledger.  Commit one aggregate per
      // wavefront instead of issuing two contended global atomics per lepton.
      KokkosProfileAtomicOperations::add(
          &accumulator.counters->moliere_newton_iterations,
          value.values[LeptonTransportMoliereNewtonIterations]);
      KokkosProfileAtomicOperations::maximum(
          &accumulator.counters->moliere_max_newton_iterations,
          value.values[LeptonTransportMoliereMaximumNewtonIterations]);
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& value) const {
      statistics(source, value);
      if (source < step_count)
        detail::accumulateLeptonProfileStep<KokkosProfileAtomicOperations,
                                            false>(
            projection, accumulator, steps(source));
    }
  };

  enum ResidentLeptonInteractionStatistic : std::size_t {
    LeptonInteractionLpmSuppressions = 0,
    LeptonInteractionGpuFinalStates,
    LeptonInteractionPhysicalSecondaries,
    LeptonInteractionBremsFinalStates,
    LeptonInteractionAnnihilationFinalStates,
    LeptonInteractionIonizationFinalStates,
    LeptonInteractionElectronPairFinalStates,
    LeptonInteractionBremsLpmTrials,
    LeptonInteractionBremsLpmSuppressions,
    LeptonInteractionElectronPairLpmTrials,
    LeptonInteractionElectronPairLpmSuppressions,
    LeptonInteractionElectronPairRejectionTrials,
    LeptonInteractionElectronPairZeroWeightSamples,
    LeptonInteractionElectronPairRejectionFallbacks,
    LeptonInteractionElectronPairEnvelopeViolations,
    LeptonInteractionThinningHillasVertices,
    LeptonInteractionThinningStatisticalVertices,
    LeptonInteractionThinningParticlesDiscarded,
    LeptonInteractionStatisticCount,
  };

  struct ResidentLeptonInteractionStatisticsValue {
    std::uint64_t values[LeptonInteractionStatisticCount]{};
  };

  constexpr std::size_t LeptonCallInteractionStatisticBase =
      LeptonCallTransportStatisticBase + LeptonTransportStatisticCount;
  constexpr std::size_t LeptonCallStatisticCount =
      LeptonCallInteractionStatisticBase + LeptonInteractionStatisticCount;

  template <class ExecutionSpace>
  struct ResidentLeptonInteractionStatisticsFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentLeptonInteractionStatisticsValue;

    Kokkos::View<detail::LeptonVertexSelectionOutcome const*, Memory>
        vertices;
    Kokkos::View<detail::LeptonFinalStateClassification const*, Memory>
        final_states;
    Kokkos::View<std::uint64_t*, Memory> call_statistics;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
      for (std::size_t index = 0; index < LeptonInteractionStatisticCount;
           ++index)
        value.values[index] = 0;
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      for (std::size_t index = 0; index < LeptonInteractionStatisticCount;
           ++index)
        destination.values[index] += source.values[index];
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const candidate,
                                           value_type& value) const {
      using namespace gpu::em;
      // Vertex and final-state storage is intentionally left untouched by
      // the transport stage.  It is valid only for compacted candidates that
      // reached a selected physical interaction.
      if (vertices(candidate).interaction_flag == 0)
        return;
      auto const& final = final_states(candidate);
      value.values[LeptonInteractionLpmSuppressions] +=
          final.suppression_flag;
      value.values[LeptonInteractionGpuFinalStates] += final.record_flag;
      if (final.record_flag != 0) {
        value.values[LeptonInteractionPhysicalSecondaries] +=
            final.child_count;
        value.values[LeptonInteractionBremsFinalStates] += final.brems_flag;
        value.values[LeptonInteractionAnnihilationFinalStates] +=
            final.annihilation_flag;
        value.values[LeptonInteractionIonizationFinalStates] +=
            final.ionization_flag;
        value.values[LeptonInteractionElectronPairFinalStates] +=
            final.electron_pair_flag;
        auto const status =
            static_cast<EmThinningStatus>(final.parameters.thinning_status);
        value.values[LeptonInteractionThinningHillasVertices] +=
            status == EmThinningStatus::Hillas;
        value.values[LeptonInteractionThinningStatisticalVertices] +=
            status == EmThinningStatus::Statistical;
        auto const original_multiplicity =
            final.parameters.process_id == ElectronPairProcessId ? 3U : 2U;
        if (final.child_count <= original_multiplicity)
          value.values[LeptonInteractionThinningParticlesDiscarded] +=
              original_multiplicity - final.child_count;
      }
      value.values[LeptonInteractionBremsLpmTrials] +=
          final.brems_flag + final.brems_suppression_flag;
      value.values[LeptonInteractionBremsLpmSuppressions] +=
          final.brems_suppression_flag;
      value.values[LeptonInteractionElectronPairLpmTrials] +=
          final.electron_pair_flag + final.electron_pair_suppression_flag;
      value.values[LeptonInteractionElectronPairLpmSuppressions] +=
          final.electron_pair_suppression_flag;
      value.values[LeptonInteractionElectronPairRejectionTrials] +=
          final.electron_pair_rejection_trials;
      value.values[LeptonInteractionElectronPairZeroWeightSamples] +=
          final.electron_pair_zero_weight_flag;
      value.values[LeptonInteractionElectronPairRejectionFallbacks] +=
          final.electron_pair_rejection_fallback_flag;
      value.values[LeptonInteractionElectronPairEnvelopeViolations] +=
          final.electron_pair_envelope_violation_flag;
    }

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const {
      for (std::size_t index = 0; index < LeptonInteractionStatisticCount;
           ++index)
        call_statistics(LeptonCallInteractionStatisticBase + index) +=
            value.values[index];
    }
  };

  /** Fuse interaction statistics with the stable packed final-state ledger. */
  template <class ExecutionSpace>
  struct ResidentLeptonInteractionStatisticsProfileFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentLeptonInteractionStatisticsValue;

    ResidentLeptonInteractionStatisticsFunctor<ExecutionSpace> statistics;
    gpu::em::detail::DeviceProfileProjection projection;
    gpu::em::detail::DeviceProfileAccumulator accumulator;
    Kokkos::View<gpu::em::LeptonTransportRecord*, Memory> steps;
    std::size_t step_count{};
    Kokkos::View<gpu::em::BremsFinalStateRecord*, Memory> records;
    std::size_t record_count{};

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
      statistics.init(value);
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      statistics.join(destination, source);
    }

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const {
      statistics.final(value);
    }

    KOKKOS_INLINE_FUNCTION void operator()(
        std::size_t const interaction_index, value_type& value) const {
      statistics(interaction_index, value);
      if (interaction_index < record_count)
        detail::accumulateLeptonProfileFinalState<
            KokkosProfileAtomicOperations>(
            projection, accumulator, steps.data(), step_count,
            records(interaction_index));
    }
  };

  /** Direct writer for one source-ordered resident lepton final state. */
  struct ResidentLeptonFinalStateWriter {
    ParticleSoARawView next{};
    RawView1D<gpu::em::BremsFinalStateRecord> records{};
    RawView1D<gpu::em::EmParticleState> photons{};
    RawView1D<gpu::em::ProposalFallbackEvent> fallbacks{};
    RawView1D<gpu::em::GpuFirstInteractionSnapshot> first_snapshots{};
    std::uint32_t* first_interaction_candidates{};
    std::uint32_t* error{};
    std::uint64_t next_offset{};
    std::uint64_t photon_offset{};
    std::uint64_t fallback_offset{};
    std::uint64_t record_offset{};
    std::uint32_t capture_first_interaction{};
    std::uint32_t next_written{};
    std::uint32_t photon_written{};

    KOKKOS_INLINE_FUNCTION void fallback(
        gpu::em::ProposalFallbackEvent const& value) {
      fallbacks(fallback_offset) = value;
    }

    KOKKOS_INLINE_FUNCTION void continuation(
        gpu::em::EmInteractionRecord const& value) {
      next.store(next_offset, value.particle);
    }

    KOKKOS_INLINE_FUNCTION void suppression(
        gpu::em::EmParticleState const& parent,
        gpu::em::EmInteractionRecord const&,
        detail::LeptonFinalStateParameters const&) {
      next.store(next_offset, parent);
    }

    KOKKOS_INLINE_FUNCTION void firstInteraction(
        gpu::em::EmParticleState const& parent, std::int32_t process_id,
        gpu::em::EmParticleState const& first,
        gpu::em::EmParticleState const* second = nullptr,
        gpu::em::EmParticleState const* third = nullptr) {
      if (capture_first_interaction == 0 || parent.generation != 0) return;
      auto const candidate = Kokkos::atomic_fetch_add(
          first_interaction_candidates, std::uint32_t{1});
      if (candidate != 0) return;
      auto& snapshot = first_snapshots(0);
      snapshot.parent_at_vertex = parent;
      snapshot.process_id = process_id;
      snapshot.secondary_count = third ? 3U : (second ? 2U : 1U);
      snapshot.secondaries[0] = first;
      snapshot.secondaries[0].weight = parent.weight;
      if (second) {
        snapshot.secondaries[1] = *second;
        snapshot.secondaries[1].weight = parent.weight;
      }
      if (third) {
        snapshot.secondaries[2] = *third;
        snapshot.secondaries[2].weight = parent.weight;
      }
    }

    KOKKOS_INLINE_FUNCTION void secondary(
        std::uint32_t, gpu::em::EmParticleState const& particle) {
      if (gpu::em::isChargedLeptonPid(particle.pid)) {
        next.store(next_offset + next_written++, particle);
      } else if (
          particle.pid == static_cast<std::int32_t>(gpu::em::EmPid::Photon)) {
        photons(photon_offset + photon_written++) = particle;
      } else {
        fail(2);
      }
    }

    KOKKOS_INLINE_FUNCTION gpu::em::BremsFinalStateRecord& record() {
      return records(record_offset);
    }

    KOKKOS_INLINE_FUNCTION void finishRecord(std::uint32_t) {}

    KOKKOS_INLINE_FUNCTION void fail(std::uint32_t value) {
      Kokkos::atomic_compare_exchange(error, std::uint32_t{0}, value);
    }
  };

  template <class Destination, class HostView>
  void appendResidentLeptonHostCopy(
      Destination& destination, HostView const& host,
      std::size_t const count) {
    if (count == 0) return;
    auto const old = destination.size();
    destination.resize(old + count);
    for (std::size_t index = 0; index < count; ++index)
      destination[old + index] = host(index);
  }

  /** Device workspace retained by KokkosEmBackend across host wavefronts. */
  template <class ExecutionSpace>
  class KokkosResidentLeptonWorkspace {
  public:
    using Memory = typename ExecutionSpace::memory_space;
    using HostMemory = HostStagingSpace<ExecutionSpace>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using InteractionCountView =
        Kokkos::View<std::uint64_t, Memory, Unmanaged>;
    using ScanTotalsView =
        Kokkos::View<ResidentLeptonScanValue, Memory, Unmanaged>;
    using ErrorView = Kokkos::View<std::uint32_t*, Memory, Unmanaged>;

    using HostProjectedStepView =
        Kokkos::View<gpu::em::ProjectedEmStepRecord*, HostMemory>;
    using HostStepView =
        Kokkos::View<gpu::em::LeptonTransportRecord*, HostMemory>;
    using HostRecordView =
        Kokkos::View<gpu::em::BremsFinalStateRecord*, HostMemory>;
    using HostParticleView =
        Kokkos::View<gpu::em::EmParticleState*, HostMemory>;
    using HostFallbackView =
        Kokkos::View<gpu::em::ProposalFallbackEvent*, HostMemory>;
    using HostObservationView =
        Kokkos::View<gpu::em::ObservationRecord*, HostMemory>;

    /**
     * Reserve the lossless next-front limit without applying that limit to
     * every source and output array.  A lepton source may produce three
     * charged children, but the current source front normally remains much
     * smaller than this logical checkpoint capacity. The queue may reserve a
     * larger geometric allocation, but all front/history checks below retain
     * `requested` as their physical checkpoint.
     */
    void ensureQueueCapacity(std::size_t const requested,
                             ExecutionSpace const& execution) {
      if (requested == 0)
        throw std::invalid_argument(
            "resident Kokkos lepton queue capacity must be positive");
      queue.ensureCapacity(requested, execution);
      ensureDiagnosticViews(execution);
    }

    KokkosMemoryProjection projectedQueueCapacity(
        std::size_t const requested) const {
      if (requested == 0)
        throw std::invalid_argument(
            "resident Kokkos lepton queue capacity must be positive");
      KokkosMemoryProjectionBuilder projection(deviceBytes());
      projection.merge(queue.deviceBytes(), queue.projectedCapacity(requested));
      appendDiagnosticProjection(projection);
      return projection.result();
    }

    /** Grow source-indexed storage only to the largest front actually run. */
    void ensureSourceCapacity(std::size_t const requested,
                              std::size_t const hard_limit,
                              ExecutionSpace const& execution) {
      if (requested == 0 || requested > hard_limit)
        throw std::length_error(
            "resident Kokkos lepton source capacity exceeds its logical limit");
      if (requested <= source_capacity_) return;

      // The preceding wavefront may still have statistics or output copies
      // reading these Views on the same execution-space instance.  Growth is
      // rare, so fence only on the allocation path before replacing them.
      execution.fence("grow resident Kokkos lepton source workspace");
      auto const grown = grownCapacity(source_capacity_, requested, hard_limit);
      selections = decltype(selections)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_selections"),
          grown);
      transports = decltype(transports)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_transports"),
          grown);
      transport_fallbacks = decltype(transport_fallbacks)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_lepton_transport_fallbacks"),
          grown);
      vertices = decltype(vertices)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_vertices"),
          grown);
      final_states = decltype(final_states)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_final_states"),
          grown);
      counts = decltype(counts)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_counts"),
          grown);
      interaction_flags = decltype(interaction_flags)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_lepton_interaction_flags"),
          grown);
      transport_states = decltype(transport_states)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_lepton_transport_states"),
          grown);
      interaction_sources = decltype(interaction_sources)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_lepton_interaction_sources"),
          grown);
      offsets = decltype(offsets)(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             "c8_kokkos_resident_lepton_offsets"),
          grown);
      source_capacity_ = grown;
      ensureDiagnosticViews(execution);
    }

    KokkosMemoryProjection projectedSourceCapacity(
        std::size_t const requested,
        std::size_t const hard_limit) const {
      if (requested == 0 || requested > hard_limit)
        throw std::length_error(
            "resident Kokkos lepton source capacity exceeds its logical limit");
      auto const current = deviceBytes();
      KokkosMemoryProjectionBuilder projection(current);
      if (requested > source_capacity_) {
        auto const grown = grownCapacity(
            source_capacity_, requested, hard_limit);
        replaceViewProjection<detail::InteractionSelectionOutcome>(
            projection, selections, grown);
        replaceViewProjection<gpu::em::LeptonTransportRecord>(
            projection, transports, grown);
        replaceViewProjection<gpu::em::ProposalFallbackEvent>(
            projection, transport_fallbacks, grown);
        replaceViewProjection<detail::LeptonVertexSelectionOutcome>(
            projection, vertices, grown);
        replaceViewProjection<detail::LeptonFinalStateClassification>(
            projection, final_states, grown);
        replaceViewProjection<ResidentLeptonSourceCounts>(
            projection, counts, grown);
        replaceViewProjection<std::uint32_t>(
            projection, interaction_flags, grown);
        replaceViewProjection<std::uint32_t>(
            projection, transport_states, grown);
        replaceViewProjection<std::size_t>(
            projection, interaction_sources, grown);
        projection.replace(
            viewBytes(offsets),
            checkedMemoryMultiply(
                checkedMemoryMultiply(grown, LeptonOffsetCount),
                sizeof(std::uint64_t)));
      }
      appendDiagnosticProjection(projection);
      return projection.result();
    }

    /**
     * Grow materialized output arrays from the source-ordered scan totals.
     * This is called only after the caller has passed the existing workspace
     * and history checkpoints, so allocation cannot change checkpoint state.
     */
    void ensureOutputCapacity(
        std::size_t const step_count, std::size_t const record_count,
        std::size_t const photon_count, std::size_t const fallback_count,
        std::size_t const observation_count, std::size_t const decay_count,
        bool const project_steps, std::size_t const source_hard_limit,
        std::size_t const photon_hard_limit,
        ExecutionSpace const& execution) {
      auto const needs_growth =
          step_count > step_capacity_ || record_count > record_capacity_ ||
          photon_count > photon_capacity_ ||
          fallback_count > fallback_capacity_ ||
          observation_count > observation_capacity_ ||
          decay_count > decay_capacity_ ||
          (project_steps && step_count > projected_step_capacity_);
      if (!needs_growth) return;

      // A preceding front can leave profile/radio work queued when no host
      // output is required.  Synchronize only on the rare growth path before
      // replacing any View still referenced by that work.
      execution.fence("grow resident Kokkos lepton output workspace");
      growView(steps, step_capacity_, step_count, source_hard_limit, execution,
               "c8_kokkos_resident_lepton_steps");
      if (project_steps)
        growView(projected_steps, projected_step_capacity_, step_count,
                 source_hard_limit, execution,
                 "c8_kokkos_resident_lepton_projected_steps");
      growView(records, record_capacity_, record_count, source_hard_limit,
               execution,
               "c8_kokkos_resident_lepton_records");
      growView(photons, photon_capacity_, photon_count, photon_hard_limit,
               execution,
               "c8_kokkos_resident_lepton_photons");
      growView(fallbacks, fallback_capacity_, fallback_count,
               source_hard_limit, execution,
               "c8_kokkos_resident_lepton_fallbacks");
      growView(observations, observation_capacity_, observation_count,
               source_hard_limit, execution,
               "c8_kokkos_resident_lepton_observations");
      growView(decays, decay_capacity_, decay_count, source_hard_limit,
               execution,
               "c8_kokkos_resident_lepton_decays");
    }

    KokkosMemoryProjection projectedOutputCapacity(
        std::size_t const step_count, std::size_t const record_count,
        std::size_t const photon_count, std::size_t const fallback_count,
        std::size_t const observation_count, std::size_t const decay_count,
        bool const project_steps, std::size_t const source_hard_limit,
        std::size_t const photon_hard_limit) const {
      KokkosMemoryProjectionBuilder projection(deviceBytes());
      appendGrowViewProjection<gpu::em::LeptonTransportRecord>(
          projection, steps, step_capacity_, step_count, source_hard_limit);
      if (project_steps)
        appendGrowViewProjection<gpu::em::ProjectedEmStepRecord>(
            projection, projected_steps, projected_step_capacity_, step_count,
            source_hard_limit);
      appendGrowViewProjection<gpu::em::BremsFinalStateRecord>(
          projection, records, record_capacity_, record_count,
          source_hard_limit);
      appendGrowViewProjection<gpu::em::EmParticleState>(
          projection, photons, photon_capacity_, photon_count,
          photon_hard_limit);
      appendGrowViewProjection<gpu::em::ProposalFallbackEvent>(
          projection, fallbacks, fallback_capacity_, fallback_count,
          source_hard_limit);
      appendGrowViewProjection<gpu::em::ObservationRecord>(
          projection, observations, observation_capacity_, observation_count,
          source_hard_limit);
      appendGrowViewProjection<gpu::em::EmParticleState>(
          projection, decays, decay_capacity_, decay_count,
          source_hard_limit);
      return projection.result();
    }

    KokkosMemoryProjection projectedBucketingCapacity(
        std::size_t const requested,
        ExecutionSpace const& execution) const {
      KokkosMemoryProjectionBuilder projection(deviceBytes());
      projection.merge(
          bucketing.deviceBytes(),
          bucketing.projectedCapacity(requested, execution));
      return projection.result();
    }

    void ensureHostOutputCapacity(
        std::size_t const projected_count, std::size_t const step_count,
        std::size_t const record_count, std::size_t const photon_count,
        std::size_t const fallback_count,
        std::size_t const observation_count, std::size_t const decay_count,
        std::size_t const source_hard_limit,
        std::size_t const photon_hard_limit) {
      growHostView(host_projected_steps, projected_count, source_hard_limit,
                   "c8_kokkos_resident_lepton_host_projected_steps");
      growHostView(host_steps, step_count, source_hard_limit,
                   "c8_kokkos_resident_lepton_host_steps");
      growHostView(host_records, record_count, source_hard_limit,
                   "c8_kokkos_resident_lepton_host_records");
      growHostView(host_photons, photon_count, photon_hard_limit,
                   "c8_kokkos_resident_lepton_host_photons");
      growHostView(host_fallbacks, fallback_count, source_hard_limit,
                   "c8_kokkos_resident_lepton_host_fallbacks");
      growHostView(host_observations, observation_count, source_hard_limit,
                   "c8_kokkos_resident_lepton_host_observations");
      growHostView(host_decays, decay_count, source_hard_limit,
                   "c8_kokkos_resident_lepton_host_decays");
    }

    std::size_t capacity() const noexcept { return queue.capacity(); }

    std::size_t deviceBytes() const noexcept {
      auto const view_bytes = [](auto const& view) {
        // A default-constructed rank-zero Kokkos::View can report span()==1
        // even though data()==nullptr and no allocation exists.
        return (view.data() == nullptr ? 0 : view.span()) *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      // error, interaction_count, vertex_interaction_count and scan_totals are
      // unmanaged aliases into front_control and are intentionally omitted.
      return queue.deviceBytes() + bucketing.deviceBytes() +
             view_bytes(selections) + view_bytes(transports) +
             view_bytes(transport_fallbacks) + view_bytes(vertices) +
             view_bytes(final_states) + view_bytes(counts) +
             view_bytes(interaction_flags) + view_bytes(transport_states) +
             view_bytes(interaction_sources) + view_bytes(offsets) +
             view_bytes(steps) + view_bytes(projected_steps) +
             view_bytes(records) + view_bytes(photons) +
             view_bytes(fallbacks) + view_bytes(observations) +
             view_bytes(decays) + view_bytes(first_snapshots) +
             view_bytes(first_interaction_candidates) +
             view_bytes(call_statistics) + view_bytes(front_control);
    }

    KokkosWavefrontQueue<ExecutionSpace> queue{};
    KokkosWavefrontBucketingWorkspace<ExecutionSpace> bucketing{};
    Kokkos::View<detail::InteractionSelectionOutcome*, Memory> selections{};
    Kokkos::View<gpu::em::LeptonTransportRecord*, Memory> transports{};
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory>
        transport_fallbacks{};
    Kokkos::View<detail::LeptonVertexSelectionOutcome*, Memory> vertices{};
    Kokkos::View<detail::LeptonFinalStateClassification*, Memory>
        final_states{};
    Kokkos::View<ResidentLeptonSourceCounts*, Memory> counts{};
    Kokkos::View<std::uint32_t*, Memory> interaction_flags{};
    Kokkos::View<std::uint32_t*, Memory> transport_states{};
    Kokkos::View<std::size_t*, Memory> interaction_sources{};
    Kokkos::View<std::uint64_t*[LeptonOffsetCount], Memory> offsets{};
    Kokkos::View<gpu::em::LeptonTransportRecord*, Memory> steps{};
    Kokkos::View<gpu::em::ProjectedEmStepRecord*, Memory> projected_steps{};
    Kokkos::View<gpu::em::BremsFinalStateRecord*, Memory> records{};
    Kokkos::View<gpu::em::EmParticleState*, Memory> photons{};
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks{};
    Kokkos::View<gpu::em::ObservationRecord*, Memory> observations{};
    Kokkos::View<gpu::em::EmParticleState*, Memory> decays{};
    Kokkos::View<gpu::em::GpuFirstInteractionSnapshot*, Memory>
        first_snapshots{};
    Kokkos::View<std::uint32_t*, Memory> first_interaction_candidates{};
    Kokkos::View<std::uint64_t*, Memory> call_statistics{};
    Kokkos::View<ResidentLeptonFrontControl, Memory> front_control{};
    ErrorView error{};
    InteractionCountView interaction_count{};
    InteractionCountView vertex_interaction_count{};
    ScanTotalsView scan_totals{};
    Kokkos::View<ResidentLeptonFrontControl, HostStagingSpace<ExecutionSpace>>
        host_front_control{};
    HostProjectedStepView host_projected_steps{};
    HostStepView host_steps{};
    HostRecordView host_records{};
    HostParticleView host_photons{};
    HostFallbackView host_fallbacks{};
    HostObservationView host_observations{};
    HostParticleView host_decays{};
    Kokkos::View<std::uint64_t*, HostMemory> host_call_statistics{};
    Kokkos::View<std::uint32_t*, HostMemory>
        host_first_interaction_candidates{};
    Kokkos::View<gpu::em::GpuFirstInteractionSnapshot*, HostMemory>
        host_first_snapshots{};

  private:
    template <class View>
    static std::size_t viewBytes(View const& view) noexcept {
      return (view.data() == nullptr ? 0 : view.span()) *
             sizeof(typename View::value_type);
    }

    template <class Element, class View>
    static void replaceViewProjection(
        KokkosMemoryProjectionBuilder& projection,
        View const& view, std::size_t const extent) {
      projection.replace(
          viewBytes(view), checkedMemoryMultiply(extent, sizeof(Element)));
    }

    template <class Element, class View>
    static void appendGrowViewProjection(
        KokkosMemoryProjectionBuilder& projection,
        View const& view, std::size_t const current,
        std::size_t const requested, std::size_t const hard_limit) {
      if (requested <= current) return;
      replaceViewProjection<Element>(
          projection, view,
          grownCapacity(current, requested, hard_limit));
    }

    void appendDiagnosticProjection(
        KokkosMemoryProjectionBuilder& projection) const {
      if (first_snapshots.extent(0) == 0) {
        projection.replace(0, sizeof(gpu::em::GpuFirstInteractionSnapshot));
        projection.replace(0, sizeof(std::uint32_t));
      }
      if (call_statistics.extent(0) == 0)
        projection.replace(
            0, checkedMemoryMultiply(
                   LeptonCallStatisticCount, sizeof(std::uint64_t)));
      if (front_control.data() == nullptr)
        projection.replace(0, sizeof(ResidentLeptonFrontControl));
    }

    static std::size_t grownCapacity(std::size_t const current,
                                     std::size_t const requested,
                                     std::size_t const hard_limit) {
      if (requested > hard_limit)
        throw std::length_error(
            "resident Kokkos lepton allocation exceeds its logical limit");
      if (requested <= current) return current;
      if (current == 0) return requested;
      if (current > hard_limit / 2) return hard_limit;
      return std::min(hard_limit, std::max(requested, current * 2));
    }

    template <class View>
    static void growView(View& view, std::size_t& current,
                         std::size_t const requested,
                         std::size_t const hard_limit,
                         ExecutionSpace const& execution,
                         char const* const label) {
      if (requested <= current) return;
      auto const grown = grownCapacity(current, requested, hard_limit);
      view = View(
          Kokkos::view_alloc(execution, Kokkos::WithoutInitializing,
                             std::string(label)),
          grown);
      current = grown;
    }

    template <class View>
    static void growHostView(View& view, std::size_t const requested,
                             std::size_t const hard_limit,
                             char const* const label) {
      if (requested <= view.extent(0)) return;
      auto const grown = grownCapacity(view.extent(0), requested, hard_limit);
      View replacement(
          Kokkos::view_alloc(
              Kokkos::WithoutInitializing, std::string(label)),
          grown);
      view = std::move(replacement);
    }

    void ensureDiagnosticViews(ExecutionSpace const& execution) {
      // There can be only one generation-zero projectile in a shower.  Keep
      // the same one-snapshot/candidate-counter layout as native CUDA rather
      // than scaling this diagnostic allocation with the lepton wavefront.
      if (first_snapshots.extent(0) == 0) {
        first_snapshots = decltype(first_snapshots)(
            Kokkos::view_alloc(
                execution, Kokkos::WithoutInitializing,
                "c8_kokkos_resident_lepton_first_snapshot"),
            1);
        first_interaction_candidates =
            decltype(first_interaction_candidates)(
                Kokkos::view_alloc(
                    execution, Kokkos::WithoutInitializing,
                    "c8_kokkos_resident_lepton_first_candidates"),
                1);
      }
      if (call_statistics.extent(0) == 0)
        call_statistics = decltype(call_statistics)(
            Kokkos::view_alloc(
                execution, Kokkos::WithoutInitializing,
                "c8_kokkos_resident_lepton_call_statistics"),
            LeptonCallStatisticCount);
      if (host_call_statistics.extent(0) == 0)
        host_call_statistics = decltype(host_call_statistics)(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "c8_kokkos_resident_lepton_host_call_statistics"),
            LeptonCallStatisticCount);
      if (front_control.data() == nullptr) {
        front_control = decltype(front_control)(
            Kokkos::view_alloc(
                execution, Kokkos::WithoutInitializing,
                "c8_kokkos_resident_lepton_front_control"));
        host_front_control = decltype(host_front_control)(
            "c8_kokkos_resident_lepton_host_front_control");
        auto* const control_bytes = reinterpret_cast<unsigned char*>(
            front_control.data());
        scan_totals = ScanTotalsView(
            reinterpret_cast<ResidentLeptonScanValue*>(
                control_bytes +
                offsetof(ResidentLeptonFrontControl, totals)));
        interaction_count = InteractionCountView(
            reinterpret_cast<std::uint64_t*>(
                control_bytes + offsetof(
                                    ResidentLeptonFrontControl,
                                    interaction_count)));
        vertex_interaction_count = InteractionCountView(
            reinterpret_cast<std::uint64_t*>(
                control_bytes + offsetof(
                                    ResidentLeptonFrontControl,
                                    vertex_interaction_count)));
        error = ErrorView(
            reinterpret_cast<std::uint32_t*>(
                control_bytes + offsetof(
                                    ResidentLeptonFrontControl,
                                    materialization_error)),
            1);
        host_first_interaction_candidates =
            decltype(host_first_interaction_candidates)(
                Kokkos::view_alloc(
                    Kokkos::WithoutInitializing,
                    "c8_kokkos_resident_lepton_host_first_candidates"),
                1);
        host_first_snapshots = decltype(host_first_snapshots)(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "c8_kokkos_resident_lepton_host_first_snapshot"),
            1);
      }
    }

    std::size_t source_capacity_{};
    std::size_t step_capacity_{};
    std::size_t projected_step_capacity_{};
    std::size_t record_capacity_{};
    std::size_t photon_capacity_{};
    std::size_t fallback_capacity_{};
    std::size_t observation_capacity_{};
    std::size_t decay_capacity_{};
  };

  KOKKOS_INLINE_FUNCTION void classifyResidentLeptonTransportLimit(
      gpu::em::LeptonTransportRecord const& step,
      ResidentLeptonSourceCounts& source_counts) {
    using namespace gpu::em;
    source_counts.step = 1;
    switch (step.limit) {
    case LeptonTransportLimit::ContinuousStep:
    case LeptonTransportLimit::LayerBoundary:
    case LeptonTransportLimit::MagneticStep:
      source_counts.next = 1;
      return;
    case LeptonTransportLimit::ObservationSurface:
    case LeptonTransportLimit::EscapedEnvironment:
      source_counts.observation = 1;
      return;
    case LeptonTransportLimit::ParticleCut:
      source_counts.observation =
          step.observation_surface_reached_before_cut != 0;
      return;
    case LeptonTransportLimit::DecayCandidate:
      source_counts.decay = 1;
      return;
    case LeptonTransportLimit::InteractionCandidate:
      break;
    }
  }

  /**
   * Complete only a compacted interaction candidate.  Keeping this stage out
   * of the transport kernel avoids paying the vertex and final-state register
   * footprint for continuous, boundary, observation, cut, decay, and
   * fallback steps.  The caller processes candidates in stable source order,
   * so all history offsets and counter-based RNG keys retain their former
   * meaning.
   */
  KOKKOS_INLINE_FUNCTION void classifyResidentLeptonInteractionSource(
      gpu::em::tables::NativePhysicsView const& physics,
      gpu::em::BremsLpmSnapshot const& brems_lpm,
      gpu::em::EmThinningConfig const& thinning,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      gpu::em::LeptonTransportRecord const& step,
      detail::LeptonVertexSelectionOutcome& vertex,
      detail::LeptonFinalStateClassification& final_state,
      ResidentLeptonSourceCounts& source_counts) {
    using namespace gpu::em;
    vertex = detail::selectLeptonVertex(
        physics, step.interaction, random_seed, shower_id);
    if (vertex.fallback_flag != 0) {
      source_counts.fallback_stage = LeptonVertexFallback;
      return;
    }
    if (vertex.continuation_flag != 0) {
      source_counts.next = 1;
      return;
    }
    if (vertex.interaction_flag == 0) {
      source_counts.fallback_stage = LeptonVertexFallback;
      vertex.fallback = detail::vertexFallback(
          step.interaction, ProposalFallbackReason::InvalidFinalState);
      vertex.fallback_flag = 1;
      return;
    }

    final_state = detail::classifyLeptonFinalState(
        brems_lpm, thinning, vertex.record, random_seed, shower_id);
    auto const& final = final_state;
    source_counts.child = final.child_count;
    source_counts.record = final.record_flag;
    if (final.fallback_flag != 0)
      source_counts.fallback_stage = LeptonFinalStateFallback;
    if (final.continuation_flag || final.suppression_flag) {
      source_counts.next = 1;
      return;
    }
    if (!final.record_flag) return;

    auto const process = final.parameters.process_id;
    if (process == BremsProcessId) {
      source_counts.next =
          (final.parameters.thinning_keep_mask & 0x1U) != 0;
      source_counts.photon =
          (final.parameters.thinning_keep_mask & 0x2U) != 0;
    } else if (process == AnnihilationProcessId) {
      source_counts.photon = final.child_count;
    } else if (process == IonizationProcessId) {
      source_counts.next = final.child_count;
    } else if (process == ElectronPairProcessId) {
      source_counts.next = final.child_count;
    }
  }

  /** Enqueue one production lepton front. No allocation, host readback or
   * fence: all selection, scattering, vertex and output scans stay on the
   * caller's execution instance. The synchronous wrapper uses this too.
   */
  template <class ExecutionSpace>
  void enqueueResidentLeptonFront(
      KokkosPhysicsContextView<ExecutionSpace> const physics_context,
      double const electron_mass_GeV, bool const air_moliere_fast_path,
      ParticleSoARawView const current,
      KokkosResidentLeptonWorkspace<ExecutionSpace>& active_workspace,
      std::size_t const current_count, std::size_t const capacity,
      std::size_t const photon_capacity, std::uint64_t const next_history_id,
      std::uint64_t const secondary_history_id_limit_exclusive,
      bool const project_steps, bool const capture_first_interaction,
      ExecutionSpace const& execution, std::size_t const openmp_chunk_size) {
    using namespace gpu::em;
    auto const Policy = [openmp_chunk_size](ExecutionSpace const& ex,
                                            std::size_t begin, std::size_t end) {
      if (begin != 0 || end < begin)
        throw std::invalid_argument("resident lepton RangePolicy requires a zero origin");
      return makeKokkosRangePolicy(ex, end, openmp_chunk_size);
    };
    auto& queue = active_workspace.queue;
    auto const& selections = active_workspace.selections;
    auto const& transports = active_workspace.transports;
    auto const& transport_fallbacks = active_workspace.transport_fallbacks;
    auto const& vertices = active_workspace.vertices;
    auto const& final_states = active_workspace.final_states;
    auto const& counts = active_workspace.counts;
    auto const& interaction_flags = active_workspace.interaction_flags;
    auto const& transport_states = active_workspace.transport_states;
    auto const& interaction_sources = active_workspace.interaction_sources;
    auto const& offsets = active_workspace.offsets;
    auto const& steps = active_workspace.steps;
    auto const& projected_steps = active_workspace.projected_steps;
    auto const& records = active_workspace.records;
    auto const& photons = active_workspace.photons;
    auto const& fallbacks = active_workspace.fallbacks;
    auto const& observations = active_workspace.observations;
    auto const& decays = active_workspace.decays;
    auto const& first_snapshots = active_workspace.first_snapshots;
    auto const& first_interaction_candidates =
        active_workspace.first_interaction_candidates;
    auto const& error = active_workspace.error;
    auto const interaction_count_device = active_workspace.interaction_count;
    auto const scan_totals_device = active_workspace.scan_totals;
    auto const front_control_device = active_workspace.front_control;

    // Keep rate inversion separate from geometry and multiple scattering.
    // The three kernels execute in source order on the same execution-space
    // instance, so splitting their register frames does not change source
    // indices, history allocation, or any history-keyed Philox draw.
    auto const selections_raw = rawDeviceView(selections);
    auto const transports_raw = rawDeviceView(transports);
    auto const transport_fallbacks_raw =
        rawDeviceView(transport_fallbacks);
    auto const vertices_raw = rawDeviceView(vertices);
    auto const final_states_raw = rawDeviceView(final_states);
    auto const counts_raw = rawDeviceView(counts);
    auto const interaction_flags_raw = rawDeviceView(interaction_flags);
    auto const transport_states_raw = rawDeviceView(transport_states);
    auto const interaction_sources_raw = rawDeviceView(interaction_sources);
    auto const select_kernel =
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const& context = physics_context();
          ResidentLeptonSourceCounts source_counts{};
          auto& selection = selections_raw(source);
          selection = detail::selectDiscreteInteraction(
              context.physics, current.load(source), source,
              context.random_seed, context.shower_id);
          if (selection.fallback_flag != 0)
            source_counts.fallback_stage = LeptonSelectionFallback;
          counts_raw(source) = source_counts;
          interaction_flags_raw(source) = 0;
          transport_states_raw(source) =
              selection.fallback_flag != 0 ? 1U : 0U;
        };
    static_assert(sizeof(select_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_lepton_select",
        Policy(execution, 0, current_count),
        select_kernel);

    auto const transport_kernel =
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const& context = physics_context();
          auto const& selection = selections_raw(source);
          if (selection.fallback_flag != 0) return;
          transport_states_raw(source) = detail::transportLepton(
              context.physics, true, context.environment,
              selection.interaction, transports_raw(source),
              transport_fallbacks_raw(source));
        };
    static_assert(sizeof(transport_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_lepton_transport",
        Policy(execution, 0, current_count),
        transport_kernel);

    if (air_moliere_fast_path) {
      auto const moliere_air_kernel =
          KOKKOS_LAMBDA(std::size_t const source) {
            auto const& context = physics_context();
            auto const& selection = selections_raw(source);
            auto& transport = transports_raw(source);
            auto& transport_fallback = transport_fallbacks_raw(source);
            auto& source_counts = counts_raw(source);
            if (selection.fallback_flag != 0) return;
            auto const state = detail::applyMoliereScatteringStage<4>(
                context.electron_moliere, context.muon_moliere,
                context.moliere_interpolation,
                context.muon_moliere_available != 0, context.random_seed,
                context.shower_id, transport, transport_fallback,
                transport_states_raw(source));
            transport_states_raw(source) = state;
            if (state != 0) {
              source_counts.fallback_stage = LeptonTransportFallback;
              return;
            }
            classifyResidentLeptonTransportLimit(transport, source_counts);
            interaction_flags_raw(source) =
                transport.limit ==
                LeptonTransportLimit::InteractionCandidate;
          };
      static_assert(sizeof(moliere_air_kernel) < 512);
      Kokkos::parallel_for(
          "c8_kokkos_resident_lepton_moliere_air",
          Policy(execution, 0, current_count),
          moliere_air_kernel);
    } else {
      auto const moliere_general_kernel =
          KOKKOS_LAMBDA(std::size_t const source) {
            auto const& context = physics_context();
            auto const& selection = selections_raw(source);
            auto& transport = transports_raw(source);
            auto& transport_fallback = transport_fallbacks_raw(source);
            auto& source_counts = counts_raw(source);
            if (selection.fallback_flag != 0) return;
            auto const state = detail::applyMoliereScatteringStage<
                MaxMoliereComponents>(
                context.electron_moliere, context.muon_moliere,
                context.moliere_interpolation,
                context.muon_moliere_available != 0, context.random_seed,
                context.shower_id, transport, transport_fallback,
                transport_states_raw(source));
            transport_states_raw(source) = state;
            if (state != 0) {
              source_counts.fallback_stage = LeptonTransportFallback;
              return;
            }
            classifyResidentLeptonTransportLimit(transport, source_counts);
            interaction_flags_raw(source) =
                transport.limit ==
                LeptonTransportLimit::InteractionCandidate;
          };
      static_assert(sizeof(moliere_general_kernel) < 512);
      Kokkos::parallel_for(
          "c8_kokkos_resident_lepton_moliere_general",
          Policy(execution, 0, current_count),
          moliere_general_kernel);
    }

    // Stable interaction-candidate compaction.  This preserves increasing
    // source order exactly, while ensuring that vertex selection and the
    // comparatively register-heavy final-state classifier run only for
    // actual interaction limits.
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_interactions",
        Policy(execution, 0, current_count),
        ResidentLeptonInteractionScanFunctor<ExecutionSpace>{
            interaction_flags, interaction_sources},
        active_workspace.interaction_count);
    auto const interaction_count_raw = interaction_count_device.data();
    auto const vertex_interaction_count_device =
        active_workspace.vertex_interaction_count;
    auto const classify_kernel =
        KOKKOS_LAMBDA(std::size_t const candidate,
                      std::uint64_t& vertex_interactions) {
          auto const& context = physics_context();
          auto const compacted_count =
              static_cast<std::size_t>(*interaction_count_raw);
          if (candidate >= compacted_count) return;
          auto const source = interaction_sources_raw(candidate);
          auto& source_counts = counts_raw(source);
          classifyResidentLeptonInteractionSource(
              context.physics, context.brems_lpm, context.thinning,
              context.random_seed, context.shower_id,
              transports_raw(source), vertices_raw(candidate),
              final_states_raw(candidate), source_counts);
          vertex_interactions +=
              vertices_raw(candidate).interaction_flag != 0;
        };
    static_assert(sizeof(classify_kernel) < 512);
    Kokkos::parallel_reduce(
        "c8_kokkos_classify_lepton_interactions",
        Policy(execution, 0, current_count),
        classify_kernel, vertex_interaction_count_device);

    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_outputs",
        Policy(execution, 0, current_count),
        ResidentLeptonScanFunctor<ExecutionSpace>{counts, offsets},
        active_workspace.scan_totals);

    auto const next = queue.next().rawDeviceView();
    auto const offsets_raw = rawDeviceView(offsets);
    auto const steps_raw = rawDeviceView(steps);
    auto const projected_steps_raw = rawDeviceView(projected_steps);
    auto const records_raw = rawDeviceView(records);
    auto const photons_raw = rawDeviceView(photons);
    auto const fallbacks_raw = rawDeviceView(fallbacks);
    auto const observations_raw = rawDeviceView(observations);
    auto const decays_raw = rawDeviceView(decays);
    auto const first_snapshots_raw = rawDeviceView(first_snapshots);
    auto const first_interaction_candidates_raw =
        first_interaction_candidates.data();
    auto const scan_totals_raw = scan_totals_device.data();
    auto const error_raw = error.data();
    auto const materialize_endpoint =
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const device_totals = *scan_totals_raw;
          if (device_totals.values[LeptonNextOffset] > capacity ||
              device_totals.values[LeptonPhotonOffset] > photon_capacity ||
              device_totals.values[LeptonChildOffset] >
                  secondary_history_id_limit_exclusive - next_history_id)
            return;
          auto const& selection = selections_raw(source);
          auto const& transport = transports_raw(source);
          auto const& source_counts = counts_raw(source);
          if (source_counts.fallback_stage > LeptonFinalStateFallback) {
            Kokkos::atomic_compare_exchange(
                error_raw, std::uint32_t{0}, std::uint32_t{31});
            return;
          }
          if (source_counts.step) {
            steps_raw(offsets_raw(source, LeptonStepOffset)) =
                transport;
            if (project_steps) {
              auto const& context = physics_context();
              projected_steps_raw(
                  offsets_raw(source, LeptonStepOffset)) =
                  detail::projectLeptonStep(
                      context.profile_projection, transport);
            }
          }
          if (selection.fallback_flag) {
            if (source_counts.fallback_stage !=
                LeptonSelectionFallback) {
              Kokkos::atomic_compare_exchange(
                  error_raw, std::uint32_t{0}, std::uint32_t{32});
              return;
            }
            fallbacks_raw(
                leptonFallbackOutputOffset(
                    source_counts.fallback_stage,
                    offsets_raw(source, LeptonSelectionFallbackOffset),
                    offsets_raw(source, LeptonTransportFallbackOffset),
                    offsets_raw(source, LeptonVertexFallbackOffset),
                    offsets_raw(source, LeptonFinalStateFallbackOffset),
                    device_totals)) =
                selection.fallback;
            return;
          }
          if (!source_counts.step) {
            if (source_counts.fallback_stage !=
                LeptonTransportFallback) {
              Kokkos::atomic_compare_exchange(
                  error_raw, std::uint32_t{0}, std::uint32_t{33});
              return;
            }
            fallbacks_raw(
                leptonFallbackOutputOffset(
                    source_counts.fallback_stage,
                    offsets_raw(source, LeptonSelectionFallbackOffset),
                    offsets_raw(source, LeptonTransportFallbackOffset),
                    offsets_raw(source, LeptonVertexFallbackOffset),
                    offsets_raw(source, LeptonFinalStateFallbackOffset),
                    device_totals)) =
                transport_fallbacks_raw(source);
            return;
          }

          auto const& step = transport;
          if (source_counts.observation) {
            ObservationRecord observation{};
            observation.particle = step.end;
            observation.status =
                step.limit == LeptonTransportLimit::EscapedEnvironment
                    ? ObservationStatus::EscapedEnvironment
                    : ObservationStatus::ReachedObservationSurface;
            observations_raw(
                offsets_raw(source, LeptonObservationOffset)) =
                observation;
          }
          if (source_counts.decay)
            decays_raw(offsets_raw(source, LeptonDecayOffset)) =
                step.end;
          if (step.limit == LeptonTransportLimit::ContinuousStep ||
              step.limit == LeptonTransportLimit::LayerBoundary ||
              step.limit == LeptonTransportLimit::MagneticStep) {
            next.store(offsets_raw(source, LeptonNextOffset),
                       step.end);
          }
        };
    static_assert(sizeof(materialize_endpoint) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_lepton_materialize_endpoints",
        Policy(execution, 0, current_count), materialize_endpoint);

    auto const materialize_kernel =
        KOKKOS_LAMBDA(std::size_t const candidate) {
          auto const device_totals = *scan_totals_raw;
          if (device_totals.values[LeptonNextOffset] > capacity ||
              device_totals.values[LeptonPhotonOffset] > photon_capacity ||
              device_totals.values[LeptonChildOffset] >
                  secondary_history_id_limit_exclusive - next_history_id)
            return;
          auto const compacted_count =
              static_cast<std::size_t>(*interaction_count_raw);
          if (candidate >= compacted_count) return;
          auto const source = interaction_sources_raw(candidate);
          auto const& vertex = vertices_raw(candidate);
          auto const& final_state = final_states_raw(candidate);
          auto const& source_counts = counts_raw(source);
          if (vertex.fallback_flag) {
            if (source_counts.fallback_stage != LeptonVertexFallback) {
              Kokkos::atomic_compare_exchange(
                  error_raw, std::uint32_t{0}, std::uint32_t{34});
              return;
            }
            fallbacks_raw(
                leptonFallbackOutputOffset(
                    source_counts.fallback_stage,
                    offsets_raw(source, LeptonSelectionFallbackOffset),
                    offsets_raw(source, LeptonTransportFallbackOffset),
                    offsets_raw(source, LeptonVertexFallbackOffset),
                    offsets_raw(source, LeptonFinalStateFallbackOffset),
                    device_totals)) =
                vertex.fallback;
            return;
          }
          if (vertex.continuation_flag) {
            if (source_counts.fallback_stage != LeptonNoFallback) {
              Kokkos::atomic_compare_exchange(
                  error_raw, std::uint32_t{0}, std::uint32_t{35});
              return;
            }
            next.store(offsets_raw(source, LeptonNextOffset),
                       vertex.record.particle);
            return;
          }
          if ((final_state.fallback_flag != 0) !=
              (source_counts.fallback_stage ==
               LeptonFinalStateFallback)) {
            Kokkos::atomic_compare_exchange(
                error_raw, std::uint32_t{0}, std::uint32_t{36});
            return;
          }

          ResidentLeptonFinalStateWriter output{
              next,
              records_raw,
              photons_raw,
              fallbacks_raw,
              first_snapshots_raw,
              first_interaction_candidates_raw,
              error_raw,
              offsets_raw(source, LeptonNextOffset),
              offsets_raw(source, LeptonPhotonOffset),
              leptonFallbackOutputOffset(
                  source_counts.fallback_stage,
                  offsets_raw(source, LeptonSelectionFallbackOffset),
                  offsets_raw(source, LeptonTransportFallbackOffset),
                  offsets_raw(source, LeptonVertexFallbackOffset),
                  offsets_raw(source, LeptonFinalStateFallbackOffset),
                  device_totals),
              offsets_raw(source, LeptonRecordOffset),
              capture_first_interaction ? 1U : 0U};
          detail::materializeLeptonFinalStateInPlace(
              vertex.record, final_state,
              offsets_raw(source, LeptonChildOffset),
              next_history_id, electron_mass_GeV, output);
        };
    static_assert(sizeof(materialize_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_lepton_materialize_interactions",
        Policy(execution, 0, current_count),
        materialize_kernel);

  }

  /** Same post-materialization profile/statistics/radio kernels as the
   * synchronous cascade. The caller has validated control and reserved radio
   * workspace before any endpoint is in flight. No host result is consumed.
   */
  template <class ExecutionSpace>
  void enqueueResidentLeptonAccumulation(
      KokkosResidentLeptonWorkspace<ExecutionSpace> const& workspace,
      std::size_t const current_count,
      ResidentLeptonFrontControl const& control,
      gpu::em::detail::DeviceProfileProjection const profile_projection,
      KokkosProfileAccumulator<ExecutionSpace>* const profile_accumulator,
      radio::kokkos_detail::KokkosRadioAccumulator<ExecutionSpace>* const radio_accumulator,
      ExecutionSpace const& execution, std::size_t const openmp_chunk_size) {
    auto const Policy = [openmp_chunk_size](ExecutionSpace const& ex,
                                            std::size_t, std::size_t end) {
      return makeKokkosRangePolicy(ex, end, openmp_chunk_size);
    };
    auto const& transports = workspace.transports;
    auto const& counts = workspace.counts;
    auto const& call_statistics = workspace.call_statistics;
    auto const& vertices = workspace.vertices;
    auto const& final_states = workspace.final_states;
    auto const& steps = workspace.steps;
    auto const& records = workspace.records;
    auto const& totals = control.totals.values;
    auto const interaction_count = static_cast<std::size_t>(control.interaction_count);
    auto const resident_step_count = static_cast<std::size_t>(totals[LeptonStepOffset]);
    auto const resident_record_count = static_cast<std::size_t>(totals[LeptonRecordOffset]);
    if (profile_accumulator != nullptr) {
      auto const device_profile = profile_accumulator->deviceView();
      Kokkos::parallel_reduce(
          "c8_kokkos_resident_lepton_transport_statistics",
          Policy(execution, 0, current_count),
          ResidentLeptonTransportStatisticsProfileFunctor<ExecutionSpace>{
              {transports, counts, call_statistics},
              profile_projection,
              device_profile,
              steps,
              resident_step_count});
      if (interaction_count != 0)
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_lepton_interaction_statistics",
            Policy(execution, 0, interaction_count),
            ResidentLeptonInteractionStatisticsProfileFunctor<
                ExecutionSpace>{{vertices, final_states, call_statistics},
                                profile_projection,
                                device_profile,
                                steps,
                                resident_step_count,
                                records,
                                resident_record_count});
    } else {
      Kokkos::parallel_reduce(
          "c8_kokkos_resident_lepton_transport_statistics",
          Policy(execution, 0, current_count),
          ResidentLeptonTransportStatisticsFunctor<ExecutionSpace>{
              transports, counts, call_statistics});
      if (interaction_count != 0)
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_lepton_interaction_statistics",
            Policy(execution, 0, interaction_count),
            ResidentLeptonInteractionStatisticsFunctor<ExecutionSpace>{
                vertices, final_states, call_statistics});
    }
    if (radio_accumulator != nullptr)
      radio_accumulator->accumulateLeptonTracks(
          steps, totals[LeptonStepOffset], execution);
  }

  template <class ExecutionSpace>
  gpu::em::ResidentLeptonCascadeResult runResidentLeptonCascade(
      KokkosPhysicsContextView<ExecutionSpace> const physics_context,
      double const electron_mass_GeV,
      bool const air_moliere_fast_path,
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t const first_secondary_history_id,
      std::size_t const maximum_wavefronts,
      std::uint64_t const secondary_history_id_limit_exclusive,
      std::size_t const minimum_resident_batch_size,
      std::optional<gpu::em::GpuFirstInteractionSnapshot>& first_interaction,
      bool const project_steps = false,
      gpu::em::detail::DeviceProfileProjection const profile_projection = {},
      KokkosProfileAccumulator<ExecutionSpace>* const profile_accumulator =
          nullptr,
      radio::kokkos_detail::KokkosRadioAccumulator<ExecutionSpace>* const
          radio_accumulator = nullptr,
      KokkosResidentLeptonWorkspace<ExecutionSpace>* const workspace =
          nullptr,
      KokkosPendingParticleQueue<ExecutionSpace>* const pending_leptons =
          nullptr,
      KokkosPendingParticleQueue<ExecutionSpace>* const pending_photons =
          nullptr,
      std::size_t const pending_input_count = 0,
      std::size_t const maximum_pending_particles =
          std::numeric_limits<std::size_t>::max(),
      ExecutionSpace const& execution = {},
      bool const capture_diagnostic_interactions = false,
      std::size_t const openmp_chunk_size = 0,
      KokkosResidentMemoryBudgetGate const memory_gate = {},
      std::size_t const reserved_source_limit = 0,
      ResidentExecutionWait<ExecutionSpace>* const cooperative_wait = nullptr) {
    using namespace gpu::em;
    auto const Policy =
        [openmp_chunk_size](ExecutionSpace const& selected_execution,
                            std::size_t const begin,
                            std::size_t const end) {
          if (begin != 0 || end < begin)
            throw std::invalid_argument(
                "resident lepton RangePolicy requires a zero origin");
          return makeKokkosRangePolicy(
              selected_execution, end, openmp_chunk_size);
        };
    ResidentLeptonCascadeResult result{};
    if ((pending_leptons == nullptr && pending_input_count != 0) ||
        (pending_leptons != nullptr &&
         pending_input_count > pending_leptons->size()))
      throw std::invalid_argument(
          "resident Kokkos lepton pending-input count is invalid");
    auto const pending_input = pending_input_count;
    if (pending_input > std::numeric_limits<std::size_t>::max() -
                            particles.size())
      throw std::overflow_error(
          "resident Kokkos lepton input size overflow");
    auto const total_input = pending_input + particles.size();
    result.input_particles = total_input;
    if (total_input == 0) {
      result.completed = true;
      return result;
    }
    if (first_secondary_history_id == 0 || maximum_wavefronts == 0 ||
        secondary_history_id_limit_exclusive <= first_secondary_history_id ||
        minimum_resident_batch_size == 0)
      throw std::invalid_argument(
          "resident Kokkos lepton cascade requires valid limits");
    if (total_input > std::numeric_limits<std::size_t>::max() / 3)
      throw std::length_error(
          "resident Kokkos lepton input exceeds addressable capacity");

    // A source can produce at most three charged children.  Keeping three
    // input fronts gives ordinary Epair growth room while retaining a strict,
    // lossless checkpoint if a later front exceeds this per-call workspace.
    auto const capacity = total_input * 3;
    auto const photon_capacity =
        capacity > std::numeric_limits<std::size_t>::max() / 2
            ? std::numeric_limits<std::size_t>::max()
            : capacity * 2;
    KokkosResidentLeptonWorkspace<ExecutionSpace> local_workspace;
    auto& active_workspace = workspace != nullptr ? *workspace : local_workspace;
    auto spill_unstaged_input = [&]() {
      result.workspace_limit_checkpoint = true;
      result.completed = true;
      result.cpu_spill_particles.reserve(total_input);
      if (pending_input != 0) {
        auto prefix = pending_leptons->downloadPrefix(
            pending_input, execution);
        result.cpu_spill_particles.insert(
            result.cpu_spill_particles.end(),
            std::make_move_iterator(prefix.begin()),
            std::make_move_iterator(prefix.end()));
        pending_leptons->consume(pending_input);
      }
      result.cpu_spill_particles.insert(
          result.cpu_spill_particles.end(), particles.begin(), particles.end());
    };
    auto const queue_projection =
        active_workspace.projectedQueueCapacity(capacity);
    if (!memory_gate.permits(
            active_workspace.deviceBytes(), queue_projection,
            "Kokkos lepton resident queue growth")) {
      spill_unstaged_input();
      return result;
    }
    active_workspace.ensureQueueCapacity(capacity, execution);
    auto const initial_source_projection =
        active_workspace.projectedSourceCapacity(total_input, capacity);
    if (!memory_gate.permits(
            active_workspace.deviceBytes(), initial_source_projection,
            "Kokkos lepton source workspace growth")) {
      spill_unstaged_input();
      return result;
    }
    active_workspace.ensureSourceCapacity(total_input, capacity, execution);
    auto const& call_statistics = active_workspace.call_statistics;
    Kokkos::deep_copy(execution, call_statistics, std::uint64_t{0});
    auto& queue = active_workspace.queue;
    KokkosWavefrontExecutionGuard<ExecutionSpace> queue_execution_guard(
        queue, execution);
    auto spill_active_front = [&]() {
      auto active = queue.download(execution);
      result.cpu_spill_particles.insert(
          result.cpu_spill_particles.end(),
          std::make_move_iterator(active.begin()),
          std::make_move_iterator(active.end()));
      queue.clear();
      result.workspace_limit_checkpoint = true;
    };
    if (pending_input != 0) {
      queue.upload(particles, pending_leptons->view(),
                   pending_leptons->head(), pending_input, execution);
      pending_leptons->consume(pending_input);
    } else {
      queue.upload(particles, execution);
    }
    auto const& selections = active_workspace.selections;
    auto const& transports = active_workspace.transports;
    auto const& transport_fallbacks = active_workspace.transport_fallbacks;
    auto const& vertices = active_workspace.vertices;
    auto const& final_states = active_workspace.final_states;
    auto const& counts = active_workspace.counts;
    auto const& interaction_flags = active_workspace.interaction_flags;
    auto const& transport_states = active_workspace.transport_states;
    auto const& interaction_sources = active_workspace.interaction_sources;
    auto const& offsets = active_workspace.offsets;
    auto const& steps = active_workspace.steps;
    auto const& projected_steps = active_workspace.projected_steps;
    auto const& records = active_workspace.records;
    auto const& photons = active_workspace.photons;
    auto const& fallbacks = active_workspace.fallbacks;
    auto const& observations = active_workspace.observations;
    auto const& decays = active_workspace.decays;
    auto const& first_snapshots = active_workspace.first_snapshots;
    auto const& first_interaction_candidates =
        active_workspace.first_interaction_candidates;
    auto const& error = active_workspace.error;
    auto const interaction_count_device = active_workspace.interaction_count;
    auto const scan_totals_device = active_workspace.scan_totals;
    auto const front_control_device = active_workspace.front_control;
    auto const host_front_control = active_workspace.host_front_control;

    std::uint64_t next_history_id = first_secondary_history_id;
    result.peak_resident_leptons = queue.size();
    auto const capture_first_interaction = !first_interaction.has_value();
    if (capture_first_interaction)
      Kokkos::deep_copy(
          execution, first_interaction_candidates, std::uint32_t{0});
    while (!queue.empty() && result.wavefronts < maximum_wavefronts) {
      auto const current_count = queue.size();
      if (reserved_source_limit != 0 && current_count > reserved_source_limit) {
        // The 3x child queue can hold this front, but the budgeted source and
        // radio arenas cannot. Return it through the existing lossless
        // checkpoint for GPU re-batching, before drawing any random numbers.
        result.workspace_limit_checkpoint = true;
        break;
      }
      if (result.wavefronts != 0 &&
          current_count < minimum_resident_batch_size) {
        result.below_minimum_batch_checkpoint = true;
        break;
      }
      auto const remaining_history_ids =
          secondary_history_id_limit_exclusive - next_history_id;
      if (current_count > remaining_history_ids / 3) {
        result.history_range_exhausted = true;
        break;
      }
      auto const source_projection =
          active_workspace.projectedSourceCapacity(current_count, capacity);
      if (!memory_gate.permits(
              active_workspace.deviceBytes(), source_projection,
              "Kokkos lepton source workspace growth")) {
        spill_active_front();
        break;
      }
      active_workspace.ensureSourceCapacity(current_count, capacity, execution);
      auto const maximum_front_photons = std::min(
          photon_capacity,
          current_count > std::numeric_limits<std::size_t>::max() / 2
              ? std::numeric_limits<std::size_t>::max()
              : current_count * 2);
      // Allocate against the source-count upper bounds before the scans.  The
      // scan totals can consequently stay device resident through endpoint
      // materialization instead of fencing merely to size these Views.
      auto const output_projection =
          active_workspace.projectedOutputCapacity(
              current_count, current_count, maximum_front_photons,
              current_count, current_count, current_count, project_steps,
              capacity, photon_capacity);
      if (!memory_gate.permits(
              active_workspace.deviceBytes(), output_projection,
              "Kokkos lepton output workspace growth")) {
        spill_active_front();
        break;
      }
      active_workspace.ensureOutputCapacity(
          current_count, current_count, maximum_front_photons, current_count,
          current_count, current_count, project_steps, capacity,
          photon_capacity, execution);
      auto const bucketing_projection =
          active_workspace.projectedBucketingCapacity(
              current_count, execution);
      if (!memory_gate.permits(
              active_workspace.deviceBytes(), bucketing_projection,
              "Kokkos lepton bucketing workspace growth")) {
        spill_active_front();
        break;
      }
      if (radio_accumulator != nullptr) {
        auto const radio_projection =
            radio_accumulator->projectedTrackCapacity(current_count);
        if (!memory_gate.permits(
                radio_accumulator->deviceBytes(), radio_projection,
                "Kokkos radio track workspace growth")) {
          spill_active_front();
          break;
        }
      }
      Kokkos::deep_copy(execution, error, std::uint32_t{0});
      auto const bucketed = bucketKokkosWavefront(
          queue.current(), current_count, active_workspace.bucketing,
          execution, openmp_chunk_size);
      auto const current = bucketed.particles.rawDeviceView();
      if (bucketed.sorted) {
        ++result.wavefront_bucketing_batches;
        result.wavefront_bucketing_particles += current_count;
      } else {
        ++result.wavefront_bucketing_small_batches;
        result.wavefront_bucketing_small_particles += current_count;
      }

      enqueueResidentLeptonFront(
          physics_context, electron_mass_GeV, air_moliere_fast_path, current,
          active_workspace, current_count, capacity, photon_capacity,
          next_history_id, secondary_history_id_limit_exclusive,
          project_steps, capture_first_interaction, execution, openmp_chunk_size);

      // The scans and final-state writers above target disjoint, aligned
      // members of front_control_device through unmanaged Views.  Preserve
      // the original single D2H/fence host checkpoint without launching a
      // one-thread packing kernel for every resident wavefront.
      Kokkos::deep_copy(
          execution, host_front_control, front_control_device);
      waitResidentExecution(execution,
          "download resident Kokkos lepton front control", cooperative_wait);
      auto const& front_control = host_front_control();
      auto const& totals = front_control.totals.values;
      if (front_control.interaction_count > current_count)
        throw std::runtime_error(
            "resident Kokkos lepton interaction compaction overflow");
      auto const interaction_count = static_cast<std::size_t>(
          front_control.interaction_count);
      auto const selection_fallback_count =
          totals[LeptonSelectionFallbackOffset];
      auto const transport_fallback_count =
          totals[LeptonTransportFallbackOffset];
      auto const vertex_fallback_count =
          totals[LeptonVertexFallbackOffset];
      auto const final_state_fallback_count =
          totals[LeptonFinalStateFallbackOffset];
      auto const total_fallback_count =
          selection_fallback_count + transport_fallback_count +
          vertex_fallback_count + final_state_fallback_count;
      auto const vertex_interaction_count =
          front_control.vertex_interaction_count;
      if (selection_fallback_count > current_count ||
          total_fallback_count > current_count ||
          vertex_fallback_count > interaction_count ||
          vertex_interaction_count >
              interaction_count - vertex_fallback_count ||
          final_state_fallback_count > vertex_interaction_count)
        throw std::runtime_error(
            "resident Kokkos lepton stage accounting overflow");
      auto const vertex_continuation_count =
          interaction_count - vertex_fallback_count -
          vertex_interaction_count;
      if (totals[LeptonNextOffset] > capacity ||
          totals[LeptonPhotonOffset] > photon_capacity) {
        result.workspace_limit_checkpoint = true;
        break;
      }
      if (totals[LeptonChildOffset] >
          secondary_history_id_limit_exclusive - next_history_id) {
        result.history_range_exhausted = true;
        break;
      }
      if (front_control.materialization_error != 0)
        throw std::runtime_error(
            "resident Kokkos lepton final-state materialization failed");

      if (capture_diagnostic_interactions && interaction_count != 0) {
        auto const active_vertices = Kokkos::subview(
            vertices,
            std::make_pair<std::size_t>(0, interaction_count));
        auto host_vertices = Kokkos::create_mirror_view(active_vertices);
        Kokkos::deep_copy(execution, host_vertices, active_vertices);
        waitResidentExecution(execution,
            "download resident Kokkos lepton interaction diagnostics", cooperative_wait);
        for (std::size_t candidate = 0; candidate < interaction_count;
             ++candidate) {
          auto const& vertex = host_vertices(candidate);
          if (vertex.interaction_flag != 0)
            result.interaction_records.push_back(vertex.record);
        }
      }

      result.selected_interactions +=
          current_count - static_cast<std::size_t>(selection_fallback_count);
      result.interaction_bearing_wavefronts += interaction_count != 0;
      result.vertex_interactions_selected += vertex_interaction_count;
      result.vertex_no_interaction_continuations +=
          vertex_continuation_count;
      result.final_state_bearing_wavefronts +=
          vertex_interaction_count != 0;

      auto const resident_step_count =
          static_cast<std::size_t>(totals[LeptonStepOffset]);
      auto const resident_record_count =
          static_cast<std::size_t>(totals[LeptonRecordOffset]);
      if (resident_step_count > current_count)
        throw std::runtime_error(
            "resident Kokkos lepton packed step count exceeds source count");
      if (resident_record_count > interaction_count)
        throw std::runtime_error(
            "resident Kokkos lepton packed record count exceeds interaction "
            "count");

      enqueueResidentLeptonAccumulation(
          active_workspace, current_count, front_control, profile_projection,
          profile_accumulator, radio_accumulator, execution, openmp_chunk_size);
      auto const projected_count = static_cast<std::size_t>(
          project_steps ? totals[LeptonStepOffset] : std::uint64_t{0});
      auto const step_count = static_cast<std::size_t>(
          !project_steps && profile_accumulator == nullptr
              ? totals[LeptonStepOffset]
              : std::uint64_t{0});
      // Keep the same host/device ownership rule as native CUDA.  A
      // device-resident profile accumulator consumes both transport and
      // final-state records; neither half of that ledger may subsequently be
      // replayed by the host router.
      auto const record_count = static_cast<std::size_t>(
          profile_accumulator == nullptr ? totals[LeptonRecordOffset]
                                         : std::uint64_t{0});
      bool photons_spilled_to_cpu = false;
      if (pending_photons != nullptr) {
        if (!pending_photons->tryAppend(
                photons, totals[LeptonPhotonOffset],
                maximum_pending_particles, execution, memory_gate,
                "Kokkos pending photon queue growth"))
          photons_spilled_to_cpu = true;
      }
      auto const photon_count = static_cast<std::size_t>(
          pending_photons == nullptr || photons_spilled_to_cpu
              ? totals[LeptonPhotonOffset]
              : std::uint64_t{0});
      auto const fallback_count =
          static_cast<std::size_t>(total_fallback_count);
      auto const observation_count =
          static_cast<std::size_t>(totals[LeptonObservationOffset]);
      auto const decay_count =
          static_cast<std::size_t>(totals[LeptonDecayOffset]);

      active_workspace.ensureHostOutputCapacity(
          projected_count, step_count, record_count, photon_count,
          fallback_count, observation_count, decay_count, capacity,
          photon_capacity);
      auto const active_projected =
          std::make_pair<std::size_t>(0, projected_count);
      auto const active_steps =
          std::make_pair<std::size_t>(0, step_count);
      auto const active_records =
          std::make_pair<std::size_t>(0, record_count);
      auto const active_photons =
          std::make_pair<std::size_t>(0, photon_count);
      auto const active_fallbacks =
          std::make_pair<std::size_t>(0, fallback_count);
      auto const active_observations =
          std::make_pair<std::size_t>(0, observation_count);
      auto const active_decays =
          std::make_pair<std::size_t>(0, decay_count);
      auto const projected_source =
          Kokkos::subview(projected_steps, active_projected);
      auto const step_source = Kokkos::subview(steps, active_steps);
      auto const record_source = Kokkos::subview(records, active_records);
      auto const photon_source = Kokkos::subview(photons, active_photons);
      auto const fallback_source = Kokkos::subview(fallbacks, active_fallbacks);
      auto const observation_source =
          Kokkos::subview(observations, active_observations);
      auto const decay_source = Kokkos::subview(decays, active_decays);
      auto const host_projected_steps = Kokkos::subview(
          active_workspace.host_projected_steps, active_projected);
      auto const host_steps =
          Kokkos::subview(active_workspace.host_steps, active_steps);
      auto const host_records =
          Kokkos::subview(active_workspace.host_records, active_records);
      auto const host_photons =
          Kokkos::subview(active_workspace.host_photons, active_photons);
      auto const host_fallbacks =
          Kokkos::subview(active_workspace.host_fallbacks, active_fallbacks);
      auto const host_observations = Kokkos::subview(
          active_workspace.host_observations, active_observations);
      auto const host_decays =
          Kokkos::subview(active_workspace.host_decays, active_decays);

      if (projected_count != 0)
        Kokkos::deep_copy(execution, host_projected_steps, projected_source);
      if (step_count != 0)
        Kokkos::deep_copy(execution, host_steps, step_source);
      if (record_count != 0)
        Kokkos::deep_copy(execution, host_records, record_source);
      if (photon_count != 0)
        Kokkos::deep_copy(execution, host_photons, photon_source);
      if (fallback_count != 0)
        Kokkos::deep_copy(execution, host_fallbacks, fallback_source);
      if (observation_count != 0)
        Kokkos::deep_copy(execution, host_observations, observation_source);
      if (decay_count != 0)
        Kokkos::deep_copy(execution, host_decays, decay_source);

      auto const has_host_outputs =
          projected_count != 0 || step_count != 0 || record_count != 0 ||
          photon_count != 0 || fallback_count != 0 ||
          observation_count != 0 || decay_count != 0;
      if (has_host_outputs)
        waitResidentExecution(execution,
            "download resident Kokkos lepton host outputs", cooperative_wait);

      appendResidentLeptonHostCopy(
          result.projected_step_records, host_projected_steps,
          projected_count);
      appendResidentLeptonHostCopy(
          result.step_records, host_steps, step_count);
      appendResidentLeptonHostCopy(
          result.final_state_records, host_records, record_count);
      if (photons_spilled_to_cpu)
        appendResidentLeptonHostCopy(
            result.cpu_spill_particles, host_photons, photon_count);
      else if (pending_photons == nullptr)
        appendResidentLeptonHostCopy(
            result.generated_photons, host_photons, photon_count);
      appendResidentLeptonHostCopy(
          result.fallback_events, host_fallbacks, fallback_count);
      appendResidentLeptonHostCopy(
          result.observations, host_observations, observation_count);
      appendResidentLeptonHostCopy(
          result.decay_candidates, host_decays, decay_count);
      result.transport_records += totals[LeptonStepOffset];
      next_history_id += totals[LeptonChildOffset];
      queue.commitNextAlreadySynchronized(totals[LeptonNextOffset]);
      result.wavefronts++;
      result.peak_resident_leptons =
          std::max(result.peak_resident_leptons, queue.size());
    }

    auto const host_call_statistics = active_workspace.host_call_statistics;
    auto const host_first_interaction_candidates =
        active_workspace.host_first_interaction_candidates;
    auto const host_first_snapshots = active_workspace.host_first_snapshots;
    Kokkos::deep_copy(execution, host_call_statistics, call_statistics);
    if (capture_first_interaction) {
      Kokkos::deep_copy(
          execution, host_first_interaction_candidates,
          first_interaction_candidates);
      // Copy the single diagnostic slot together with the candidate count.
      // Its bytes are inspected only when the count proves it was written.
      Kokkos::deep_copy(
          execution, host_first_snapshots, first_snapshots);
    }
    waitResidentExecution(execution,
        "download resident Kokkos lepton call diagnostics", cooperative_wait);
    queue.markExecutionSynchronized();
    queue_execution_guard.release();
    auto const transport_statistic = [&](std::size_t const index) {
      return host_call_statistics(LeptonCallTransportStatisticBase + index);
    };
    auto const interaction_statistic = [&](std::size_t const index) {
      return host_call_statistics(LeptonCallInteractionStatisticBase + index);
    };

    result.interaction_vertices +=
        transport_statistic(LeptonTransportInteractionVertices);
    result.lpm_suppressions +=
        interaction_statistic(LeptonInteractionLpmSuppressions);
    auto& process_statistics = result.process_statistics;
    process_statistics.gpu_final_states +=
        interaction_statistic(LeptonInteractionGpuFinalStates);
    process_statistics.physical_secondaries_generated +=
        interaction_statistic(LeptonInteractionPhysicalSecondaries);
    process_statistics.brems_final_states +=
        interaction_statistic(LeptonInteractionBremsFinalStates);
    process_statistics.annihilation_final_states +=
        interaction_statistic(LeptonInteractionAnnihilationFinalStates);
    process_statistics.ionization_final_states +=
        interaction_statistic(LeptonInteractionIonizationFinalStates);
    process_statistics.electron_pair_final_states +=
        interaction_statistic(LeptonInteractionElectronPairFinalStates);
    process_statistics.brems_lpm_trials +=
        interaction_statistic(LeptonInteractionBremsLpmTrials);
    process_statistics.brems_lpm_suppressions +=
        interaction_statistic(LeptonInteractionBremsLpmSuppressions);
    process_statistics.electron_pair_lpm_trials +=
        interaction_statistic(LeptonInteractionElectronPairLpmTrials);
    process_statistics.electron_pair_lpm_suppressions +=
        interaction_statistic(LeptonInteractionElectronPairLpmSuppressions);
    process_statistics.electron_pair_rejection_trials +=
        interaction_statistic(LeptonInteractionElectronPairRejectionTrials);
    process_statistics.electron_pair_zero_weight_samples +=
        interaction_statistic(
            LeptonInteractionElectronPairZeroWeightSamples);
    process_statistics.electron_pair_rejection_fallbacks +=
        interaction_statistic(
            LeptonInteractionElectronPairRejectionFallbacks);
    process_statistics.electron_pair_envelope_violations +=
        interaction_statistic(
            LeptonInteractionElectronPairEnvelopeViolations);
    process_statistics.thinning_hillas_vertices +=
        interaction_statistic(LeptonInteractionThinningHillasVertices);
    process_statistics.thinning_statistical_vertices +=
        interaction_statistic(LeptonInteractionThinningStatisticalVertices);
    process_statistics.thinning_particles_discarded +=
        interaction_statistic(LeptonInteractionThinningParticlesDiscarded);
    process_statistics.moliere_trials +=
        transport_statistic(LeptonTransportMoliereTrials);
    process_statistics.moliere_deflections +=
        transport_statistic(LeptonTransportMoliereDeflections);
    process_statistics.moliere_zero_deflections +=
        transport_statistic(LeptonTransportMoliereZeroDeflections);
    process_statistics.moliere_newton_iterations +=
        transport_statistic(LeptonTransportMoliereNewtonIterations);
    process_statistics.moliere_max_newton_iterations = std::max(
        process_statistics.moliere_max_newton_iterations,
        static_cast<std::uint32_t>(transport_statistic(
            LeptonTransportMoliereMaximumNewtonIterations)));
    for (std::size_t limit = 0;
         limit < process_statistics.lepton_transport_limits.size(); ++limit)
      process_statistics.lepton_transport_limits[limit] +=
          transport_statistic(LeptonTransportLimitBase + limit);

    result.secondary_history_ids_used =
        next_history_id - first_secondary_history_id;
    result.completed = queue.empty();
    if (!result.completed)
      result.remaining_leptons = queue.download(execution);
    if (capture_first_interaction) {
      if (host_first_interaction_candidates(0) > 1) {
        throw std::runtime_error(
            "resident Kokkos lepton transport produced more than one "
            "generation-zero interaction candidate");
      }
      if (host_first_interaction_candidates(0) == 1)
        first_interaction = host_first_snapshots(0);
    }
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
