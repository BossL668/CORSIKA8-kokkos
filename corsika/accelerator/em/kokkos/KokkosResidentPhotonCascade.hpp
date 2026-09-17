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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <corsika/accelerator/em/detail/InteractionSelection.hpp>
#include <corsika/accelerator/em/detail/ProfileProjectionStep.hpp>
#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosRangePolicy.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontBucketing.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  enum ResidentPhotonFallbackStage : std::uint32_t {
    PhotonNoFallback = 0,
    PhotonSelectionFallback = 1,
    PhotonTransportFallback = 2,
    PhotonFinalStateFallback = 3,
  };

  struct ResidentPhotonSourceCounts {
    std::uint32_t child{};
    std::uint32_t next{};
    std::uint32_t charged{};
    std::uint32_t fallback_stage{PhotonNoFallback};
    std::uint32_t observation{};
    std::uint32_t step{};
    std::uint32_t record{};
  };

  enum ResidentPhotonOffset : std::size_t {
    PhotonChildOffset = 0,
    PhotonNextOffset = 1,
    PhotonChargedOffset = 2,
    PhotonSelectionFallbackOffset = 3,
    PhotonTransportFallbackOffset = 4,
    PhotonFinalStateFallbackOffset = 5,
    PhotonObservationOffset = 6,
    PhotonStepOffset = 7,
    PhotonRecordOffset = 8,
    PhotonOffsetCount = 9,
  };

  struct ResidentPhotonScanValue {
    std::uint64_t values[PhotonOffsetCount]{};
  };

  KOKKOS_INLINE_FUNCTION std::uint64_t photonFallbackOutputOffset(
      std::uint32_t const stage, std::uint64_t const selection_offset,
      std::uint64_t const transport_offset,
      std::uint64_t const final_state_offset,
      ResidentPhotonScanValue const& totals) {
    if (stage == PhotonSelectionFallback) return selection_offset;
    if (stage == PhotonTransportFallback)
      return totals.values[PhotonSelectionFallbackOffset] +
             transport_offset;
    if (stage == PhotonFinalStateFallback)
      return totals.values[PhotonSelectionFallbackOffset] +
             totals.values[PhotonTransportFallbackOffset] +
             final_state_offset;
    return ~std::uint64_t{0};
  }

  template <class ExecutionSpace>
  struct ResidentPhotonScanFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentPhotonScanValue;

    Kokkos::View<ResidentPhotonSourceCounts const*, Memory> counts;
    Kokkos::View<std::uint64_t*[PhotonOffsetCount], Memory> offsets;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const { value = {}; }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      for (std::size_t column = 0; column < PhotonOffsetCount; ++column)
        destination.values[column] += source.values[column];
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& update,
                                           bool const final) const {
      if (final) {
        for (std::size_t column = 0; column < PhotonOffsetCount; ++column)
          offsets(source, column) = update.values[column];
      }
      auto const& source_counts = counts(source);
      update.values[PhotonChildOffset] += source_counts.child;
      update.values[PhotonNextOffset] += source_counts.next;
      update.values[PhotonChargedOffset] += source_counts.charged;
      update.values[PhotonSelectionFallbackOffset] +=
          source_counts.fallback_stage == PhotonSelectionFallback;
      update.values[PhotonTransportFallbackOffset] +=
          source_counts.fallback_stage == PhotonTransportFallback;
      update.values[PhotonFinalStateFallbackOffset] +=
          source_counts.fallback_stage == PhotonFinalStateFallback;
      update.values[PhotonObservationOffset] += source_counts.observation;
      update.values[PhotonStepOffset] += source_counts.step;
      update.values[PhotonRecordOffset] += source_counts.record;
    }
  };

  /**
   * The only wavefront state that the host must observe before committing the
   * next queue.  Both scan totals remain device resident until interaction
   * materialization has completed, so one small copy replaces three implicit
   * scan/error synchronizations.
   */
  struct ResidentPhotonFrontControl {
    ResidentPhotonScanValue totals{};
    std::uint64_t interaction_count{};
    std::uint32_t materialization_error{};
  };

  static_assert(std::is_standard_layout_v<ResidentPhotonScanValue>);
  static_assert(std::is_trivially_copyable_v<ResidentPhotonScanValue>);
  static_assert(
      sizeof(ResidentPhotonScanValue) ==
      PhotonOffsetCount * sizeof(std::uint64_t));
  static_assert(std::is_standard_layout_v<ResidentPhotonFrontControl>);
  static_assert(std::is_trivially_copyable_v<ResidentPhotonFrontControl>);
  static_assert(offsetof(ResidentPhotonFrontControl, totals) == 0);
  static_assert(
      offsetof(ResidentPhotonFrontControl, interaction_count) ==
      sizeof(ResidentPhotonScanValue));
  static_assert(
      offsetof(ResidentPhotonFrontControl, materialization_error) ==
      sizeof(ResidentPhotonScanValue) + sizeof(std::uint64_t));

  enum ResidentPhotonTransportStatistic : std::size_t {
    PhotonTransportInteractionVertices = 0,
    PhotonTransportLayerBoundaries,
    PhotonTransportParticleCuts,
    PhotonSelectionNativeNewtonIterations,
    PhotonSelectionNativeBisectionIterations,
    PhotonSelectionNativeInverseFailures,
    PhotonTransportStatisticCount,
  };

  inline constexpr std::size_t PhotonCallTransportStatisticOffset = 0;

  struct ResidentPhotonTransportStatisticsValue {
    std::uint64_t values[PhotonTransportStatisticCount]{};
  };

  /**
   * Selection statistics visit the source wavefront while transport
   * statistics visit the independently packed step ledger.
   *
   * Both domains are stable and have at most current_count entries.  Using
   * the packed ledger avoids retaining a second source-indexed transport
   * state solely for statistics and matches the native-CUDA accounting
   * contract.
   */
  template <class ExecutionSpace>
  struct ResidentPhotonTransportStatisticsFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentPhotonTransportStatisticsValue;

    Kokkos::View<detail::InteractionSelectionOutcome const*, Memory>
        selections;
    Kokkos::View<gpu::em::PhotonTransportRecord const*, Memory> packed_steps;
    std::size_t step_count{};
    Kokkos::View<std::uint64_t*, Memory> call_statistics;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
      for (std::size_t index = 0; index < PhotonTransportStatisticCount;
           ++index)
        value.values[index] = 0;
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      for (std::size_t index = 0; index < PhotonTransportStatisticCount;
           ++index)
        destination.values[index] += source.values[index];
    }

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const {
      for (std::size_t index = 0; index < PhotonTransportStatisticCount;
           ++index)
        call_statistics(PhotonCallTransportStatisticOffset + index) +=
            value.values[index];
    }

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& value) const {
      using namespace gpu::em;
      auto const& selection = selections(source);
      value.values[PhotonSelectionNativeNewtonIterations] +=
          selection.native_newton_iterations;
      value.values[PhotonSelectionNativeBisectionIterations] +=
          selection.native_bisection_iterations;
      value.values[PhotonSelectionNativeInverseFailures] +=
          selection.native_inverse_failures;
      if (source >= step_count) return;
      auto const limit = packed_steps(source).limit;
      value.values[PhotonTransportInteractionVertices] +=
          limit == PhotonTransportLimit::Interaction;
      value.values[PhotonTransportLayerBoundaries] +=
          limit == PhotonTransportLimit::LayerBoundary;
      value.values[PhotonTransportParticleCuts] +=
          limit == PhotonTransportLimit::ParticleCut;
    }
  };

  /**
   * Fuse source-wide selection statistics and packed transport statistics
   * with the profile pass over that same packed step ledger.
   */
  template <class ExecutionSpace>
  struct ResidentPhotonTransportStatisticsProfileFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentPhotonTransportStatisticsValue;

    ResidentPhotonTransportStatisticsFunctor<ExecutionSpace> statistics;
    gpu::em::detail::DeviceProfileProjection projection;
    gpu::em::detail::DeviceProfileAccumulator accumulator;

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

    KOKKOS_INLINE_FUNCTION void operator()(std::size_t const source,
                                           value_type& value) const {
      statistics(source, value);
      if (source < statistics.step_count)
        detail::accumulatePhotonProfileStep<KokkosProfileAtomicOperations>(
            projection, accumulator, statistics.packed_steps(source));
    }
  };

  enum ResidentPhotonInteractionStatistic : std::size_t {
    PhotonInteractionLpmSuppressions = 0,
    PhotonInteractionGpuFinalStates,
    PhotonInteractionPhysicalSecondaries,
    PhotonInteractionPairFinalStates,
    PhotonInteractionComptonFinalStates,
    PhotonInteractionPhotoelectricFinalStates,
    PhotonInteractionLpmTrials,
    PhotonInteractionThinningHillasVertices,
    PhotonInteractionThinningStatisticalVertices,
    PhotonInteractionThinningParticlesDiscarded,
    PhotonInteractionStatisticCount,
  };

  inline constexpr std::size_t PhotonCallInteractionStatisticOffset =
      PhotonTransportStatisticCount;
  inline constexpr std::size_t PhotonCallStatisticCount =
      PhotonTransportStatisticCount + PhotonInteractionStatisticCount;

  struct ResidentPhotonInteractionStatisticsValue {
    std::uint64_t values[PhotonInteractionStatisticCount]{};
  };

  /** Statistics evaluated only for the stable compacted interaction list. */
  template <class ExecutionSpace>
  struct ResidentPhotonInteractionStatisticsFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentPhotonInteractionStatisticsValue;

    Kokkos::View<detail::PhotonFinalStateClassification const*, Memory>
        final_states;
    Kokkos::View<std::uint64_t*, Memory> call_statistics;

    KOKKOS_INLINE_FUNCTION void init(value_type& value) const {
      for (std::size_t index = 0; index < PhotonInteractionStatisticCount;
           ++index)
        value.values[index] = 0;
    }

    KOKKOS_INLINE_FUNCTION void join(value_type& destination,
                                     value_type const& source) const {
      for (std::size_t index = 0; index < PhotonInteractionStatisticCount;
           ++index)
        destination.values[index] += source.values[index];
    }

    KOKKOS_INLINE_FUNCTION void final(value_type& value) const {
      for (std::size_t index = 0; index < PhotonInteractionStatisticCount;
           ++index)
        call_statistics(PhotonCallInteractionStatisticOffset + index) +=
            value.values[index];
    }

    KOKKOS_INLINE_FUNCTION void operator()(
        std::size_t const interaction_index, value_type& value) const {
      using namespace gpu::em;
      auto const& final = final_states(interaction_index);
      value.values[PhotonInteractionLpmSuppressions] +=
          final.suppression_flag;
      value.values[PhotonInteractionGpuFinalStates] += final.record_flag;
      if (final.record_flag != 0) {
        value.values[PhotonInteractionPhysicalSecondaries] +=
            final.child_count;
        value.values[PhotonInteractionPairFinalStates] +=
            final.photon_pair_flag;
        value.values[PhotonInteractionComptonFinalStates] +=
            final.compton_flag;
        value.values[PhotonInteractionPhotoelectricFinalStates] +=
            final.photoelectric_flag;
        auto const status =
            static_cast<EmThinningStatus>(final.parameters.thinning_status);
        value.values[PhotonInteractionThinningHillasVertices] +=
            status == EmThinningStatus::Hillas;
        value.values[PhotonInteractionThinningStatisticalVertices] +=
            status == EmThinningStatus::Statistical;
        auto const original_multiplicity =
            final.parameters.process_id == PhotoelectricProcessId ? 1U : 2U;
        if (final.child_count <= original_multiplicity)
          value.values[PhotonInteractionThinningParticlesDiscarded] +=
              original_multiplicity - final.child_count;
      }
      value.values[PhotonInteractionLpmTrials] +=
          final.photon_pair_flag + final.suppression_flag;
    }
  };

  /** Fuse interaction statistics with the stable packed final-state ledger. */
  template <class ExecutionSpace>
  struct ResidentPhotonInteractionStatisticsProfileFunctor {
    using Memory = typename ExecutionSpace::memory_space;
    using value_type = ResidentPhotonInteractionStatisticsValue;

    ResidentPhotonInteractionStatisticsFunctor<ExecutionSpace> statistics;
    gpu::em::detail::DeviceProfileProjection projection;
    gpu::em::detail::DeviceProfileAccumulator accumulator;
    Kokkos::View<gpu::em::PhotonTransportRecord*, Memory> steps;
    std::size_t step_count{};
    Kokkos::View<gpu::em::PhotonPairFinalStateRecord*, Memory> records;
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
        detail::accumulatePhotonProfileFinalState<
            KokkosProfileAtomicOperations>(
            projection, accumulator, steps.data(), step_count,
            records(interaction_index));
    }
  };

  /** Stable exclusive scan used to compact one classification stage. */
  template <class ExecutionSpace>
  struct ResidentPhotonCompactionScanFunctor {
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

  /** Direct writer for one source-ordered resident photon final state. */
  struct ResidentPhotonFinalStateWriter {
    ParticleSoARawView next{};
    RawView1D<gpu::em::PhotonPairFinalStateRecord> records{};
    RawView1D<gpu::em::EmParticleState> charged{};
    RawView1D<gpu::em::ProposalFallbackEvent> fallbacks{};
    RawView1D<gpu::em::GpuFirstInteractionSnapshot> first_snapshots{};
    std::uint32_t* first_interaction_candidates{};
    std::uint32_t* error{};
    std::uint64_t next_offset{};
    std::uint64_t charged_offset{};
    std::uint64_t fallback_offset{};
    std::uint64_t record_offset{};
    std::uint32_t capture_first_interaction{};
    std::uint32_t next_written{};
    std::uint32_t charged_written{};

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
        detail::PhotonFinalStateParameters const&) {
      next.store(next_offset, parent);
    }

    KOKKOS_INLINE_FUNCTION void firstInteraction(
        gpu::em::EmParticleState const& parent, std::int32_t process_id,
        gpu::em::EmParticleState const& first,
        gpu::em::EmParticleState const* second) {
      if (capture_first_interaction == 0 || parent.generation != 0) return;
      auto const candidate = Kokkos::atomic_fetch_add(
          first_interaction_candidates, std::uint32_t{1});
      if (candidate != 0) return;
      auto& snapshot = first_snapshots(0);
      snapshot.parent_at_vertex = parent;
      snapshot.process_id = process_id;
      snapshot.secondary_count = second == nullptr ? 1U : 2U;
      snapshot.secondaries[0] = first;
      snapshot.secondaries[0].weight = parent.weight;
      if (second != nullptr) {
        snapshot.secondaries[1] = *second;
        snapshot.secondaries[1].weight = parent.weight;
      }
    }

    KOKKOS_INLINE_FUNCTION void secondary(
        std::uint32_t, gpu::em::EmParticleState const& particle) {
      if (particle.pid ==
          static_cast<std::int32_t>(gpu::em::EmPid::Photon)) {
        next.store(next_offset + next_written++, particle);
      } else {
        charged(charged_offset + charged_written++) = particle;
      }
    }

    KOKKOS_INLINE_FUNCTION gpu::em::PhotonPairFinalStateRecord& record() {
      return records(record_offset);
    }

    KOKKOS_INLINE_FUNCTION void finishRecord(std::uint32_t) {}

    KOKKOS_INLINE_FUNCTION void fail(std::uint32_t value) {
      Kokkos::atomic_compare_exchange(error, std::uint32_t{0}, value);
    }
  };

  /** Device workspace retained by KokkosEmBackend across host wavefronts. */
  template <class ExecutionSpace>
  class KokkosResidentPhotonWorkspace {
  public:
    using Memory = typename ExecutionSpace::memory_space;
    using HostMemory = HostStagingSpace<ExecutionSpace>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using InteractionCountView =
        Kokkos::View<std::uint64_t, Memory, Unmanaged>;
    using ScanTotalsView =
        Kokkos::View<ResidentPhotonScanValue, Memory, Unmanaged>;
    using ErrorView = Kokkos::View<std::uint32_t*, Memory, Unmanaged>;

    using HostProjectedStepView =
        Kokkos::View<gpu::em::ProjectedEmStepRecord*, HostMemory>;
    using HostStepView =
        Kokkos::View<gpu::em::PhotonTransportRecord*, HostMemory>;
    using HostRecordView =
        Kokkos::View<gpu::em::PhotonPairFinalStateRecord*, HostMemory>;
    using HostParticleView =
        Kokkos::View<gpu::em::EmParticleState*, HostMemory>;
    using HostFallbackView =
        Kokkos::View<gpu::em::ProposalFallbackEvent*, HostMemory>;
    using HostObservationView =
        Kokkos::View<gpu::em::ObservationRecord*, HostMemory>;

    void ensureCapacity(std::size_t const requested,
                        ExecutionSpace const& execution) {
      if (requested == 0)
        throw std::invalid_argument(
            "resident Kokkos photon workspace capacity must be positive");
      if (call_statistics.extent(0) == 0)
        call_statistics = decltype(call_statistics)(
            Kokkos::view_alloc(
                execution, Kokkos::WithoutInitializing,
                "c8_kokkos_resident_photon_call_statistics"),
            PhotonCallStatisticCount);
      if (host_call_statistics.extent(0) == 0)
        host_call_statistics = decltype(host_call_statistics)(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "c8_kokkos_resident_photon_host_call_statistics"),
            PhotonCallStatisticCount);
      if (front_control.data() == nullptr) {
        front_control = decltype(front_control)(
            Kokkos::view_alloc(
                execution, Kokkos::WithoutInitializing,
                "c8_kokkos_resident_photon_front_control"));
        host_front_control = decltype(host_front_control)(
            "c8_kokkos_resident_photon_host_front_control");
        auto* const control_bytes = reinterpret_cast<unsigned char*>(
            front_control.data());
        scan_totals = ScanTotalsView(
            reinterpret_cast<ResidentPhotonScanValue*>(
                control_bytes +
                offsetof(ResidentPhotonFrontControl, totals)));
        interaction_count = InteractionCountView(
            reinterpret_cast<std::uint64_t*>(
                control_bytes + offsetof(
                                    ResidentPhotonFrontControl,
                                    interaction_count)));
        error = ErrorView(
            reinterpret_cast<std::uint32_t*>(
                control_bytes + offsetof(
                                    ResidentPhotonFrontControl,
                                    materialization_error)),
            1);
        host_first_interaction_candidates =
            decltype(host_first_interaction_candidates)(
                Kokkos::view_alloc(
                    Kokkos::WithoutInitializing,
                    "c8_kokkos_resident_photon_host_first_candidates"),
                1);
        host_first_snapshots = decltype(host_first_snapshots)(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "c8_kokkos_resident_photon_host_first_snapshot"),
            1);
      }
      if (requested <= capacity_) return;
      auto const grown = capacity_ > std::numeric_limits<std::size_t>::max() / 2
                             ? requested
                             : std::max(requested, capacity_ * 2);
      if (grown > std::numeric_limits<std::size_t>::max() / 2)
        throw std::length_error(
            "resident Kokkos photon charged workspace exceeds addressable capacity");
      auto const charged_capacity = grown * 2;
      queue.ensureCapacity(grown, execution);
      selections = decltype(selections)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_selections"),
          grown);
      transports = decltype(transports)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_transports"),
          grown);
      transport_fallbacks = decltype(transport_fallbacks)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_transport_fallbacks"),
          grown);
      counts = decltype(counts)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_counts"),
          grown);
      interaction_flags = decltype(interaction_flags)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_interaction_flags"),
          grown);
      interaction_sources = decltype(interaction_sources)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_interaction_sources"),
          grown);
      offsets = decltype(offsets)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_offsets"),
          grown);
      steps = decltype(steps)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_steps"),
          grown);
      projected_steps = decltype(projected_steps)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_projected_steps"),
          grown);
      records = decltype(records)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_records"),
          grown);
      charged = decltype(charged)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_charged"),
          charged_capacity);
      fallbacks = decltype(fallbacks)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_fallbacks"),
          grown);
      observations = decltype(observations)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_observations"),
          grown);
      // A shower has exactly one generation-zero projectile.  Match the
      // native-CUDA capture contract and retain only its first interaction,
      // instead of allocating and downloading one large snapshot per active
      // photon on every wavefront.
      if (first_snapshots.extent(0) == 0) {
        first_snapshots = decltype(first_snapshots)(
            Kokkos::view_alloc(
                execution, Kokkos::WithoutInitializing,
                "c8_kokkos_resident_photon_first_snapshot"),
            1);
        first_interaction_candidates =
            decltype(first_interaction_candidates)(
                Kokkos::view_alloc(
                    execution, Kokkos::WithoutInitializing,
                    "c8_kokkos_resident_photon_first_candidates"),
                1);
      }
      // Commit the advertised capacity only after every allocation succeeds.
      // If an allocation throws, a retry must not observe a partially grown
      // workspace and return early with stale/null Views.
      capacity_ = grown;
    }

    KokkosMemoryProjection projectedCapacity(
        std::size_t const requested) const {
      if (requested == 0)
        throw std::invalid_argument(
            "resident Kokkos photon workspace capacity must be positive");
      auto const current = deviceBytes();
      auto const view_bytes = [](auto const& view) {
        return view.span() *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      auto const bytes = [](std::size_t const count,
                            std::size_t const element) {
        return checkedMemoryMultiply(count, element);
      };
      KokkosMemoryProjectionBuilder projection(current);
      if (call_statistics.extent(0) == 0)
        projection.replace(
            0, bytes(PhotonCallStatisticCount, sizeof(std::uint64_t)));
      if (front_control.data() == nullptr)
        projection.replace(0, sizeof(ResidentPhotonFrontControl));
      if (requested <= capacity_) return projection.result();

      auto const grown =
          capacity_ > std::numeric_limits<std::size_t>::max() / 2
              ? requested
              : std::max(requested, capacity_ * 2);
      if (grown > std::numeric_limits<std::size_t>::max() / 2)
        throw std::length_error(
            "resident Kokkos photon charged workspace exceeds addressable capacity");
      projection.merge(queue.deviceBytes(), queue.projectedCapacity(grown));
      projection.replace(
          view_bytes(selections),
          bytes(grown, sizeof(detail::InteractionSelectionOutcome)));
      projection.replace(
          view_bytes(transports),
          bytes(grown, sizeof(gpu::em::PhotonTransportRecord)));
      projection.replace(
          view_bytes(transport_fallbacks),
          bytes(grown, sizeof(gpu::em::ProposalFallbackEvent)));
      projection.replace(
          view_bytes(counts), bytes(grown, sizeof(ResidentPhotonSourceCounts)));
      projection.replace(
          view_bytes(interaction_flags), bytes(grown, sizeof(std::uint32_t)));
      projection.replace(
          view_bytes(interaction_sources), bytes(grown, sizeof(std::size_t)));
      projection.replace(
          view_bytes(offsets),
          bytes(bytes(grown, PhotonOffsetCount), sizeof(std::uint64_t)));
      projection.replace(
          view_bytes(steps),
          bytes(grown, sizeof(gpu::em::PhotonTransportRecord)));
      projection.replace(
          view_bytes(projected_steps),
          bytes(grown, sizeof(gpu::em::ProjectedEmStepRecord)));
      projection.replace(
          view_bytes(records),
          bytes(grown, sizeof(gpu::em::PhotonPairFinalStateRecord)));
      projection.replace(
          view_bytes(charged),
          bytes(bytes(grown, 2), sizeof(gpu::em::EmParticleState)));
      projection.replace(
          view_bytes(fallbacks),
          bytes(grown, sizeof(gpu::em::ProposalFallbackEvent)));
      projection.replace(
          view_bytes(observations),
          bytes(grown, sizeof(gpu::em::ObservationRecord)));
      if (first_snapshots.extent(0) == 0) {
        projection.replace(0, sizeof(gpu::em::GpuFirstInteractionSnapshot));
        projection.replace(0, sizeof(std::uint32_t));
      }
      return projection.result();
    }

    /**
     * Grow the register-heavy final-state arena to the active-front upper
     * bound.  Actual interaction work is still guarded by the compacted count
     * retained on the device.
     */
    void ensureCandidateCapacity(std::size_t const requested,
                                 ExecutionSpace const& execution) {
      if (requested <= candidate_capacity_) return;
      auto const grown =
          candidate_capacity_ > std::numeric_limits<std::size_t>::max() / 2
              ? requested
              : std::max(requested, candidate_capacity_ * 2);
      // Statistics/profile kernels from the preceding front may still read
      // the old arena when no host output was downloaded.  Growth is rare;
      // synchronize only before replacing the View.
      execution.fence("grow resident Kokkos photon candidate workspace");
      final_states = decltype(final_states)(
          Kokkos::view_alloc(
              execution, Kokkos::WithoutInitializing,
              "c8_kokkos_resident_photon_final_states"),
          grown);
      candidate_capacity_ = grown;
    }

    KokkosMemoryProjection projectedCandidateCapacity(
        std::size_t const requested) const {
      auto const current = deviceBytes();
      if (requested <= candidate_capacity_) return {current, current};
      auto const grown =
          candidate_capacity_ > std::numeric_limits<std::size_t>::max() / 2
              ? requested
              : std::max(requested, candidate_capacity_ * 2);
      auto const old_bytes = final_states.span() *
                             sizeof(typename decltype(final_states)::value_type);
      auto const new_bytes = checkedMemoryMultiply(
          grown, sizeof(detail::PhotonFinalStateClassification));
      KokkosMemoryProjectionBuilder projection(current);
      projection.replace(old_bytes, new_bytes);
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
        std::size_t const record_count, std::size_t const charged_count,
        std::size_t const fallback_count,
        std::size_t const observation_count) {
      growHostView(host_projected_steps, projected_count,
                   "c8_kokkos_resident_photon_host_projected_steps");
      growHostView(host_steps, step_count,
                   "c8_kokkos_resident_photon_host_steps");
      growHostView(host_records, record_count,
                   "c8_kokkos_resident_photon_host_records");
      growHostView(host_charged, charged_count,
                   "c8_kokkos_resident_photon_host_charged");
      growHostView(host_fallbacks, fallback_count,
                   "c8_kokkos_resident_photon_host_fallbacks");
      growHostView(host_observations, observation_count,
                   "c8_kokkos_resident_photon_host_observations");
    }

    std::size_t capacity() const noexcept { return capacity_; }

    std::size_t deviceBytes() const noexcept {
      auto const view_bytes = [](auto const& view) {
        // A default-constructed rank-zero Kokkos::View can report span()==1
        // even though data()==nullptr and no allocation exists.
        return (view.data() == nullptr ? 0 : view.span()) *
               sizeof(typename std::decay_t<decltype(view)>::value_type);
      };
      // error, interaction_count and scan_totals are unmanaged aliases into
      // front_control and therefore must not be counted a second time.
      return queue.deviceBytes() + bucketing.deviceBytes() +
             view_bytes(selections) + view_bytes(transports) +
             view_bytes(transport_fallbacks) + view_bytes(final_states) +
             view_bytes(counts) + view_bytes(interaction_flags) +
             view_bytes(interaction_sources) + view_bytes(offsets) +
             view_bytes(steps) + view_bytes(projected_steps) +
             view_bytes(records) + view_bytes(charged) +
             view_bytes(fallbacks) + view_bytes(observations) +
             view_bytes(first_snapshots) +
             view_bytes(first_interaction_candidates) +
             view_bytes(call_statistics) + view_bytes(front_control);
    }

    KokkosWavefrontQueue<ExecutionSpace> queue{};
    KokkosWavefrontBucketingWorkspace<ExecutionSpace> bucketing{};
    Kokkos::View<detail::InteractionSelectionOutcome*, Memory> selections{};
    Kokkos::View<gpu::em::PhotonTransportRecord*, Memory> transports{};
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory>
        transport_fallbacks{};
    Kokkos::View<detail::PhotonFinalStateClassification*, Memory>
        final_states{};
    Kokkos::View<ResidentPhotonSourceCounts*, Memory> counts{};
    Kokkos::View<std::uint32_t*, Memory> interaction_flags{};
    Kokkos::View<std::size_t*, Memory> interaction_sources{};
    Kokkos::View<std::uint64_t*[PhotonOffsetCount], Memory> offsets{};
    Kokkos::View<gpu::em::PhotonTransportRecord*, Memory> steps{};
    Kokkos::View<gpu::em::ProjectedEmStepRecord*, Memory> projected_steps{};
    Kokkos::View<gpu::em::PhotonPairFinalStateRecord*, Memory> records{};
    Kokkos::View<gpu::em::EmParticleState*, Memory> charged{};
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks{};
    Kokkos::View<gpu::em::ObservationRecord*, Memory> observations{};
    Kokkos::View<gpu::em::GpuFirstInteractionSnapshot*, Memory>
        first_snapshots{};
    Kokkos::View<std::uint32_t*, Memory> first_interaction_candidates{};
    Kokkos::View<std::uint64_t*, Memory> call_statistics{};
    Kokkos::View<ResidentPhotonFrontControl, Memory> front_control{};
    ErrorView error{};
    InteractionCountView interaction_count{};
    ScanTotalsView scan_totals{};
    Kokkos::View<ResidentPhotonFrontControl, HostStagingSpace<ExecutionSpace>>
        host_front_control{};
    HostProjectedStepView host_projected_steps{};
    HostStepView host_steps{};
    HostRecordView host_records{};
    HostParticleView host_charged{};
    HostFallbackView host_fallbacks{};
    HostObservationView host_observations{};
    Kokkos::View<std::uint64_t*, HostMemory> host_call_statistics{};
    Kokkos::View<std::uint32_t*, HostMemory>
        host_first_interaction_candidates{};
    Kokkos::View<gpu::em::GpuFirstInteractionSnapshot*, HostMemory>
        host_first_snapshots{};

  private:
    template <class View>
    static void growHostView(View& view, std::size_t const requested,
                             char const* const label) {
      if (requested <= view.extent(0)) return;
      auto const current = view.extent(0);
      auto const grown =
          current > std::numeric_limits<std::size_t>::max() / 2
              ? requested
              : std::max(requested, current * 2);
      View replacement(
          Kokkos::view_alloc(
              Kokkos::WithoutInitializing, std::string(label)),
          grown);
      view = std::move(replacement);
    }

    std::size_t capacity_{};
    std::size_t candidate_capacity_{};
  };

  /** Enqueue one resident photon front using the production physics kernels.
   * All storage must already be allocated. This function neither downloads
   * results nor fences the execution space. Queue ownership remains with the
   * caller until the materialization control checkpoint has completed.
   */
  template <class ExecutionSpace>
  void enqueueResidentPhotonFront(
      KokkosPhysicsContextView<ExecutionSpace> const physics_context,
      ParticleSoARawView const current, ParticleSoARawView const next,
      KokkosResidentPhotonWorkspace<ExecutionSpace>& workspace,
      std::size_t const current_count, std::size_t const capacity,
      std::uint64_t const next_history_id, bool const project_steps,
      bool const capture_first_interaction,
      ExecutionSpace const& execution, std::size_t const openmp_chunk_size) {
    using namespace gpu::em;
    auto const Policy = [openmp_chunk_size](ExecutionSpace const& ex,
                                            std::size_t, std::size_t end) {
      return makeKokkosRangePolicy(ex, end, openmp_chunk_size);
    };
    auto const& selections = workspace.selections;
    auto const& transports = workspace.transports;
    auto const& transport_fallbacks = workspace.transport_fallbacks;
    auto const& final_states = workspace.final_states;
    auto const& counts = workspace.counts;
    auto const& interaction_flags = workspace.interaction_flags;
    auto const& interaction_sources = workspace.interaction_sources;
    auto const& offsets = workspace.offsets;
    auto const& steps = workspace.steps;
    auto const& projected_steps = workspace.projected_steps;
    auto const& records = workspace.records;
    auto const& charged = workspace.charged;
    auto const& fallbacks = workspace.fallbacks;
    auto const& observations = workspace.observations;
    auto const& first_snapshots = workspace.first_snapshots;
    auto const& first_interaction_candidates = workspace.first_interaction_candidates;
    auto const& error = workspace.error;
    auto const interaction_count_device = workspace.interaction_count;
    auto const scan_totals_device = workspace.scan_totals;
    auto const charged_capacity = capacity * 2;
    auto const selections_raw = rawDeviceView(selections);
    auto const transports_raw = rawDeviceView(transports);
    auto const transport_fallbacks_raw =
        rawDeviceView(transport_fallbacks);
    auto const counts_raw = rawDeviceView(counts);
    auto const interaction_flags_raw = rawDeviceView(interaction_flags);
    auto const interaction_sources_raw = rawDeviceView(interaction_sources);
    auto const select_kernel =
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const& context = physics_context();
          ResidentPhotonSourceCounts source_counts{};
          auto const particle = current.load(source);
          auto& selection = selections_raw(source);
          selection = detail::selectDiscreteInteraction(
              context.physics, particle, source, context.random_seed,
              context.shower_id);
          if (selection.fallback_flag != 0) {
            source_counts.fallback_stage = PhotonSelectionFallback;
          }
          counts_raw(source) = source_counts;
        };
    static_assert(sizeof(select_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_photon_select",
        Policy(execution, 0, current_count),
        select_kernel);

    // Transport is a separate kernel so its geometry state does not share
    // registers with rate inversion.  Selection fallbacks return before
    // transport without introducing an additional scan of the full front.
    auto const transport_kernel =
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const& context = physics_context();
          auto const& selection = selections_raw(source);
          auto& source_counts = counts_raw(source);
          interaction_flags_raw(source) = 0;
          if (selection.fallback_flag != 0) return;
          auto const transport = detail::transportPhoton(
              context.environment, selection.interaction);
          transports_raw(source) = transport.record;
          transport_fallbacks_raw(source) = transport.fallback;
          if (transport.fallback_flag != 0) {
            source_counts.fallback_stage = PhotonTransportFallback;
            return;
          }
          source_counts.step = 1;
          auto const& step = transport.record;
          if (step.limit == PhotonTransportLimit::LayerBoundary) {
            source_counts.next = 1;
          } else if (
              step.limit == PhotonTransportLimit::ObservationSurface ||
              step.limit == PhotonTransportLimit::EscapedEnvironment ||
              (step.limit == PhotonTransportLimit::ParticleCut &&
               step.observation_surface_reached_before_cut != 0)) {
            source_counts.observation = 1;
          } else if (step.limit == PhotonTransportLimit::Interaction) {
            interaction_flags_raw(source) = 1;
          }
        };
    static_assert(sizeof(transport_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_photon_transport",
        Policy(execution, 0, current_count),
        transport_kernel);

    // Only true interaction vertices execute the comparatively heavy LPM,
    // final-state kinematics and thinning classifier.  A second stable scan
    // keeps their order identical to the original source wavefront.
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_interactions",
        Policy(execution, 0, current_count),
        ResidentPhotonCompactionScanFunctor<ExecutionSpace>{
            interaction_flags, interaction_sources},
        workspace.interaction_count);
    auto const final_states_raw = rawDeviceView(final_states);
    auto const interaction_count_raw = interaction_count_device.data();
    auto const final_state_kernel =
        KOKKOS_LAMBDA(std::size_t const interaction_index) {
          auto const& context = physics_context();
          auto const compacted_count =
              static_cast<std::size_t>(*interaction_count_raw);
          if (interaction_index >= compacted_count) return;
          auto const source = interaction_sources_raw(interaction_index);
          auto& source_counts = counts_raw(source);
          auto const& step = transports_raw(source);
          auto& final = final_states_raw(interaction_index);
          final = detail::classifyPhotonFinalState(
              context.physics, context.photon_pair_lpm, context.thinning,
              step.interaction, context.random_seed, context.shower_id);
          source_counts.child = final.child_count;
          source_counts.record = final.record_flag;
          if (final.fallback_flag != 0)
            source_counts.fallback_stage = PhotonFinalStateFallback;
          if (final.suppression_flag || final.continuation_flag) {
            source_counts.next = 1;
          } else if (final.record_flag) {
            auto const process = final.parameters.process_id;
            if (process == ComptonProcessId) {
              source_counts.next =
                  (final.parameters.thinning_keep_mask & 0x1U) != 0;
              source_counts.charged =
                  (final.parameters.thinning_keep_mask & 0x2U) != 0;
            } else {
              source_counts.charged = final.child_count;
            }
          }
        };
    static_assert(sizeof(final_state_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_photon_final_state",
        Policy(execution, 0, current_count),
        final_state_kernel);

    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_outputs",
        Policy(execution, 0, current_count),
        ResidentPhotonScanFunctor<ExecutionSpace>{counts, offsets},
        workspace.scan_totals);

    auto const offsets_raw = rawDeviceView(offsets);
    auto const steps_raw = rawDeviceView(steps);
    auto const projected_steps_raw = rawDeviceView(projected_steps);
    auto const records_raw = rawDeviceView(records);
    auto const charged_raw = rawDeviceView(charged);
    auto const fallbacks_raw = rawDeviceView(fallbacks);
    auto const observations_raw = rawDeviceView(observations);
    auto const first_snapshots_raw = rawDeviceView(first_snapshots);
    auto const first_interaction_candidates_raw =
        first_interaction_candidates.data();
    auto const scan_totals_raw = scan_totals_device.data();
    auto const error_raw = error.data();
    auto const materialize_endpoint =
        KOKKOS_LAMBDA(std::size_t const source) {
          auto const device_totals = *scan_totals_raw;
          if (device_totals.values[PhotonNextOffset] > capacity ||
              device_totals.values[PhotonChargedOffset] > charged_capacity)
            return;
          auto const& selection = selections_raw(source);
          auto const& transport = transports_raw(source);
          auto const& source_counts = counts_raw(source);
          if (source_counts.fallback_stage > PhotonFinalStateFallback) {
            Kokkos::atomic_compare_exchange(
                error_raw, std::uint32_t{0}, std::uint32_t{21});
            return;
          }
          if (source_counts.step) {
            steps_raw(offsets_raw(source, PhotonStepOffset)) =
                transport;
            if (project_steps) {
              auto const& context = physics_context();
              projected_steps_raw(
                  offsets_raw(source, PhotonStepOffset)) =
                  detail::projectPhotonStep(
                      context.profile_projection, transport);
            }
          }
          if (selection.fallback_flag) {
            if (source_counts.fallback_stage !=
                PhotonSelectionFallback) {
              Kokkos::atomic_compare_exchange(
                  error_raw, std::uint32_t{0}, std::uint32_t{22});
              return;
            }
            fallbacks_raw(
                photonFallbackOutputOffset(
                    source_counts.fallback_stage,
                    offsets_raw(source, PhotonSelectionFallbackOffset),
                    offsets_raw(source, PhotonTransportFallbackOffset),
                    offsets_raw(source, PhotonFinalStateFallbackOffset),
                    device_totals)) =
                selection.fallback;
            return;
          }
          if (!source_counts.step) {
            if (source_counts.fallback_stage !=
                PhotonTransportFallback) {
              Kokkos::atomic_compare_exchange(
                  error_raw, std::uint32_t{0}, std::uint32_t{23});
              return;
            }
            fallbacks_raw(
                photonFallbackOutputOffset(
                    source_counts.fallback_stage,
                    offsets_raw(source, PhotonSelectionFallbackOffset),
                    offsets_raw(source, PhotonTransportFallbackOffset),
                    offsets_raw(source, PhotonFinalStateFallbackOffset),
                    device_totals)) =
                transport_fallbacks_raw(source);
            return;
          }
          auto const& step = transport;
          if (source_counts.observation) {
            ObservationRecord observation{};
            observation.particle = step.end;
            observation.status =
                step.limit == PhotonTransportLimit::EscapedEnvironment
                    ? ObservationStatus::EscapedEnvironment
                    : ObservationStatus::ReachedObservationSurface;
            observations_raw(
                offsets_raw(source, PhotonObservationOffset)) =
                observation;
          }
          if (step.limit == PhotonTransportLimit::LayerBoundary) {
            next.store(offsets_raw(source, PhotonNextOffset),
                       step.end);
          }
        };
    static_assert(sizeof(materialize_endpoint) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_photon_materialize_endpoints",
        Policy(execution, 0, current_count), materialize_endpoint);

    // Materializing an interaction carries the complete final-state
    // kinematics and therefore substantially more live state than writing
    // a transport endpoint.  Run it only for the stable interaction list;
    // output offsets and history IDs still come from the source-ordered
    // composite scan above.
    auto const materialize_kernel =
        KOKKOS_LAMBDA(std::size_t const interaction_index) {
          auto const device_totals = *scan_totals_raw;
          if (device_totals.values[PhotonNextOffset] > capacity ||
              device_totals.values[PhotonChargedOffset] > charged_capacity)
            return;
          auto const compacted_count =
              static_cast<std::size_t>(*interaction_count_raw);
          if (interaction_index >= compacted_count) return;
          auto const source = interaction_sources_raw(interaction_index);
          auto const& step = transports_raw(source);
          auto const& final = final_states_raw(interaction_index);
          auto const& source_counts = counts_raw(source);
          if ((final.fallback_flag != 0) !=
              (source_counts.fallback_stage ==
               PhotonFinalStateFallback)) {
            Kokkos::atomic_compare_exchange(
                error_raw, std::uint32_t{0}, std::uint32_t{24});
            return;
          }
          ResidentPhotonFinalStateWriter output{
              next,
              records_raw,
              charged_raw,
              fallbacks_raw,
              first_snapshots_raw,
              first_interaction_candidates_raw,
              error_raw,
              offsets_raw(source, PhotonNextOffset),
              offsets_raw(source, PhotonChargedOffset),
              photonFallbackOutputOffset(
                  source_counts.fallback_stage,
                  offsets_raw(source, PhotonSelectionFallbackOffset),
                  offsets_raw(source, PhotonTransportFallbackOffset),
                  offsets_raw(source, PhotonFinalStateFallbackOffset),
                  device_totals),
              offsets_raw(source, PhotonRecordOffset),
              capture_first_interaction ? 1U : 0U};
          detail::materializePhotonFinalStateInPlace(
              step.interaction, final,
              offsets_raw(source, PhotonChildOffset),
              next_history_id, output);
        };
    static_assert(sizeof(materialize_kernel) < 512);
    Kokkos::parallel_for(
        "c8_kokkos_resident_photon_materialize_interactions",
        Policy(execution, 0, current_count),
        materialize_kernel);

  }

  template <class ExecutionSpace>
  gpu::em::ResidentPhotonCascadeResult runResidentPhotonCascade(
      KokkosPhysicsContextView<ExecutionSpace> const physics_context,
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t const first_secondary_history_id,
      std::size_t const maximum_wavefronts,
      std::size_t const minimum_resident_batch_size,
      std::optional<gpu::em::GpuFirstInteractionSnapshot>& first_interaction,
      bool const project_steps = false,
      gpu::em::detail::DeviceProfileProjection const profile_projection = {},
      KokkosProfileAccumulator<ExecutionSpace>* const profile_accumulator =
          nullptr,
      KokkosResidentPhotonWorkspace<ExecutionSpace>* const workspace =
          nullptr,
      KokkosPendingParticleQueue<ExecutionSpace>* const pending_photons =
          nullptr,
      KokkosPendingParticleQueue<ExecutionSpace>* const pending_leptons =
          nullptr,
      std::size_t const pending_input_count = 0,
      std::size_t const maximum_pending_particles =
          std::numeric_limits<std::size_t>::max(),
      ExecutionSpace const& execution = {},
      bool const capture_diagnostic_interactions = false,
      std::size_t const openmp_chunk_size = 0,
      KokkosResidentMemoryBudgetGate const memory_gate = {},
      ResidentExecutionWait<ExecutionSpace>* const cooperative_wait = nullptr) {
    using namespace gpu::em;
    auto const Policy =
        [openmp_chunk_size](ExecutionSpace const& selected_execution,
                            std::size_t const begin,
                            std::size_t const end) {
          if (begin != 0 || end < begin)
            throw std::invalid_argument(
                "resident photon RangePolicy requires a zero origin");
          return makeKokkosRangePolicy(
              selected_execution, end, openmp_chunk_size);
        };
    ResidentPhotonCascadeResult result{};
    if ((pending_photons == nullptr && pending_input_count != 0) ||
        (pending_photons != nullptr &&
         pending_input_count > pending_photons->size()))
      throw std::invalid_argument(
          "resident Kokkos photon pending-input count is invalid");
    auto const pending_input = pending_input_count;
    if (pending_input > std::numeric_limits<std::size_t>::max() -
                            particles.size())
      throw std::overflow_error(
          "resident Kokkos photon input size overflow");
    auto const total_input = pending_input + particles.size();
    result.input_particles = total_input;
    if (total_input == 0) {
      result.completed = true;
      return result;
    }
    if (first_secondary_history_id == 0 || maximum_wavefronts == 0 ||
        minimum_resident_batch_size == 0)
      throw std::invalid_argument(
          "resident Kokkos photon cascade requires nonzero limits");

    auto const capacity = total_input;
    if (capacity > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error(
          "resident Kokkos photon input exceeds addressable charged capacity");
    auto const charged_capacity = capacity * 2;
    KokkosResidentPhotonWorkspace<ExecutionSpace> local_workspace;
    auto& active_workspace = workspace != nullptr ? *workspace : local_workspace;
    auto spill_unstaged_input = [&]() {
      result.workspace_limit_checkpoint = true;
      result.completed = true;
      result.cpu_spill_particles.reserve(total_input);
      if (pending_input != 0) {
        auto prefix = pending_photons->downloadPrefix(
            pending_input, execution);
        result.cpu_spill_particles.insert(
            result.cpu_spill_particles.end(),
            std::make_move_iterator(prefix.begin()),
            std::make_move_iterator(prefix.end()));
        pending_photons->consume(pending_input);
      }
      result.cpu_spill_particles.insert(
          result.cpu_spill_particles.end(), particles.begin(), particles.end());
    };
    auto const initial_projection =
        active_workspace.projectedCapacity(capacity);
    if (!memory_gate.permits(
            active_workspace.deviceBytes(), initial_projection,
            "Kokkos photon resident workspace growth")) {
      spill_unstaged_input();
      return result;
    }
    active_workspace.ensureCapacity(capacity, execution);
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
      queue.upload(particles, pending_photons->view(),
                   pending_photons->head(), pending_input, execution);
      pending_photons->consume(pending_input);
    } else {
      queue.upload(particles, execution);
    }
    auto const& selections = active_workspace.selections;
    auto const& transports = active_workspace.transports;
    auto const& transport_fallbacks = active_workspace.transport_fallbacks;
    auto const& final_states = active_workspace.final_states;
    auto const& counts = active_workspace.counts;
    auto const& interaction_flags = active_workspace.interaction_flags;
    auto const& interaction_sources = active_workspace.interaction_sources;
    auto const& offsets = active_workspace.offsets;
    auto const& steps = active_workspace.steps;
    auto const& projected_steps = active_workspace.projected_steps;
    auto const& records = active_workspace.records;
    auto const& charged = active_workspace.charged;
    auto const& fallbacks = active_workspace.fallbacks;
    auto const& observations = active_workspace.observations;
    auto const& first_snapshots = active_workspace.first_snapshots;
    auto const& first_interaction_candidates =
        active_workspace.first_interaction_candidates;
    auto const& error = active_workspace.error;
    auto const& call_statistics = active_workspace.call_statistics;
    auto const interaction_count_device = active_workspace.interaction_count;
    auto const scan_totals_device = active_workspace.scan_totals;
    auto const front_control_device = active_workspace.front_control;
    auto const host_front_control = active_workspace.host_front_control;
    Kokkos::deep_copy(execution, call_statistics, std::uint64_t{0});

    auto append_host = [](auto& destination, auto const& host) {
      auto const count = host.extent(0);
      if (count == 0) return;
      auto const old = destination.size();
      destination.resize(old + count);
      for (std::size_t i = 0; i < count; ++i)
        destination[old + i] = host(i);
    };

    std::uint64_t next_history_id = first_secondary_history_id;
    result.peak_resident_photons = queue.size();
    auto const capture_first_interaction = !first_interaction;
    if (capture_first_interaction)
      Kokkos::deep_copy(
          execution, first_interaction_candidates, std::uint32_t{0});
    while (!queue.empty() && result.wavefronts < maximum_wavefronts) {
      auto const current_count = queue.size();
      if (result.wavefronts != 0 &&
          current_count < minimum_resident_batch_size) {
        result.below_minimum_batch_checkpoint = true;
        break;
      }
      auto const candidate_projection =
          active_workspace.projectedCandidateCapacity(current_count);
      if (!memory_gate.permits(
              active_workspace.deviceBytes(), candidate_projection,
              "Kokkos photon final-state workspace growth")) {
        spill_active_front();
        break;
      }
      active_workspace.ensureCandidateCapacity(current_count, execution);
      auto const bucketing_projection =
          active_workspace.projectedBucketingCapacity(
              current_count, execution);
      if (!memory_gate.permits(
              active_workspace.deviceBytes(), bucketing_projection,
              "Kokkos photon bucketing workspace growth")) {
        spill_active_front();
        break;
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
      enqueueResidentPhotonFront(
          physics_context, current, queue.next().rawDeviceView(),
          active_workspace, current_count, capacity, next_history_id,
          project_steps, capture_first_interaction, execution,
          openmp_chunk_size);

      // Scan totals, compacted interaction count and materialization error
      // are unmanaged views onto this single device POD.  Copy it only after
      // all producers in the execution stream have completed; this preserves
      // the original one-D2H/one-fence host checkpoint without launching a
      // one-thread packing kernel for every resident wavefront.
      Kokkos::deep_copy(
          execution, host_front_control, front_control_device);
      waitResidentExecution(execution,
          "download resident Kokkos photon front control", cooperative_wait);
      auto const& front_control = host_front_control();
      auto const& totals = front_control.totals.values;
      if (front_control.interaction_count > current_count)
        throw std::runtime_error(
            "resident Kokkos photon interaction compaction overflow");
      auto const interaction_count = static_cast<std::size_t>(
          front_control.interaction_count);
      auto const selection_fallback_count =
          totals[PhotonSelectionFallbackOffset];
      auto const transport_fallback_count =
          totals[PhotonTransportFallbackOffset];
      auto const final_state_fallback_count =
          totals[PhotonFinalStateFallbackOffset];
      auto const total_fallback_count = selection_fallback_count +
                                        transport_fallback_count +
                                        final_state_fallback_count;
      if (selection_fallback_count > current_count ||
          total_fallback_count > current_count)
        throw std::runtime_error(
            "resident Kokkos photon fallback accounting overflow");
      result.selected_interactions +=
          current_count - static_cast<std::size_t>(selection_fallback_count);
      result.interaction_bearing_wavefronts += interaction_count != 0;
      if (totals[PhotonNextOffset] > capacity ||
          totals[PhotonChargedOffset] > charged_capacity)
        throw std::runtime_error(
            "resident Kokkos photon endpoint capacity exceeded");
      if (front_control.materialization_error != 0)
        throw std::runtime_error(
            "resident Kokkos photon final-state materialization failed");

      auto const step_count =
          static_cast<std::size_t>(totals[PhotonStepOffset]);
      auto const record_count =
          static_cast<std::size_t>(totals[PhotonRecordOffset]);
      if (step_count > current_count)
        throw std::runtime_error(
            "resident Kokkos photon packed step count exceeds source count");
      if (record_count > interaction_count)
        throw std::runtime_error(
            "resident Kokkos photon packed record count exceeds interaction "
            "count");

      if (profile_accumulator != nullptr) {
        auto const device_profile = profile_accumulator->deviceView();
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_photon_transport_statistics",
            Policy(execution, 0, current_count),
            ResidentPhotonTransportStatisticsProfileFunctor<ExecutionSpace>{
                {selections, steps, step_count, call_statistics},
                profile_projection,
                device_profile});
        if (interaction_count != 0)
          Kokkos::parallel_reduce(
              "c8_kokkos_resident_photon_interaction_statistics",
              Policy(execution, 0, interaction_count),
              ResidentPhotonInteractionStatisticsProfileFunctor<
                  ExecutionSpace>{{final_states, call_statistics},
                                  profile_projection,
                                  device_profile,
                                  steps,
                                  step_count,
                                  records,
                                  record_count});
      } else {
        Kokkos::parallel_reduce(
            "c8_kokkos_resident_photon_transport_statistics",
            Policy(execution, 0, current_count),
            ResidentPhotonTransportStatisticsFunctor<ExecutionSpace>{
                selections, steps, step_count, call_statistics});
        if (interaction_count != 0)
          Kokkos::parallel_reduce(
              "c8_kokkos_resident_photon_interaction_statistics",
              Policy(execution, 0, interaction_count),
              ResidentPhotonInteractionStatisticsFunctor<ExecutionSpace>{
                  final_states, call_statistics});
      }

      auto const charged_count =
          static_cast<std::size_t>(totals[PhotonChargedOffset]);
      auto const fallback_count =
          static_cast<std::size_t>(total_fallback_count);
      auto const observation_count =
          static_cast<std::size_t>(totals[PhotonObservationOffset]);
      auto const copy_projected = project_steps && step_count != 0;
      auto const copy_steps =
          (capture_diagnostic_interactions ||
           (!project_steps && profile_accumulator == nullptr)) &&
          step_count != 0;
      auto const copy_records =
          profile_accumulator == nullptr && record_count != 0;
      bool copy_charged = pending_leptons == nullptr && charged_count != 0;
      // Match the native-CUDA resident-output contract: when the complete
      // profile ledger is accumulated on the execution space, final-state
      // records remain resident as well.  Returning them without their
      // matching transport records would make the host router account the
      // same vertex a second time and breaks the history ledger.
      if (pending_leptons != nullptr) {
        if (!pending_leptons->tryAppend(
                charged, charged_count,
                maximum_pending_particles, execution, memory_gate,
                "Kokkos pending lepton queue growth"))
          copy_charged = charged_count != 0;
      }

      active_workspace.ensureHostOutputCapacity(
          copy_projected ? step_count : 0,
          copy_steps ? step_count : 0,
          copy_records ? record_count : 0,
          copy_charged ? charged_count : 0, fallback_count,
          observation_count);

      auto const active_steps =
          std::make_pair<std::size_t>(0, copy_steps ? step_count : 0);
      auto const active_projected =
          std::make_pair<std::size_t>(0, copy_projected ? step_count : 0);
      auto const active_records =
          std::make_pair<std::size_t>(0, copy_records ? record_count : 0);
      auto const active_charged =
          std::make_pair<std::size_t>(0, copy_charged ? charged_count : 0);
      auto const active_fallbacks =
          std::make_pair<std::size_t>(0, fallback_count);
      auto const active_observations =
          std::make_pair<std::size_t>(0, observation_count);
      auto projected_source =
          Kokkos::subview(projected_steps, active_projected);
      auto step_source = Kokkos::subview(steps, active_steps);
      auto record_source = Kokkos::subview(records, active_records);
      auto charged_source = Kokkos::subview(charged, active_charged);
      auto fallback_source = Kokkos::subview(fallbacks, active_fallbacks);
      auto observation_source =
          Kokkos::subview(observations, active_observations);
      auto host_projected = Kokkos::subview(
          active_workspace.host_projected_steps, active_projected);
      auto host_steps =
          Kokkos::subview(active_workspace.host_steps, active_steps);
      auto host_records =
          Kokkos::subview(active_workspace.host_records, active_records);
      auto host_charged =
          Kokkos::subview(active_workspace.host_charged, active_charged);
      auto host_fallbacks =
          Kokkos::subview(active_workspace.host_fallbacks, active_fallbacks);
      auto host_observations = Kokkos::subview(
          active_workspace.host_observations, active_observations);

      // Queue every host-visible output transfer first.  One fence then makes
      // all mirrors visible and also completes endpoint materialization in a
      // no-interaction wavefront.  This preserves the source-ordered scans
      // while avoiding one synchronization per output vector.
      if (copy_projected)
        Kokkos::deep_copy(execution, host_projected, projected_source);
      if (copy_steps)
        Kokkos::deep_copy(execution, host_steps, step_source);
      if (copy_records)
        Kokkos::deep_copy(execution, host_records, record_source);
      if (copy_charged)
        Kokkos::deep_copy(execution, host_charged, charged_source);
      if (fallback_count != 0)
        Kokkos::deep_copy(execution, host_fallbacks, fallback_source);
      if (observation_count != 0)
        Kokkos::deep_copy(execution, host_observations, observation_source);
      auto const has_host_outputs =
          copy_projected || copy_steps || copy_records || copy_charged ||
          fallback_count != 0 || observation_count != 0;
      if (has_host_outputs)
        waitResidentExecution(execution,
            "download resident Kokkos photon outputs", cooperative_wait);

      if (copy_projected)
        append_host(result.projected_step_records, host_projected);
      auto const first_copied_step = result.step_records.size();
      if (copy_steps) append_host(result.step_records, host_steps);
      if (capture_diagnostic_interactions) {
        for (auto index = first_copied_step;
             index < result.step_records.size(); ++index) {
          auto const& step = result.step_records[index];
          if (step.limit == PhotonTransportLimit::Interaction)
            result.interaction_records.push_back(step.interaction);
        }
      }
      if (copy_records)
        append_host(result.final_state_records, host_records);
      if (copy_charged) {
        if (pending_leptons != nullptr)
          append_host(result.cpu_spill_particles, host_charged);
        else
          append_host(result.electromagnetic_secondaries, host_charged);
      }
      if (fallback_count != 0)
        append_host(result.fallback_events, host_fallbacks);
      if (observation_count != 0)
        append_host(result.observations, host_observations);
      result.transport_records += totals[PhotonStepOffset];
      if (totals[PhotonChildOffset] >
          std::numeric_limits<std::uint64_t>::max() - next_history_id)
        throw std::overflow_error(
            "resident Kokkos photon history ID overflow");
      next_history_id += totals[PhotonChildOffset];
      queue.commitNextAlreadySynchronized(totals[PhotonNextOffset]);
      result.wavefronts++;
      result.peak_resident_photons =
          std::max(result.peak_resident_photons, queue.size());
    }
    auto const host_call_statistics = active_workspace.host_call_statistics;
    auto const host_candidates =
        active_workspace.host_first_interaction_candidates;
    auto const host_snapshot = active_workspace.host_first_snapshots;
    Kokkos::deep_copy(execution, host_call_statistics, call_statistics);
    if (capture_first_interaction) {
      Kokkos::deep_copy(
          execution, host_candidates, first_interaction_candidates);
      // Copying the single snapshot speculatively lets statistics, candidate
      // count and first-interaction state share one call-end synchronization.
      // It is read only when the candidate count confirms initialization.
      Kokkos::deep_copy(execution, host_snapshot, first_snapshots);
    }
    waitResidentExecution(execution,
        "download resident Kokkos photon call results", cooperative_wait);
    queue.markExecutionSynchronized();
    queue_execution_guard.release();
    result.interaction_vertices += host_call_statistics(
        PhotonCallTransportStatisticOffset +
        PhotonTransportInteractionVertices);
    result.layer_boundaries += host_call_statistics(
        PhotonCallTransportStatisticOffset + PhotonTransportLayerBoundaries);
    result.particle_cuts += host_call_statistics(
        PhotonCallTransportStatisticOffset + PhotonTransportParticleCuts);
    result.native_newton_iterations += host_call_statistics(
        PhotonCallTransportStatisticOffset +
        PhotonSelectionNativeNewtonIterations);
    result.native_bisection_iterations += host_call_statistics(
        PhotonCallTransportStatisticOffset +
        PhotonSelectionNativeBisectionIterations);
    result.native_inverse_failures += host_call_statistics(
        PhotonCallTransportStatisticOffset +
        PhotonSelectionNativeInverseFailures);
    result.lpm_suppressions += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionLpmSuppressions);
    auto& process_statistics = result.process_statistics;
    process_statistics.gpu_final_states += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionGpuFinalStates);
    process_statistics.physical_secondaries_generated += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionPhysicalSecondaries);
    process_statistics.photon_pair_final_states += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionPairFinalStates);
    process_statistics.compton_final_states += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionComptonFinalStates);
    process_statistics.photoelectric_final_states += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionPhotoelectricFinalStates);
    process_statistics.photon_pair_lpm_trials += host_call_statistics(
        PhotonCallInteractionStatisticOffset + PhotonInteractionLpmTrials);
    process_statistics.photon_pair_lpm_suppressions += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionLpmSuppressions);
    process_statistics.thinning_hillas_vertices += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionThinningHillasVertices);
    process_statistics.thinning_statistical_vertices += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionThinningStatisticalVertices);
    process_statistics.thinning_particles_discarded += host_call_statistics(
        PhotonCallInteractionStatisticOffset +
        PhotonInteractionThinningParticlesDiscarded);
    if (capture_first_interaction) {
      if (host_candidates(0) > 1) {
        throw std::runtime_error(
            "resident Kokkos photon transport produced more than one "
            "generation-zero interaction candidate");
      }
      if (host_candidates(0) == 1) {
        first_interaction = host_snapshot(0);
      }
    }
    result.completed = queue.empty();
    if (!result.completed)
      result.remaining_photons = queue.download(execution, cooperative_wait);
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
