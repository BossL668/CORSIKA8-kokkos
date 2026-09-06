/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <corsika/accelerator/em/detail/PhotonFinalStateStep.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct KokkosPhotonFinalStateResult {
    gpu::em::EmFinalStateBatchResult batch{};
    std::optional<gpu::em::GpuFirstInteractionSnapshot> first_interaction{};
  };

  /**
   * Kokkos implementation of photon pair, Compton and photoelectric final
   * states.  Child identities are allocated by exclusive scans, so changing
   * team size or execution space cannot change the shower history tree.
   */
  template <class ExecutionSpace>
  KokkosPhotonFinalStateResult generatePhotonFinalStates(
      gpu::em::tables::NativePhysicsView const physics,
      gpu::em::PhotonPairLpmSnapshot const lpm_snapshot,
      gpu::em::EmThinningConfig const thinning,
      std::vector<gpu::em::EmInteractionRecord> const& input,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      std::uint64_t const first_secondary_history_id,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    using Classification = detail::PhotonFinalStateClassification;
    using Materialization = detail::PhotonFinalStateMaterialization;
    KokkosPhotonFinalStateResult result{};
    result.batch.input_interactions = input.size();
    if (input.empty()) return result;
    if (first_secondary_history_id == 0) {
      throw std::invalid_argument(
          "Kokkos photon secondary history IDs must start above zero");
    }
    if (input.size() > std::numeric_limits<std::uint32_t>::max() / 2) {
      throw std::length_error(
          "Kokkos photon final-state batch exceeds 32-bit offsets");
    }

    auto const count = input.size();
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> interactions(
        "c8_kokkos_photon_final_state_input", count);
    Kokkos::View<Classification*, Memory> classifications(
        "c8_kokkos_photon_final_state_classifications", count);
    Kokkos::View<std::uint32_t*[5], Memory> offsets(
        "c8_kokkos_photon_final_state_offsets", count);
    Kokkos::View<Materialization*, Memory> materialized(
        "c8_kokkos_photon_final_state_materialized", count);
    Kokkos::View<gpu::em::PhotonPairFinalStateRecord*, Memory> records(
        "c8_kokkos_photon_final_state_records", count);
    Kokkos::View<gpu::em::EmParticleState*, Memory> secondaries(
        "c8_kokkos_photon_final_state_secondaries", 2 * count);
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_photon_final_state_fallbacks", count);
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> continuations(
        "c8_kokkos_photon_final_state_continuations", count);
    Kokkos::View<gpu::em::PhotonPairLpmSuppressionRecord*, Memory>
        suppressions("c8_kokkos_photon_final_state_suppressions", count);
    Kokkos::View<std::uint32_t*, Memory> error(
        "c8_kokkos_photon_final_state_error", 1);
    Kokkos::deep_copy(execution, error, std::uint32_t{0});

    auto host_input = Kokkos::create_mirror_view(interactions);
    for (std::size_t i = 0; i < count; ++i) host_input(i) = input[i];
    Kokkos::deep_copy(execution, interactions, host_input);
    Kokkos::parallel_for(
        "c8_kokkos_classify_photon_final_states",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          classifications(index) = detail::classifyPhotonFinalState(
              physics, lpm_snapshot, thinning, interactions(index),
              random_seed, shower_id);
        });

    std::uint32_t totals[5]{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_children", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 0) = update;
          update += classifications(index).child_count;
        },
        totals[0]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_records", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 1) = update;
          update += classifications(index).record_flag;
        },
        totals[1]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_fallbacks", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 2) = update;
          update += classifications(index).fallback_flag;
        },
        totals[2]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_continuations", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 3) = update;
          update += classifications(index).continuation_flag;
        },
        totals[3]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_photon_suppressions", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 4) = update;
          update += classifications(index).suppression_flag;
        },
        totals[4]);

    Kokkos::parallel_for(
        "c8_kokkos_materialize_photon_final_states",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const classification = classifications(index);
          auto const output = detail::materializePhotonFinalState(
              interactions(index), classification, offsets(index, 0),
              first_secondary_history_id);
          materialized(index) = output;
          if (output.error != 0) {
            Kokkos::atomic_compare_exchange(error.data(), std::uint32_t{0},
                                            output.error);
            return;
          }
          if (output.has_record != 0) {
            records(offsets(index, 1)) = output.record;
            for (std::uint32_t child = 0; child < output.secondary_count;
                 ++child) {
              secondaries(offsets(index, 0) + child) =
                  output.secondaries[child];
            }
          } else if (output.has_fallback != 0) {
            fallbacks(offsets(index, 2)) = output.fallback;
          } else if (output.has_continuation != 0) {
            continuations(offsets(index, 3)) = output.continuation;
          } else if (output.has_suppression != 0) {
            suppressions(offsets(index, 4)) = output.suppression;
          }
        });
    execution.fence("complete Kokkos photon final states");

    auto host_error = Kokkos::create_mirror_view(error);
    Kokkos::deep_copy(execution, host_error, error);
    execution.fence("download Kokkos photon final-state status");
    if (host_error(0) != 0) {
      throw std::runtime_error(
          "Kokkos photon final-state materialization failed");
    }
    if (totals[1] + totals[2] + totals[3] + totals[4] != count) {
      throw std::runtime_error(
          "Kokkos photon final-state partition is not exhaustive");
    }
    auto host_records = Kokkos::create_mirror_view(records);
    auto host_secondaries = Kokkos::create_mirror_view(secondaries);
    auto host_fallbacks = Kokkos::create_mirror_view(fallbacks);
    auto host_continuations = Kokkos::create_mirror_view(continuations);
    auto host_suppressions = Kokkos::create_mirror_view(suppressions);
    auto host_classifications = Kokkos::create_mirror_view(classifications);
    auto host_materialized = Kokkos::create_mirror_view(materialized);
    Kokkos::deep_copy(execution, host_records, records);
    Kokkos::deep_copy(execution, host_secondaries, secondaries);
    Kokkos::deep_copy(execution, host_fallbacks, fallbacks);
    Kokkos::deep_copy(execution, host_continuations, continuations);
    Kokkos::deep_copy(execution, host_suppressions, suppressions);
    Kokkos::deep_copy(execution, host_classifications, classifications);
    Kokkos::deep_copy(execution, host_materialized, materialized);
    execution.fence("download Kokkos photon final states");

    result.batch.gpu_interactions = totals[1];
    result.batch.final_state_records.resize(totals[1]);
    result.batch.secondaries.resize(totals[0]);
    result.batch.fallback_events.resize(totals[2]);
    result.batch.continuations.resize(totals[3]);
    result.batch.lpm_suppressed.resize(totals[4]);
    for (std::size_t i = 0; i < totals[1]; ++i)
      result.batch.final_state_records[i] = host_records(i);
    for (std::size_t i = 0; i < totals[0]; ++i)
      result.batch.secondaries[i] = host_secondaries(i);
    for (std::size_t i = 0; i < totals[2]; ++i)
      result.batch.fallback_events[i] = host_fallbacks(i);
    for (std::size_t i = 0; i < totals[3]; ++i)
      result.batch.continuations[i] = host_continuations(i);
    for (std::size_t i = 0; i < totals[4]; ++i)
      result.batch.lpm_suppressed[i] = host_suppressions(i);
    for (std::size_t i = 0; i < count; ++i) {
      auto const& classification = host_classifications(i);
      result.batch.photon_pair_interactions +=
          classification.photon_pair_flag;
      result.batch.compton_interactions += classification.compton_flag;
      result.batch.photoelectric_interactions +=
          classification.photoelectric_flag;
      auto const& output = host_materialized(i);
      if (!result.first_interaction && output.has_first_interaction != 0) {
        result.first_interaction = output.first_interaction;
      }
    }
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
