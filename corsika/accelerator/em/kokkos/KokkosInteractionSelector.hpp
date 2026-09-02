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
#include <stdexcept>
#include <vector>

#include <corsika/accelerator/em/detail/InteractionSelection.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct KokkosInteractionSelectionResult {
    gpu::em::EmInteractionBatchResult batch{};
    std::uint64_t native_newton_iterations{};
    std::uint64_t native_bisection_iterations{};
    std::uint64_t native_inverse_failures{};
  };

  /**
   * Selection plus stable flag/scan/scatter compaction shared by every Kokkos
   * execution space.  Each particle uses the same history-keyed Philox draws
   * and the same device function as native CUDA; only the launch mechanism
   * and memory space differ.
   */
  template <class ExecutionSpace>
  KokkosInteractionSelectionResult selectInteractions(
      gpu::em::tables::FlatRateTableView const physics,
      std::vector<gpu::em::EmParticleState> const& particles,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    using Outcome = accelerator::em::detail::InteractionSelectionOutcome;
    KokkosInteractionSelectionResult result{};
    result.batch.input_particles = particles.size();
    if (particles.empty()) return result;
    if (particles.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "Kokkos interaction selector exceeds 32-bit stable offsets");
    }

    Kokkos::View<gpu::em::EmParticleState*, Memory> device_particles(
        "c8_kokkos_selection_particles", particles.size());
    Kokkos::View<Outcome*, Memory> outcomes(
        "c8_kokkos_selection_outcomes", particles.size());
    Kokkos::View<std::uint32_t*, Memory> flags(
        "c8_kokkos_selection_fallback_flags", particles.size());
    Kokkos::View<std::uint32_t*, Memory> offsets(
        "c8_kokkos_selection_fallback_offsets", particles.size());
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> interactions(
        "c8_kokkos_selected_interactions", particles.size());
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_selection_fallbacks", particles.size());

    auto host_particles = Kokkos::create_mirror_view(device_particles);
    for (std::size_t i = 0; i < particles.size(); ++i) {
      host_particles(i) = particles[i];
    }
    Kokkos::deep_copy(execution, device_particles, host_particles);
    Kokkos::parallel_for(
        "c8_kokkos_select_interactions",
        Policy(execution, 0, particles.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const outcome =
              accelerator::em::detail::selectDiscreteInteraction(
                  physics, device_particles(index), index, random_seed,
                  shower_id);
          outcomes(index) = outcome;
          flags(index) = outcome.fallback_flag;
        });

    std::uint32_t fallback_count{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_interaction_fallbacks",
        Policy(execution, 0, particles.size()),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index) = update;
          update += flags(index);
        },
        fallback_count);
    Kokkos::parallel_for(
        "c8_kokkos_compact_interaction_selection",
        Policy(execution, 0, particles.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const fallback_offset = offsets(index);
          if (flags(index) != 0u) {
            fallbacks(fallback_offset) = outcomes(index).fallback;
          } else {
            interactions(index - fallback_offset) =
                outcomes(index).interaction;
          }
        });
    execution.fence("complete Kokkos interaction selection");

    auto host_interactions = Kokkos::create_mirror_view(interactions);
    auto host_fallbacks = Kokkos::create_mirror_view(fallbacks);
    auto host_outcomes = Kokkos::create_mirror_view(outcomes);
    Kokkos::deep_copy(execution, host_interactions, interactions);
    Kokkos::deep_copy(execution, host_fallbacks, fallbacks);
    Kokkos::deep_copy(execution, host_outcomes, outcomes);
    execution.fence("download Kokkos interaction selection");

    auto const interaction_count = particles.size() - fallback_count;
    result.batch.interactions.resize(interaction_count);
    result.batch.fallback_events.resize(fallback_count);
    for (std::size_t i = 0; i < interaction_count; ++i) {
      result.batch.interactions[i] = host_interactions(i);
    }
    for (std::size_t i = 0; i < fallback_count; ++i) {
      result.batch.fallback_events[i] = host_fallbacks(i);
    }
    for (std::size_t i = 0; i < particles.size(); ++i) {
      result.native_newton_iterations +=
          host_outcomes(i).native_newton_iterations;
      result.native_bisection_iterations +=
          host_outcomes(i).native_bisection_iterations;
      result.native_inverse_failures +=
          host_outcomes(i).native_inverse_failures;
    }
    return result;
  }

} // namespace corsika::accelerator::em::kokkos_detail
