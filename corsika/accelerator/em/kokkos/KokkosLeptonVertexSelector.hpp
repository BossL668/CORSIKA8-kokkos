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

#include <corsika/accelerator/em/detail/LeptonVertexSelection.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct KokkosLeptonVertexResult {
    gpu::em::LeptonVertexSelectionBatchResult batch{};
    std::uint64_t native_newton_iterations{};
    std::uint64_t native_bisection_iterations{};
    std::uint64_t native_inverse_failures{};
  };

  template <class ExecutionSpace>
  KokkosLeptonVertexResult selectLeptonVertices(
      gpu::em::tables::NativePhysicsView const physics,
      std::vector<gpu::em::EmInteractionRecord> const& input,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      ExecutionSpace const& execution = {}) {
    using Memory = typename ExecutionSpace::memory_space;
    using Policy = Kokkos::RangePolicy<ExecutionSpace>;
    using Outcome = detail::LeptonVertexSelectionOutcome;
    KokkosLeptonVertexResult result{};
    result.batch.input_candidates = input.size();
    if (input.empty()) return result;
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "Kokkos lepton vertex selection exceeds 32-bit stable offsets");
    }
    auto const count = input.size();
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> candidates(
        "c8_kokkos_lepton_vertex_candidates", count);
    Kokkos::View<Outcome*, Memory> outcomes(
        "c8_kokkos_lepton_vertex_outcomes", count);
    Kokkos::View<std::uint32_t*[3], Memory> offsets(
        "c8_kokkos_lepton_vertex_offsets", count);
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> interactions(
        "c8_kokkos_lepton_vertex_interactions", count);
    Kokkos::View<gpu::em::EmInteractionRecord*, Memory> continuations(
        "c8_kokkos_lepton_vertex_continuations", count);
    Kokkos::View<gpu::em::ProposalFallbackEvent*, Memory> fallbacks(
        "c8_kokkos_lepton_vertex_fallbacks", count);
    auto host_input = Kokkos::create_mirror_view(candidates);
    for (std::size_t i = 0; i < count; ++i) host_input(i) = input[i];
    Kokkos::deep_copy(execution, candidates, host_input);
    Kokkos::parallel_for(
        "c8_kokkos_select_lepton_vertices", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          outcomes(index) = detail::selectLeptonVertex(
              physics, candidates(index), random_seed, shower_id);
        });
    std::uint32_t totals[3]{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_vertex_interactions",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 0) = update;
          update += outcomes(index).interaction_flag;
        }, totals[0]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_vertex_continuations",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 1) = update;
          update += outcomes(index).continuation_flag;
        }, totals[1]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_vertex_fallbacks",
        Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update,
                      bool const final) {
          if (final) offsets(index, 2) = update;
          update += outcomes(index).fallback_flag;
        }, totals[2]);
    Kokkos::parallel_for(
        "c8_kokkos_compact_lepton_vertices", Policy(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const outcome = outcomes(index);
          if (outcome.interaction_flag != 0u)
            interactions(offsets(index, 0)) = outcome.record;
          else if (outcome.continuation_flag != 0u)
            continuations(offsets(index, 1)) = outcome.record;
          else if (outcome.fallback_flag != 0u)
            fallbacks(offsets(index, 2)) = outcome.fallback;
        });
    execution.fence("complete Kokkos lepton vertex selection");
    if (static_cast<std::size_t>(totals[0]) + totals[1] + totals[2] != count)
      throw std::runtime_error(
          "Kokkos lepton vertex classification does not conserve inputs");

    auto host_outcomes = Kokkos::create_mirror_view(outcomes);
    auto host_interactions = Kokkos::create_mirror_view(interactions);
    auto host_continuations = Kokkos::create_mirror_view(continuations);
    auto host_fallbacks = Kokkos::create_mirror_view(fallbacks);
    Kokkos::deep_copy(execution, host_outcomes, outcomes);
    Kokkos::deep_copy(execution, host_interactions, interactions);
    Kokkos::deep_copy(execution, host_continuations, continuations);
    Kokkos::deep_copy(execution, host_fallbacks, fallbacks);
    execution.fence("download Kokkos lepton vertex selection");
    result.batch.interactions.resize(totals[0]);
    result.batch.continuations.resize(totals[1]);
    result.batch.fallback_events.resize(totals[2]);
    for (std::size_t i = 0; i < totals[0]; ++i)
      result.batch.interactions[i] = host_interactions(i);
    for (std::size_t i = 0; i < totals[1]; ++i)
      result.batch.continuations[i] = host_continuations(i);
    for (std::size_t i = 0; i < totals[2]; ++i)
      result.batch.fallback_events[i] = host_fallbacks(i);
    for (std::size_t i = 0; i < count; ++i) {
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
