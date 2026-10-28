/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class TExecutionSpace>
  LeptonVertexBatchResult select_lepton_vertices_batch(
      InteractionPhysicsView const physics, std::vector<InteractionRecord> const& input,
      std::uint64_t const randomSeed, std::uint64_t const showerId,
      TExecutionSpace const& execution) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using policy_type = Kokkos::RangePolicy<TExecutionSpace>;
    using outcome_type = LeptonVertexSelectionOutcome;
    LeptonVertexBatchResult result{};
    result.batch_.inputCandidates_ = input.size();
    if (input.empty()) return result;
    if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "Kokkos lepton vertex selection exceeds 32-bit stable offsets");
    }
    auto const count = input.size();
    Kokkos::View<InteractionRecord*, memory_space_type> candidates(
        "c8_kokkos_lepton_vertex_candidates", count);
    Kokkos::View<outcome_type*, memory_space_type> outcomes(
        "c8_kokkos_lepton_vertex_outcomes", count);
    Kokkos::View<std::uint32_t * [3], memory_space_type> offsets(
        "c8_kokkos_lepton_vertex_offsets", count);
    Kokkos::View<InteractionRecord*, memory_space_type> interactions(
        "c8_kokkos_lepton_vertex_interactions", count);
    Kokkos::View<InteractionRecord*, memory_space_type> continuations(
        "c8_kokkos_lepton_vertex_continuations", count);
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks(
        "c8_kokkos_lepton_vertex_fallbacks", count);
    auto hostInput = Kokkos::create_mirror_view(candidates);
    for (std::size_t i = 0; i < count; ++i) hostInput(i) = input[i];
    Kokkos::deep_copy(execution, candidates, hostInput);
    Kokkos::parallel_for(
        "c8_kokkos_select_lepton_vertices", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          outcomes(index) =
              select_lepton_vertex(physics, candidates(index), randomSeed, showerId);
        });
    std::uint32_t totals[3]{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_vertex_interactions", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 0) = update;
          update += outcomes(index).interactionFlag_;
        },
        totals[0]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_vertex_continuations", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 1) = update;
          update += outcomes(index).continuationFlag_;
        },
        totals[1]);
    Kokkos::parallel_scan(
        "c8_kokkos_scan_lepton_vertex_fallbacks", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index, 2) = update;
          update += outcomes(index).fallbackFlag_;
        },
        totals[2]);
    Kokkos::parallel_for(
        "c8_kokkos_compact_lepton_vertices", policy_type(execution, 0, count),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const outcome = outcomes(index);
          if (outcome.interactionFlag_ != 0u)
            interactions(offsets(index, 0)) = outcome.record_;
          else if (outcome.continuationFlag_ != 0u)
            continuations(offsets(index, 1)) = outcome.record_;
          else if (outcome.fallbackFlag_ != 0u)
            fallbacks(offsets(index, 2)) = outcome.fallback_;
        });
    execution.fence("complete Kokkos lepton vertex selection");
    if (static_cast<std::size_t>(totals[0]) + totals[1] + totals[2] != count)
      throw std::runtime_error(
          "Kokkos lepton vertex classification does not conserve inputs");

    auto hostOutcomes = Kokkos::create_mirror_view(outcomes);
    auto hostInteractions = Kokkos::create_mirror_view(interactions);
    auto hostContinuations = Kokkos::create_mirror_view(continuations);
    auto hostFallbacks = Kokkos::create_mirror_view(fallbacks);
    Kokkos::deep_copy(execution, hostOutcomes, outcomes);
    Kokkos::deep_copy(execution, hostInteractions, interactions);
    Kokkos::deep_copy(execution, hostContinuations, continuations);
    Kokkos::deep_copy(execution, hostFallbacks, fallbacks);
    execution.fence("download Kokkos lepton vertex selection");
    result.batch_.interactions_.resize(totals[0]);
    result.batch_.continuations_.resize(totals[1]);
    result.batch_.fallbackEvents_.resize(totals[2]);
    for (std::size_t i = 0; i < totals[0]; ++i)
      result.batch_.interactions_[i] = hostInteractions(i);
    for (std::size_t i = 0; i < totals[1]; ++i)
      result.batch_.continuations_[i] = hostContinuations(i);
    for (std::size_t i = 0; i < totals[2]; ++i)
      result.batch_.fallbackEvents_[i] = hostFallbacks(i);
    for (std::size_t i = 0; i < count; ++i) {
      result.nativeNewtonIterations_ += hostOutcomes(i).nativeNewtonIterations_;
      result.nativeBisectionIterations_ += hostOutcomes(i).nativeBisectionIterations_;
      result.nativeInverseFailures_ += hostOutcomes(i).nativeInverseFailures_;
    }
    return result;
  }
} // namespace corsika::kokkos
