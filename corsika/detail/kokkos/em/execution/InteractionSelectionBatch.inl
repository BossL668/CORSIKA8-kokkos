/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class TExecutionSpace>
  InteractionSelectionBatchResult select_interactions_batch(
      InteractionPhysicsView const physics, std::vector<ParticleState> const& particles,
      std::uint64_t const randomSeed, std::uint64_t const showerId,
      TExecutionSpace const& execution) {
    using memory_space_type = typename TExecutionSpace::memory_space;
    using policy_type = Kokkos::RangePolicy<TExecutionSpace>;
    using outcome_type = InteractionSelectionOutcome;
    InteractionSelectionBatchResult result{};
    result.batch_.inputParticles_ = particles.size();
    if (particles.empty()) return result;
    if (particles.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error(
          "Kokkos interaction selector exceeds 32-bit stable offsets");
    }

    Kokkos::View<ParticleState*, memory_space_type> deviceParticles(
        "c8_kokkos_selection_particles", particles.size());
    Kokkos::View<outcome_type*, memory_space_type> outcomes(
        "c8_kokkos_selection_outcomes", particles.size());
    Kokkos::View<std::uint32_t*, memory_space_type> flags(
        "c8_kokkos_selection_fallback_flags", particles.size());
    Kokkos::View<std::uint32_t*, memory_space_type> offsets(
        "c8_kokkos_selection_fallback_offsets", particles.size());
    Kokkos::View<InteractionRecord*, memory_space_type> interactions(
        "c8_kokkos_selected_interactions", particles.size());
    Kokkos::View<FallbackRequest*, memory_space_type> fallbacks(
        "c8_kokkos_selection_fallbacks", particles.size());

    auto hostParticles = Kokkos::create_mirror_view(deviceParticles);
    for (std::size_t i = 0; i < particles.size(); ++i) {
      hostParticles(i) = particles[i];
    }
    Kokkos::deep_copy(execution, deviceParticles, hostParticles);
    Kokkos::parallel_for(
        "c8_kokkos_select_interactions", policy_type(execution, 0, particles.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const outcome = select_discrete_interaction(
              physics, deviceParticles(index), index, randomSeed, showerId);
          outcomes(index) = outcome;
          flags(index) = outcome.fallbackFlag_;
        });

    std::uint32_t fallbackCount{};
    Kokkos::parallel_scan(
        "c8_kokkos_scan_interaction_fallbacks",
        policy_type(execution, 0, particles.size()),
        KOKKOS_LAMBDA(std::size_t const index, std::uint32_t& update, bool const final) {
          if (final) offsets(index) = update;
          update += flags(index);
        },
        fallbackCount);
    Kokkos::parallel_for(
        "c8_kokkos_compact_interaction_selection",
        policy_type(execution, 0, particles.size()),
        KOKKOS_LAMBDA(std::size_t const index) {
          auto const fallbackOffset = offsets(index);
          if (flags(index) != 0u) {
            fallbacks(fallbackOffset) = outcomes(index).fallback_;
          } else {
            interactions(index - fallbackOffset) = outcomes(index).interaction_;
          }
        });
    execution.fence("complete Kokkos interaction selection");

    auto hostInteractions = Kokkos::create_mirror_view(interactions);
    auto hostFallbacks = Kokkos::create_mirror_view(fallbacks);
    auto hostOutcomes = Kokkos::create_mirror_view(outcomes);
    Kokkos::deep_copy(execution, hostInteractions, interactions);
    Kokkos::deep_copy(execution, hostFallbacks, fallbacks);
    Kokkos::deep_copy(execution, hostOutcomes, outcomes);
    execution.fence("download Kokkos interaction selection");

    auto const interactionCount = particles.size() - fallbackCount;
    result.batch_.interactions_.resize(interactionCount);
    result.batch_.fallbackEvents_.resize(fallbackCount);
    for (std::size_t i = 0; i < interactionCount; ++i) {
      result.batch_.interactions_[i] = hostInteractions(i);
    }
    for (std::size_t i = 0; i < fallbackCount; ++i) {
      result.batch_.fallbackEvents_[i] = hostFallbacks(i);
    }
    for (std::size_t i = 0; i < particles.size(); ++i) {
      result.nativeNewtonIterations_ += hostOutcomes(i).nativeNewtonIterations_;
      result.nativeBisectionIterations_ += hostOutcomes(i).nativeBisectionIterations_;
      result.nativeInverseFailures_ += hostOutcomes(i).nativeInverseFailures_;
    }
    return result;
  }
} // namespace corsika::kokkos
