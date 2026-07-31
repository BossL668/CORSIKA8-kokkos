/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <PROPOSAL/particle/Particle.h>

#include <corsika/framework/core/ParticleProperties.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace corsika::proposal {

  /**
   * Counter-based random-number address reserved for a future GPU/CPU fallback.
   *
   * The current legacy CPU path uses the sequential "proposal" RNG stream and therefore
   * leaves this field empty. Device-selected interactions can attach the exact Philox
   * address without changing the physical interaction record.
   */
  struct ProposalRandomKey {
    std::uint64_t seed{};
    std::uint64_t shower_id{};
    std::uint64_t history_id{};
    std::uint64_t step_id{};
    std::uint32_t process_id{};
    std::uint64_t draw_id{};
  };

  /**
   * Projectile state at the stochastic interaction vertex in PROPOSAL units.
   */
  struct ProposalInteractionContext {
    Code projectile_id{Code::Unknown};
    std::size_t medium_hash{};
    double projectile_energy_MeV{};
    std::array<double, 3> position_cm{};
    std::array<double, 3> direction{};
    double time_s{};
  };

  /**
   * Fully specified stochastic interaction, but without generated secondaries.
   *
   * type, component_hash and v_loss are sampled exactly once. A final-state generator
   * must consume this record and must not sample a different interaction.
   */
  struct ProposalInteractionRecord {
    ProposalInteractionContext context{};
    std::size_t interaction_hash{};
    PROPOSAL::InteractionType type{PROPOSAL::InteractionType::Undefined};
    std::size_t component_hash{};
    double v_loss{};
    double selection_uniform{};
    std::optional<ProposalRandomKey> random_key{};
  };

} // namespace corsika::proposal
