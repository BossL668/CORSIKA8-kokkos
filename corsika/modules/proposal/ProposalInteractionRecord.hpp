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
   * Counter-based random-number address used by a device-selected CPU fallback.
   *
   * The current legacy CPU path uses the sequential "proposal" RNG stream and therefore
   * leaves this field empty. Device-selected interactions can attach the exact Philox
   * address without changing the physical interaction record.
   */
  struct ProposalRandomKey {
    std::uint64_t seed_{};
    std::uint64_t showerId_{};
    std::uint64_t historyId_{};
    std::uint64_t stepId_{};
    std::uint32_t processId_{};
    std::uint64_t drawId_{};
  };

  /**
   * Projectile state at the stochastic interaction vertex in PROPOSAL units.
   */
  struct ProposalInteractionContext {
    Code projectileId_{Code::Unknown};
    std::size_t mediumHash_{};
    double projectileEnergyMeV_{};
    std::array<double, 3> positionCm_{};
    std::array<double, 3> direction_{};
    double timeS_{};
  };

  /**
   * Fully specified stochastic interaction, but without generated secondaries.
   *
   * type_, componentHash_ and vLoss_ are sampled exactly once. A final-state generator
   * must consume this record and must not sample a different interaction.
   */
  struct ProposalInteractionRecord {
    ProposalInteractionContext context_{};
    std::size_t interactionHash_{};
    PROPOSAL::InteractionType type_{PROPOSAL::InteractionType::Undefined};
    std::size_t componentHash_{};
    double vLoss_{};
    double selectionUniform_{};
    std::optional<ProposalRandomKey> randomKey_{};
  };

} // namespace corsika::proposal
