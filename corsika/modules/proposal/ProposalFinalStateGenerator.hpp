/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <PROPOSAL/PROPOSAL.h>

#include <corsika/modules/proposal/ProposalInteractionRecord.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace corsika::proposal {

  struct ProposalFinalState {
    PROPOSAL::Component target{};
    std::vector<PROPOSAL::ParticleState> secondaries{};
  };

  /**
   * Generate secondaries for an already selected interaction record.
   *
   * This class deliberately has no process-selection API. It can only call
   * CalculateSecondaries() for the type/component/v stored in the record.
   */
  class ProposalFinalStateGenerator {
  public:
    std::size_t requiredRandomNumbers(
        PROPOSAL::SecondariesCalculator const& calculator,
        ProposalInteractionRecord const& record) const {
      requireSpecified(record);
      return calculator.RequiredRandomNumbers(record.type);
    }

    ProposalFinalState generate(
        PROPOSAL::SecondariesCalculator& calculator,
        ProposalInteractionRecord const& record,
        std::vector<double> random_numbers) const {
      requireSpecified(record);
      auto const required = calculator.RequiredRandomNumbers(record.type);
      if (random_numbers.size() != required) {
        throw std::invalid_argument(
            "Wrong number of random values for the specified PROPOSAL final state");
      }
      for (double const value : random_numbers) {
        if (!std::isfinite(value) || value < 0. || value >= 1.) {
          throw std::invalid_argument(
              "PROPOSAL final-state random numbers must be in [0, 1)");
        }
      }

      auto const& context = record.context;
      auto const point =
          PROPOSAL::Cartesian3D(context.position_cm[0], context.position_cm[1],
                                context.position_cm[2]);
      auto const direction =
          PROPOSAL::Cartesian3D(context.direction[0], context.direction[1],
                                context.direction[2]);
      auto const loss = PROPOSAL::StochasticLoss(
          static_cast<int>(record.type),
          record.v_loss * context.projectile_energy_MeV, point, direction,
          context.time_s, 0., context.projectile_energy_MeV);

      PROPOSAL::Component target;
      if (record.type != PROPOSAL::InteractionType::Ioniz) {
        target =
            PROPOSAL::Component::GetComponentForHash(record.component_hash);
      }

      auto secondaries =
          calculator.CalculateSecondaries(loss, target, random_numbers);
      return ProposalFinalState{std::move(target), std::move(secondaries)};
    }

  private:
    static void requireSpecified(ProposalInteractionRecord const& record) {
      if (record.type == PROPOSAL::InteractionType::Undefined) {
        throw std::invalid_argument(
            "Cannot generate a final state for an undefined PROPOSAL interaction");
      }
      if (!std::isfinite(record.v_loss) || record.v_loss < 0. ||
          record.v_loss > 1. ||
          !std::isfinite(record.context.projectile_energy_MeV) ||
          record.context.projectile_energy_MeV <= 0.) {
        throw std::invalid_argument(
            "Specified PROPOSAL interaction record is invalid");
      }
      if (!std::isfinite(record.context.time_s)) {
        throw std::invalid_argument(
            "Specified PROPOSAL interaction record has an invalid time");
      }
      double direction_norm_squared = 0.;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(record.context.position_cm[axis]) ||
            !std::isfinite(record.context.direction[axis])) {
          throw std::invalid_argument(
              "Specified PROPOSAL interaction record has invalid geometry");
        }
        direction_norm_squared +=
            record.context.direction[axis] * record.context.direction[axis];
      }
      if (!(direction_norm_squared > 0.) ||
          !std::isfinite(direction_norm_squared)) {
        throw std::invalid_argument(
            "Specified PROPOSAL interaction record has a zero direction");
      }
    }
  };

} // namespace corsika::proposal
