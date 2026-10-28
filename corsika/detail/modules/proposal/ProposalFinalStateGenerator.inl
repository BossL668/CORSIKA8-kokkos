/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::proposal {
  inline std::size_t ProposalFinalStateGenerator::requiredRandomNumbers(
      PROPOSAL::SecondariesCalculator const& calculator,
      ProposalInteractionRecord const& record) const {
    requireSpecified(record);
    return calculator.RequiredRandomNumbers(record.type_);
  }

  inline ProposalFinalState ProposalFinalStateGenerator::generate(
      PROPOSAL::SecondariesCalculator& calculator,
      ProposalInteractionRecord const& record, std::vector<double> randomNumbers) const {
    requireSpecified(record);
    auto const required = calculator.RequiredRandomNumbers(record.type_);
    if (randomNumbers.size() != required) {
      throw std::invalid_argument(
          "Wrong number of random values for the specified PROPOSAL final state");
    }
    for (double const value : randomNumbers) {
      if (!std::isfinite(value) || value < 0. || value >= 1.) {
        throw std::invalid_argument(
            "PROPOSAL final-state random numbers must be in [0, 1)");
      }
    }

    auto const& context = record.context_;
    auto const point = PROPOSAL::Cartesian3D(
        context.positionCm_[0], context.positionCm_[1], context.positionCm_[2]);
    auto const direction = PROPOSAL::Cartesian3D(
        context.direction_[0], context.direction_[1], context.direction_[2]);
    auto const loss = PROPOSAL::StochasticLoss(
        static_cast<int>(record.type_), record.vLoss_ * context.projectileEnergyMeV_,
        point, direction, context.timeS_, 0., context.projectileEnergyMeV_);

    PROPOSAL::Component target;
    if (record.type_ != PROPOSAL::InteractionType::Ioniz) {
      target = PROPOSAL::Component::GetComponentForHash(record.componentHash_);
    }

    auto secondaries = calculator.CalculateSecondaries(loss, target, randomNumbers);
    return ProposalFinalState{std::move(target), std::move(secondaries)};
  }

  inline void ProposalFinalStateGenerator::requireSpecified(
      ProposalInteractionRecord const& record) {
    if (record.type_ == PROPOSAL::InteractionType::Undefined) {
      throw std::invalid_argument(
          "Cannot generate a final state for an undefined PROPOSAL interaction");
    }
    if (!std::isfinite(record.vLoss_) || record.vLoss_ < 0. || record.vLoss_ > 1. ||
        !std::isfinite(record.context_.projectileEnergyMeV_) ||
        record.context_.projectileEnergyMeV_ <= 0.) {
      throw std::invalid_argument("Specified PROPOSAL interaction record is invalid");
    }
    if (!std::isfinite(record.context_.timeS_)) {
      throw std::invalid_argument(
          "Specified PROPOSAL interaction record has an invalid time");
    }
    double directionNormSquared = 0.;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(record.context_.positionCm_[axis]) ||
          !std::isfinite(record.context_.direction_[axis])) {
        throw std::invalid_argument(
            "Specified PROPOSAL interaction record has invalid geometry");
      }
      directionNormSquared +=
          record.context_.direction_[axis] * record.context_.direction_[axis];
    }
    if (!(directionNormSquared > 0.) || !std::isfinite(directionNormSquared)) {
      throw std::invalid_argument(
          "Specified PROPOSAL interaction record has a zero direction");
    }
  }
} // namespace corsika::proposal
