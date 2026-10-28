/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::proposal {
  inline double ProposalRateTable::energyMeV() const { return energyMeV_; }

  inline std::size_t ProposalRateTable::interactionHash() const {
    return interactionHash_;
  }

  inline std::vector<ProposalRateEntry> const& ProposalRateTable::entries() const {
    return entries_;
  }

  inline std::vector<PROPOSAL::Interaction::Rate> const& ProposalRateTable::nativeRates()
      const {
    return nativeRates_;
  }

  inline double ProposalRateTable::totalRate() const {
    return std::accumulate(entries_.begin(), entries_.end(), 0.,
                           [](double const sum, ProposalRateEntry const& entry) {
                             return sum + entry.rate_;
                           });
  }

  inline ProposalRateTable ProposalRateProvider::rates(PROPOSAL::Interaction& interaction,
                                                       double const energyMeV) const {
    if (!std::isfinite(energyMeV) || energyMeV <= 0.) {
      throw std::invalid_argument(
          "PROPOSAL rates require a finite positive projectile energy");
    }

    ProposalRateTable table;
    table.energyMeV_ = energyMeV;
    table.interactionHash_ = interaction.GetHash();
    table.nativeRates_ = interaction.Rates(energyMeV);
    table.entries_.reserve(table.nativeRates_.size());

    for (auto const& native : table.nativeRates_) {
      if (!native.crosssection) {
        throw std::runtime_error("PROPOSAL returned a rate without a cross section");
      }
      if (!std::isfinite(native.rate) || native.rate < 0.) {
        throw std::runtime_error("PROPOSAL returned an invalid interaction rate");
      }
      table.entries_.push_back(ProposalRateEntry{
          native.crosssection->GetInteractionType(), native.comp_hash, native.rate});
    }
    if (!std::isfinite(table.totalRate())) {
      throw std::runtime_error("PROPOSAL returned a non-finite total interaction rate");
    }
    return table;
  }

  inline ProposalInteractionRecord ProposalRateProvider::sample(
      PROPOSAL::Interaction& interaction, ProposalRateTable const& table,
      ProposalInteractionContext const& context, double const selectionUniform,
      std::optional<ProposalRandomKey> randomKey) const {
    validateContext(context);
    if (table.energyMeV() != context.projectileEnergyMeV_) {
      throw std::invalid_argument(
          "PROPOSAL rate table and interaction context energies differ");
    }
    if (table.interactionHash() != interaction.GetHash()) {
      throw std::invalid_argument(
          "PROPOSAL rate table belongs to a different interaction calculator");
    }
    if (!std::isfinite(selectionUniform) || selectionUniform < 0. ||
        selectionUniform >= 1.) {
      throw std::invalid_argument(
          "PROPOSAL interaction selection random number must be in [0, 1)");
    }

    auto const loss = interaction.SampleLoss(context.projectileEnergyMeV_,
                                             table.nativeRates(), selectionUniform);
    if (loss.type != PROPOSAL::InteractionType::Undefined &&
        (!std::isfinite(loss.v_loss) || loss.v_loss < 0. || loss.v_loss > 1.)) {
      throw std::runtime_error("PROPOSAL sampled an invalid fractional loss");
    }

    return ProposalInteractionRecord{
        context,     table.interactionHash(), loss.type,           loss.comp_hash,
        loss.v_loss, selectionUniform,        std::move(randomKey)};
  }

  inline double ProposalRateProvider::sampleSelectedLoss(ProposalRateTable const& table,
                                                         PROPOSAL::InteractionType type,
                                                         std::size_t componentHash,
                                                         double lossQuantile) const {
    if (type == PROPOSAL::InteractionType::Undefined) {
      throw std::invalid_argument(
          "PROPOSAL selected-loss sampling requires a process type");
    }
    if (!std::isfinite(lossQuantile) || lossQuantile < 0. || lossQuantile >= 1.) {
      throw std::invalid_argument("PROPOSAL selected-loss quantile must be in [0, 1)");
    }
    for (auto const& native : table.nativeRates()) {
      if (!native.crosssection || native.crosssection->GetInteractionType() != type ||
          native.comp_hash != componentHash) {
        continue;
      }
      if (!std::isfinite(native.rate) || !(native.rate > 0.)) {
        throw std::runtime_error("PROPOSAL selected-loss column has a non-positive rate");
      }
      auto const loss = native.crosssection->CalculateStochasticLoss(
          componentHash, table.energyMeV(), lossQuantile * native.rate);
      if (!std::isfinite(loss) || loss < 0. || loss > 1.) {
        throw std::runtime_error(
            "PROPOSAL selected-loss sampler returned an invalid fraction");
      }
      return loss;
    }
    throw std::invalid_argument(
        "PROPOSAL selected-loss process/component is unavailable");
  }

  inline void ProposalRateProvider::validateContext(
      ProposalInteractionContext const& context) {
    if (context.projectileId_ == Code::Unknown) {
      throw std::invalid_argument(
          "PROPOSAL interaction context has an unknown projectile");
    }
    if (!std::isfinite(context.projectileEnergyMeV_) ||
        context.projectileEnergyMeV_ <= 0. || !std::isfinite(context.timeS_)) {
      throw std::invalid_argument(
          "PROPOSAL interaction context has invalid energy or time");
    }
    double directionNormSquared = 0.;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(context.positionCm_[axis]) ||
          !std::isfinite(context.direction_[axis])) {
        throw std::invalid_argument(
            "PROPOSAL interaction context has non-finite geometry");
      }
      directionNormSquared += context.direction_[axis] * context.direction_[axis];
    }
    if (!(directionNormSquared > 0.) || !std::isfinite(directionNormSquared)) {
      throw std::invalid_argument("PROPOSAL interaction context has a zero direction");
    }
  }
} // namespace corsika::proposal
