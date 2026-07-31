/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <PROPOSAL/crosssection/CrossSection.h>
#include <PROPOSAL/propagation_utility/Interaction.h>

#include <corsika/modules/proposal/ProposalInteractionRecord.hpp>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace corsika::proposal {

  /**
   * Public, serialization-friendly view of one PROPOSAL process/component rate.
   *
   * rate uses PROPOSAL's native dN/dX convention. Unit conversion and table
   * serialization will be introduced by gpu_em_tablegen rather than hidden here.
   */
  struct ProposalRateEntry {
    PROPOSAL::InteractionType type{PROPOSAL::InteractionType::Undefined};
    std::size_t component_hash{};
    double rate{};
  };

  /**
   * Rates evaluated at one fixed projectile energy.
   *
   * nativeRates() is retained during the CPU split because PROPOSAL::SampleLoss needs
   * the cross-section object associated with every rate. GPU tables will consume
   * entries(), not the native pointers.
   */
  class ProposalRateTable {
  public:
    double energyMeV() const { return energy_MeV_; }
    std::size_t interactionHash() const { return interaction_hash_; }
    std::vector<ProposalRateEntry> const& entries() const { return entries_; }
    std::vector<PROPOSAL::Interaction::Rate> const& nativeRates() const {
      return native_rates_;
    }

    double totalRate() const {
      return std::accumulate(
          entries_.begin(), entries_.end(), 0.,
          [](double const sum, ProposalRateEntry const& entry) {
            return sum + entry.rate;
          });
    }

  private:
    friend class ProposalRateProvider;

    double energy_MeV_{};
    std::size_t interaction_hash_{};
    std::vector<ProposalRateEntry> entries_{};
    std::vector<PROPOSAL::Interaction::Rate> native_rates_{};
  };

  /**
   * Adapter separating process/component rates and loss selection from final-state
   * generation.
   */
  class ProposalRateProvider {
  public:
    ProposalRateTable rates(PROPOSAL::Interaction& interaction,
                            double const energy_MeV) const {
      if (!std::isfinite(energy_MeV) || energy_MeV <= 0.) {
        throw std::invalid_argument(
            "PROPOSAL rates require a finite positive projectile energy");
      }

      ProposalRateTable table;
      table.energy_MeV_ = energy_MeV;
      table.interaction_hash_ = interaction.GetHash();
      table.native_rates_ = interaction.Rates(energy_MeV);
      table.entries_.reserve(table.native_rates_.size());

      for (auto const& native : table.native_rates_) {
        if (!native.crosssection) {
          throw std::runtime_error("PROPOSAL returned a rate without a cross section");
        }
        if (!std::isfinite(native.rate) || native.rate < 0.) {
          throw std::runtime_error("PROPOSAL returned an invalid interaction rate");
        }
        table.entries_.push_back(ProposalRateEntry{
            native.crosssection->GetInteractionType(), native.comp_hash,
            native.rate});
      }
      if (!std::isfinite(table.totalRate())) {
        throw std::runtime_error(
            "PROPOSAL returned a non-finite total interaction rate");
      }
      return table;
    }

    ProposalInteractionRecord sample(
        PROPOSAL::Interaction& interaction, ProposalRateTable const& table,
        ProposalInteractionContext const& context, double const selection_uniform,
        std::optional<ProposalRandomKey> random_key = std::nullopt) const {
      validateContext(context);
      if (table.energyMeV() != context.projectile_energy_MeV) {
        throw std::invalid_argument(
            "PROPOSAL rate table and interaction context energies differ");
      }
      if (table.interactionHash() != interaction.GetHash()) {
        throw std::invalid_argument(
            "PROPOSAL rate table belongs to a different interaction calculator");
      }
      if (!std::isfinite(selection_uniform) || selection_uniform < 0. ||
          selection_uniform >= 1.) {
        throw std::invalid_argument(
            "PROPOSAL interaction selection random number must be in [0, 1)");
      }

      auto const loss = interaction.SampleLoss(
          context.projectile_energy_MeV, table.nativeRates(), selection_uniform);
      if (loss.type != PROPOSAL::InteractionType::Undefined &&
          (!std::isfinite(loss.v_loss) || loss.v_loss < 0. ||
           loss.v_loss > 1.)) {
        throw std::runtime_error("PROPOSAL sampled an invalid fractional loss");
      }

      return ProposalInteractionRecord{
          context, table.interactionHash(), loss.type, loss.comp_hash,
          loss.v_loss, selection_uniform, std::move(random_key)};
    }

    /**
     * Complete only the fractional loss for an already selected
     * process/component. This is the CPU continuation for a GPU inverse-CDF
     * capability miss and deliberately does not call Interaction::SampleLoss,
     * so the selected process and target cannot change.
     */
    double sampleSelectedLoss(
        ProposalRateTable const& table,
        PROPOSAL::InteractionType type,
        std::size_t component_hash,
        double loss_quantile) const {
      if (type == PROPOSAL::InteractionType::Undefined) {
        throw std::invalid_argument(
            "PROPOSAL selected-loss sampling requires a process type");
      }
      if (!std::isfinite(loss_quantile) ||
          loss_quantile < 0. || loss_quantile >= 1.) {
        throw std::invalid_argument(
            "PROPOSAL selected-loss quantile must be in [0, 1)");
      }
      for (auto const& native : table.nativeRates()) {
        if (!native.crosssection ||
            native.crosssection->GetInteractionType() != type ||
            native.comp_hash != component_hash) {
          continue;
        }
        if (!std::isfinite(native.rate) || !(native.rate > 0.)) {
          throw std::runtime_error(
              "PROPOSAL selected-loss column has a non-positive rate");
        }
        auto const loss =
            native.crosssection->CalculateStochasticLoss(
                component_hash, table.energyMeV(),
                loss_quantile * native.rate);
        if (!std::isfinite(loss) || loss < 0. || loss > 1.) {
          throw std::runtime_error(
              "PROPOSAL selected-loss sampler returned an invalid fraction");
        }
        return loss;
      }
      throw std::invalid_argument(
          "PROPOSAL selected-loss process/component is unavailable");
    }

  private:
    static void validateContext(ProposalInteractionContext const& context) {
      if (context.projectile_id == Code::Unknown) {
        throw std::invalid_argument(
            "PROPOSAL interaction context has an unknown projectile");
      }
      if (!std::isfinite(context.projectile_energy_MeV) ||
          context.projectile_energy_MeV <= 0. || !std::isfinite(context.time_s)) {
        throw std::invalid_argument(
            "PROPOSAL interaction context has invalid energy or time");
      }
      double direction_norm_squared = 0.;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(context.position_cm[axis]) ||
            !std::isfinite(context.direction[axis])) {
          throw std::invalid_argument(
              "PROPOSAL interaction context has non-finite geometry");
        }
        direction_norm_squared +=
            context.direction[axis] * context.direction[axis];
      }
      if (!(direction_norm_squared > 0.) ||
          !std::isfinite(direction_norm_squared)) {
        throw std::invalid_argument(
            "PROPOSAL interaction context has a zero direction");
      }
    }
  };

} // namespace corsika::proposal
