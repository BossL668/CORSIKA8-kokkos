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
   * rate uses PROPOSAL's native dN/dX convention. Unit conversion and table serialization
   * are handled by the export layer.
   */
  struct ProposalRateEntry {
    PROPOSAL::InteractionType type_{PROPOSAL::InteractionType::Undefined};
    std::size_t componentHash_{};
    double rate_{};
  };

  /**
   * Rates evaluated at one fixed projectile energy.
   *
   * nativeRates() is retained during the CPU split because PROPOSAL::SampleLoss needs
   * the cross-section object associated with every rate. Device table export consumes
   * entries(), not these native pointers.
   */
  class ProposalRateTable {
  public:
    double energyMeV() const;
    std::size_t interactionHash() const;
    std::vector<ProposalRateEntry> const& entries() const;
    std::vector<PROPOSAL::Interaction::Rate> const& nativeRates() const;

    double totalRate() const;

  private:
    friend class ProposalRateProvider;

    double energyMeV_{};
    std::size_t interactionHash_{};
    std::vector<ProposalRateEntry> entries_{};
    std::vector<PROPOSAL::Interaction::Rate> nativeRates_{};
  };

  /**
   * Adapter separating process/component rates and loss selection from final-state
   * generation.
   */
  class ProposalRateProvider {
  public:
    ProposalRateTable rates(PROPOSAL::Interaction& interaction,
                            double const energyMeV) const;

    ProposalInteractionRecord sample(
        PROPOSAL::Interaction& interaction, ProposalRateTable const& table,
        ProposalInteractionContext const& context, double const selectionUniform,
        std::optional<ProposalRandomKey> randomKey = std::nullopt) const;

    /**
     * Complete only the fractional loss for an already selected
     * process/component. This is the CPU continuation for a GPU inverse-CDF
     * capability miss and deliberately does not call Interaction::SampleLoss,
     * so the selected process and target cannot change.
     */
    double sampleSelectedLoss(ProposalRateTable const& table,
                              PROPOSAL::InteractionType type, std::size_t componentHash,
                              double lossQuantile) const;

  private:
    static void validateContext(ProposalInteractionContext const& context);
  };

} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/ProposalRateProvider.inl>
