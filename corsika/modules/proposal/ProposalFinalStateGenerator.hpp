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
    PROPOSAL::Component target_{};
    std::vector<PROPOSAL::ParticleState> secondaries_{};
  };

  /**
   * Generate secondaries for an already selected interaction record.
   *
   * This class deliberately has no process-selection API. It can only call
   * CalculateSecondaries() for the type/component/v stored in the record.
   */
  class ProposalFinalStateGenerator {
  public:
    std::size_t requiredRandomNumbers(PROPOSAL::SecondariesCalculator const& calculator,
                                      ProposalInteractionRecord const& record) const;

    ProposalFinalState generate(PROPOSAL::SecondariesCalculator& calculator,
                                ProposalInteractionRecord const& record,
                                std::vector<double> randomNumbers) const;

  private:
    static void requireSpecified(ProposalInteractionRecord const& record);
  };

} // namespace corsika::proposal

#include <corsika/detail/modules/proposal/ProposalFinalStateGenerator.inl>
