/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cstdint>

namespace corsika::kokkos {
  struct CorsikaOutputSinkStatistics {
    std::uint64_t steps_{};
    std::uint64_t depositedSteps_{};
    std::uint64_t radioTracks_{};
    std::uint64_t observations_{};
    std::uint64_t firstInteractions_{};
    double weightedDepositedEnergyGeV_{};
    double weightedMuonParentProductions_{};
  };
} // namespace corsika::kokkos
