/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {

  /**
   * Value-only transport segment for host/device exchange and output adapters.
   *
   * Endpoint positions use the same common coordinate system as ParticleState.
   * Energies are in GeV, positions in metres and times in seconds. Deposits are
   * unweighted; the output adapter applies weight_ once. Copying a record neither
   * deposits energy nor allocates a particle/step identifier.
   *
   * Public members preserve aggregate access. This standard-layout, trivially
   * copyable type has no explicit alignment requirement beyond its members and is
   * not a portable wire format. No Kokkos or physics-library header is required.
   */
  struct StepRecord {
    std::uint64_t historyId_{};
    std::uint64_t stepId_{};
    std::int32_t pid_{};       ///< Signed PDG identifier.
    std::int32_t processId_{}; ///< Producer's process identifier; zero if unselected.
    double startPositionM_[3]{};
    double endPositionM_[3]{};
    double startTimeS_{};
    double endTimeS_{};
    double startEnergyGeV_{}; ///< Total energy, including rest mass.
    double endEnergyGeV_{};   ///< Total energy, including rest mass.
    double depositedEnergyGeV_{};
    double weight_{}; ///< Preserved zero default; unlike ParticleState's unit weight.
    /** Already included in depositedEnergyGeV_; written at the endpoint instead
     * of spread along the preceding segment. The existing tagged native muon-pair
     * cut can carry a signed kinetic-energy residual in a zero-length record.
     * Neither deposit field therefore imposes a nonnegative-value constraint.
     */
    double cutDepositedEnergyGeV_{};
    std::uint32_t nativeMuonPairCut_{}; ///< Preserve the producer's compatibility tag.
  };

} // namespace corsika::kokkos
