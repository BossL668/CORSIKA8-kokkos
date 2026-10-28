/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::kokkos {
  /** Decoded production histograms and weighted energy ledger in GeV. */
  struct ProfileResult {
    std::vector<double> photons_{};
    std::vector<double> electrons_{};
    std::vector<double> positrons_{};
    std::vector<double> muonsMinus_{};
    std::vector<double> muonsPlus_{};
    /**
     * Muons produced by a muon parent at accepted device-resident discrete
     * interaction vertices.  This is the accelerator counterpart of
     * ProductionProfile::doSecondaries(), not a transported-particle count.
     */
    std::vector<double> muonParentProductions_{};
    std::vector<double> energyLossGeV_{};
    std::vector<double> muonEnergyLossGeV_{};
    std::uint64_t steps_{};
    std::uint64_t depositedSteps_{};
    std::uint64_t particleCuts_{};
    std::uint64_t fixedPointOverflows_{};
    std::uint64_t invalidRecords_{};
    double weightedDepositedEnergyGeV_{};
    /**
     * Deterministic energy-ledger terms accumulated from the same resident
     * transport/final-state records as the longitudinal profile.
     *
     * Total particle energies include rest mass.  Atomic-electron processes
     * (Compton, photoelectric, ionization and positron annihilation) therefore
     * import one electron rest mass from the medium.  ParticleCut deposits only
     * kinetic energy, so a terminated electron/positron rest mass is retained as
     * an explicit terminal sink instead of being silently hidden.
     */
    double weightedMediumRestMassInputGeV_{};
    double weightedCutRestMassEnergyGeV_{};
    double weightedObservedTotalEnergyGeV_{};
    double weightedEscapedTotalEnergyGeV_{};
    // PROPOSAL's scalar writer does not emit the photoelectric binding loss.
    // Keep it outside legacy dE/dX, but explicitly account for it in closure.
    double weightedUnwrittenPhotoelectricBindingEnergyGeV_{};
    // ObservationPlane precedes ParticleCut; their terminal records can overlap.
    double weightedObservationCutOverlapEnergyGeV_{};
    // Signed numerical convention change from PROPOSAL total energy to the
    // CORSIKA kinetic-energy stack; not a physical interaction with the medium.
    double weightedMassConventionCorrectionGeV_{};
    // Signed ParticleCut bookkeeping for tagged sub-rest-mass BKK children.
    double weightedNativeMuonPairCutEnergyGeV_{};
    std::uint64_t leptonMaterialBoundaries_{};
  };
} // namespace corsika::kokkos
