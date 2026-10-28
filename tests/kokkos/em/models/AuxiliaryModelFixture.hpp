/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/models/BremsLpm.hpp>
#include <corsika/kokkos/em/models/MoliereSnapshot.hpp>
#include <corsika/kokkos/em/models/PhotonPairLpm.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace corsika::kokkos::test {
  enum class AuxiliaryOperation : std::uint32_t {
    Photon,
    Electron,
    Muon,
    ElectronMoliere,
    MuonMoliere
  };

  struct AuxiliarySnapshots {
    PhotonPairLpmSnapshot photon_{};
    BremsLpmSnapshot electron_{};
    BremsLpmSnapshot muon_{};
    MoliereSnapshot electronMoliere_{};
    MoliereSnapshot muonMoliere_{};
  };

  struct AuxiliaryQuery {
    AuxiliaryOperation operation_{};
    std::uint32_t snapshotIndex_{};
    std::uint64_t componentHash_{};
    double energyMeV_{};
    double fraction_{};
    double densityGPerCm3_{};
    double nativeExpected_{};
    std::uint32_t nativeComparable_{};
    std::uint32_t branchMask_{};
  };

  struct AuxiliaryDataset {
    std::vector<AuxiliarySnapshots> snapshots_;
    std::vector<AuxiliaryQuery> queries_;
    std::uint32_t componentCount_{};
  };

  struct AuxiliaryAudit {
    std::vector<std::pair<std::string, bool>> checks_;
    std::size_t corruptionCases_{};
  };

  /** Use real PROPOSAL helpers; no stochastic interpolation tables are built. */
  AuxiliaryDataset make_auxiliary_dataset();

  /** Independently exercise live-helper export and the production v4 cache. */
  AuxiliaryAudit audit_auxiliary_models();
} // namespace corsika::kokkos::test
