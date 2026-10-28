/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include "AuxiliaryModelConversion.hpp"

namespace corsika::kokkos::test {
  struct AuxiliaryOutcome {
    std::uint32_t status_{};
    std::uint32_t reserved_{};
    double value_{};
  };

  CORSIKA_HOST_DEVICE inline AuxiliaryOutcome evaluate_auxiliary(
      AuxiliarySnapshots const& snapshot, AuxiliaryQuery const& query) {
    if (query.operation_ == AuxiliaryOperation::Photon) {
      auto const value = photon_pair_lpm_suppression_factor(
          snapshot.photon_, query.componentHash_, query.energyMeV_, query.fraction_,
          query.densityGPerCm3_);
      return {static_cast<std::uint32_t>(value.status_), value.reserved_,
              value.survivalProbability_};
    }
    if (query.operation_ == AuxiliaryOperation::Electron ||
        query.operation_ == AuxiliaryOperation::Muon) {
      auto const& selected = query.operation_ == AuxiliaryOperation::Electron
                                 ? snapshot.electron_
                                 : snapshot.muon_;
      auto const value =
          brems_lpm_suppression_factor(selected, query.componentHash_, query.energyMeV_,
                                       query.fraction_, query.densityGPerCm3_);
      return {static_cast<std::uint32_t>(value.status_), value.reserved_,
              value.survivalProbability_};
    }
    auto const& selected = query.operation_ == AuxiliaryOperation::ElectronMoliere
                               ? snapshot.electronMoliere_
                               : snapshot.muonMoliere_;
    return {valid_moliere_snapshot(selected) ? 1u : 0u, 0u, 0.};
  }

  CORSIKA_HOST_DEVICE inline AuxiliaryOutcome reference_auxiliary(
      AuxiliaryReferenceSnapshots const& snapshot, AuxiliaryQuery const& query) {
    if (query.operation_ == AuxiliaryOperation::Photon) {
      auto const value = auxiliary_reference::photonPairLpmSuppressionFactor(
          snapshot.photon_, query.componentHash_, query.energyMeV_, query.fraction_,
          query.densityGPerCm3_);
      return {static_cast<std::uint32_t>(value.status), value.reserved,
              value.survival_probability};
    }
    if (query.operation_ == AuxiliaryOperation::Electron ||
        query.operation_ == AuxiliaryOperation::Muon) {
      auto const& selected = query.operation_ == AuxiliaryOperation::Electron
                                 ? snapshot.electron_
                                 : snapshot.muon_;
      auto const value = auxiliary_reference::bremsLpmSuppressionFactor(
          selected, query.componentHash_, query.energyMeV_, query.fraction_,
          query.densityGPerCm3_);
      return {static_cast<std::uint32_t>(value.status), value.reserved,
              value.survival_probability};
    }
    auto const& selected = query.operation_ == AuxiliaryOperation::ElectronMoliere
                               ? snapshot.electronMoliere_
                               : snapshot.muonMoliere_;
    return {auxiliary_reference::moliere_detail::validSnapshot(selected) ? 1u : 0u, 0u,
            0.};
  }
} // namespace corsika::kokkos::test
