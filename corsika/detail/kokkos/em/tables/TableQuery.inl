/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline TableQueryResult execute_table_query(
      ContinuousPhysicsView const& view, TableQuery const& query) {
    if (query.kind_ == TableQueryKind::Rate) {
      return query_discrete_rate(view.getNativeView(), query.pdgId_, query.processId_,
                                 query.componentHash_, query.energyMeV_);
    }
    if (query.kind_ == TableQueryKind::TotalRate) {
      return query_discrete_total_rate(view.getNativeView(), query.pdgId_,
                                       query.energyMeV_);
    }
    if (query.kind_ == TableQueryKind::LossFraction) {
      return query_discrete_loss_fraction(view.getNativeView(), query.pdgId_,
                                          query.processId_, query.componentHash_,
                                          query.energyMeV_, query.quantile_);
    }
    if (query.kind_ == TableQueryKind::ContinuousDedx) {
      return query_continuous_dedx(view, query.pdgId_, query.energyMeV_);
    }
    if (query.kind_ == TableQueryKind::ContinuousRange) {
      return query_continuous_range(view, query.pdgId_, query.energyMeV_);
    }
    if (query.kind_ == TableQueryKind::ContinuousEnergy) {
      return query_continuous_energy(view, query.pdgId_, query.energyMeV_);
    }
    if (query.kind_ == TableQueryKind::EnergyAfterContinuousLoss) {
      return query_energy_after_continuous_loss(view, query.pdgId_, query.energyMeV_,
                                                query.quantile_);
    }
    return {TableLookupStatus::InvalidQueryKind, 0, 0.};
  }

} // namespace corsika::kokkos
