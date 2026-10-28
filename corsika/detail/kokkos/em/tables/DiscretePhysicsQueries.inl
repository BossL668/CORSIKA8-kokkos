/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline TableQueryResult query_discrete_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV) {
    return table_query_detail::from_proposal_native(
        query_proposal_native_rate(view, pdg_id, process_id, component_hash, energy_MeV));
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_discrete_total_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV) {
    return table_query_detail::from_proposal_native(
        query_proposal_native_total_rate(view, pdg_id, energy_MeV));
  }

  CORSIKA_HOST_DEVICE inline RateColumnSelectionResult select_rate_column_by_uniform(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV,
      double uniform) {
    if (!table_query_detail::finite_value(uniform) || uniform < 0. || uniform >= 1.)
      return {TableLookupStatus::NonFiniteInput, 0, 0, 0., 0., 0};

    auto const total =
        query_proposal_native_selection_total_rate(view, pdg_id, energy_MeV);
    auto const converted_total = table_query_detail::from_proposal_native(total);
    if (converted_total.status_ != TableLookupStatus::Success)
      return {converted_total.status_, 0, 0, 0., 0., 0};
    if (!(converted_total.value_ > 0.))
      return {TableLookupStatus::Success, 0, 0, converted_total.value_, 0., 0};

    auto sampled_rate = uniform * converted_total.value_;
    auto const range = proposal_native_detail::dndx_particle_range(view, pdg_id);
    for (auto position = range.begin_; position < range.end_; ++position) {
      auto const* column = proposal_native_detail::selection_dndx(view, position);
      if (!column || column->pdgId_ != pdg_id)
        return {TableLookupStatus::InvalidTableView, 0, 0, converted_total.value_, 0., 0};
      NativeQueryResult native_rate{NativeQueryStatus::Success, 0, 0, 0.};
      if (!proposal_native_rate_below_threshold(*column, energy_MeV))
        native_rate = query_proposal_native_rate_for_column(view, *column, energy_MeV);
      auto const rate = table_query_detail::from_proposal_native(native_rate);
      if (rate.status_ != TableLookupStatus::Success)
        return {rate.status_,
                column->processId_,
                column->componentHash_,
                converted_total.value_,
                0.,
                0};
      sampled_rate -= rate.value_;
      if (sampled_rate < 0.) {
        auto const quantile = (-sampled_rate) / rate.value_;
        if (!table_query_detail::finite_value(quantile) || quantile < 0. || quantile > 1.)
          return {TableLookupStatus::InvalidTableView,
                  column->processId_,
                  column->componentHash_,
                  converted_total.value_,
                  0.,
                  0};
        return {TableLookupStatus::Success, column->processId_, column->componentHash_,
                converted_total.value_,     quantile,           1u};
      }
    }
    return {TableLookupStatus::Success, 0, 0, converted_total.value_, 0., 0};
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_discrete_loss_fraction(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV, double quantile) {
    if (process_id == PhotonMuonPairFinalStateProcessId && pdg_id == 22) {
      return table_query_detail::from_proposal_native(query_proposal_native_loss_fraction(
          view, pdg_id, PhotonMuonPairProcessId, component_hash, energy_MeV, quantile,
          true));
    }
    if (process_id == PhotonPairFinalStateProcessId && pdg_id == 22) {
      auto result =
          table_query_detail::from_proposal_native(query_proposal_native_loss_fraction(
              view, pdg_id, PhotonPairProcessId, component_hash, energy_MeV, quantile,
              true));
      if (result.status_ != TableLookupStatus::Success) return result;
      double normalized = 0.;
      if (!encode_photon_pair_normalized_split(energy_MeV / 1000., result.value_,
                                               normalized))
        return {TableLookupStatus::InverseCdfUnavailable, result.reserved_, 0.};
      result.value_ = normalized;
      return result;
    }
    return table_query_detail::from_proposal_native(query_proposal_native_loss_fraction(
        view, pdg_id, process_id, component_hash, energy_MeV, quantile));
  }

} // namespace corsika::kokkos
