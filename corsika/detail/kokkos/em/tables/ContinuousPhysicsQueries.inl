/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline ContinuousPhysicsView::ContinuousPhysicsView() {}

  CORSIKA_HOST_DEVICE inline ContinuousPhysicsView::ContinuousPhysicsView(
      ProposalNativeDeviceView const& nativeView, double emTransportCutMeV,
      double muonTransportCutMeV)
      : nativeView_(nativeView)
      , emTransportCutMeV_(emTransportCutMeV)
      , muonTransportCutMeV_(muonTransportCutMeV) {}

  CORSIKA_HOST_DEVICE inline ProposalNativeDeviceView const&
  ContinuousPhysicsView::getNativeView() const {
    return nativeView_;
  }

  CORSIKA_HOST_DEVICE inline double ContinuousPhysicsView::getEmTransportCutMeV() const {
    return emTransportCutMeV_;
  }

  CORSIKA_HOST_DEVICE inline double ContinuousPhysicsView::getMuonTransportCutMeV()
      const {
    return muonTransportCutMeV_;
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_dedx(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double energy_MeV) {
    return table_query_detail::from_proposal_native(
        query_proposal_native_dedx(view.getNativeView(), pdg_id, energy_MeV), true);
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_mass(
      ContinuousPhysicsView const& view, std::int32_t pdg_id) {
    auto const* utility =
        proposal_native_detail::find_utility(view.getNativeView(), pdg_id);
    if (!utility) return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    if (!table_query_detail::finite_value(utility->particleMassMeV_) ||
        !(utility->particleMassMeV_ > 0.))
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    return {TableLookupStatus::Success, 0, utility->particleMassMeV_};
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_transport_mass(
      ContinuousPhysicsView const& view, std::int32_t pdg_id) {
    auto const native_mass = query_continuous_mass(view, pdg_id);
    if (native_mass.status_ != TableLookupStatus::Success) return native_mass;
    auto const transport_mass_MeV = 1000. * transport_mass_gev(pdg_id);
    if (!(transport_mass_MeV > 0.))
      return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    return {TableLookupStatus::Success, 0, transport_mass_MeV};
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_transport_cut(
      ContinuousPhysicsView const& view, std::int32_t pdg_id) {
    auto const cut = (pdg_id == 13 || pdg_id == -13) ? view.getMuonTransportCutMeV()
                                                     : view.getEmTransportCutMeV();
    if (!table_query_detail::finite_value(cut) || !(cut > 0.))
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    return {TableLookupStatus::Success, 0, cut};
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_minimum_energy(
      ContinuousPhysicsView const& view, std::int32_t pdg_id) {
    auto const* utility =
        proposal_native_detail::find_utility(view.getNativeView(), pdg_id);
    if (!utility) return {TableLookupStatus::ContinuousParticleNotFound, 0, 0.};
    // PROPOSAL's native displacement interpolant starts at the particle
    // mass. The scalar ContinuousProcess applies its safety factor to the
    // TOTAL cut energy, not just the kinetic cut. This endpoint is only a
    // step limiter: callers must query the user cut separately rather than
    // invert this expression to decide ParticleCut membership.
    auto const transport_cut = query_continuous_transport_cut(view, pdg_id);
    if (transport_cut.status_ != TableLookupStatus::Success) return transport_cut;
    auto const minimum = (1000. * transport_mass_gev(pdg_id) + transport_cut.value_) *
                         ContinuousCutSafetyFactor;
    if (!table_query_detail::finite_value(utility->lowerEnergyLimitMeV_) ||
        utility->lowerEnergyLimitMeV_ < utility->particleMassMeV_ ||
        !table_query_detail::finite_value(minimum) ||
        !(minimum > utility->particleMassMeV_) ||
        minimum < utility->lowerEnergyLimitMeV_ || minimum > utility->spline_.axis_.high_)
      return {TableLookupStatus::InvalidTableView, 0, 0.};
    return {TableLookupStatus::Success, 0, minimum};
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_range(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double energy_MeV) {
    return table_query_detail::from_proposal_native(
        query_proposal_native_range(view.getNativeView(), pdg_id, energy_MeV), true);
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_energy(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double range_g_per_cm2) {
    auto result = table_query_detail::from_proposal_native(
        query_proposal_native_energy(view.getNativeView(), pdg_id, range_g_per_cm2),
        true);
    if (result.status_ == TableLookupStatus::ContinuousEnergyOutOfRange)
      result.status_ = TableLookupStatus::ContinuousRangeOutOfRange;
    return result;
  }

  CORSIKA_HOST_DEVICE inline TableQueryResult query_energy_after_continuous_loss(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double initial_energy_MeV,
      double grammage_g_per_cm2) {
    if (!table_query_detail::finite_value(grammage_g_per_cm2) ||
        grammage_g_per_cm2 < 0.) {
      return {TableLookupStatus::NonFiniteInput, 0, 0.};
    }
    auto const initial_range = query_continuous_range(view, pdg_id, initial_energy_MeV);
    if (initial_range.status_ != TableLookupStatus::Success) { return initial_range; }
    if (grammage_g_per_cm2 >= initial_range.value_ && grammage_g_per_cm2 != 0.) {
      return {TableLookupStatus::TransportCutReached, 0, 0.};
    }
    return table_query_detail::from_proposal_native(
        query_proposal_native_energy_after_continuous_loss(
            view.getNativeView(), pdg_id, initial_energy_MeV, grammage_g_per_cm2),
        true);
  }

} // namespace corsika::kokkos
