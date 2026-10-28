/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <corsika/kokkos/em/algorithms/TransportMass.hpp>
#include <corsika/kokkos/em/tables/TableQueryResult.hpp>

#include <cstdint>
#include <type_traits>

/**
 * \file
 * Continuous-loss queries over the original format-v7 PROPOSAL tables.
 * Energies, masses and kinetic transport cuts are in MeV; range and grammage
 * are in g/cm^2, and stopping power is in MeV/(g/cm^2). Each query retains
 * its production validation order, endpoint rules and failure value.
 * This layer does not export tables, allocate memory or select a new solver.
 */

namespace corsika::kokkos {

  /** Safety factor applied to the total cut energy by the continuous step limiter. */
  inline constexpr double ContinuousCutSafetyFactor = 0.9999;

  /**
   * Borrowed native tables and the two kinetic transport cuts used by these queries.
   * Construction copies the view and cuts without validation or normalization.
   * Referenced arrays must remain alive and accessible to the querying execution
   * space until all queries finish; copying this object does not own their storage.
   * Unused stochastic-query state from the production aggregate is not included.
   */
  class ContinuousPhysicsView {
  public:
    CORSIKA_HOST_DEVICE ContinuousPhysicsView();
    CORSIKA_HOST_DEVICE ContinuousPhysicsView(ProposalNativeDeviceView const& nativeView,
                                              double emTransportCutMeV,
                                              double muonTransportCutMeV);

    [[nodiscard]] CORSIKA_HOST_DEVICE ProposalNativeDeviceView const& getNativeView()
        const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getEmTransportCutMeV() const;
    [[nodiscard]] CORSIKA_HOST_DEVICE double getMuonTransportCutMeV() const;

  private:
    ProposalNativeDeviceView nativeView_{};
    double emTransportCutMeV_{};
    double muonTransportCutMeV_{};
  };

  static_assert(std::is_trivially_copyable_v<ContinuousPhysicsView>);

  /** Sum native continuous stopping powers in their original accumulation order. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_dedx(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double energy_MeV);

  /** Return the positive finite mass stored in the particle's native utility column. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_mass(
      ContinuousPhysicsView const& view, std::int32_t pdg_id);

  /** Check native mass availability, then return the C8 transport mass in MeV. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult
  query_continuous_transport_mass(ContinuousPhysicsView const& view, std::int32_t pdg_id);

  /** Select the positive finite muon or electromagnetic kinetic transport cut. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult
  query_continuous_transport_cut(ContinuousPhysicsView const& view, std::int32_t pdg_id);

  /**
   * Return the step-limiting total energy (transport mass + kinetic cut) times
   * ContinuousCutSafetyFactor, subject to the original native-table endpoint checks.
   * This is not the user cut: callers must query the transport cut separately.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult
  query_continuous_minimum_energy(ContinuousPhysicsView const& view, std::int32_t pdg_id);

  /** Return the native range relative to the particle's lower-energy limit. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_range(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double energy_MeV);

  /** Invert native range, retaining the continuous-range failure-status mapping. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult query_continuous_energy(
      ContinuousPhysicsView const& view, std::int32_t pdg_id, double range_g_per_cm2);

  /**
   * Invert continuous loss after checking grammage and initial range in that order.
   * A nonzero grammage at or beyond the initial range returns TransportCutReached;
   * otherwise the original native inverse and its iteration counts are retained.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline TableQueryResult
  query_energy_after_continuous_loss(ContinuousPhysicsView const& view,
                                     std::int32_t pdg_id, double initial_energy_MeV,
                                     double grammage_g_per_cm2);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/tables/ContinuousPhysicsQueries.inl>
