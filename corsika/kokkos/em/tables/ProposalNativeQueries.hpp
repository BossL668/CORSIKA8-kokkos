/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/tables/ProposalNativeData.hpp>

#include <corsika/framework/core/HostDevice.hpp>

/**
 * \file
 * Numerical queries consume one validated format-v7 export in accessible memory.
 * Energies and masses are in MeV; grammage and range are in g/cm^2, rates in
 * (g/cm^2)^-1, and stopping power in MeV/(g/cm^2). The caller retains storage.
 * Each entry point preserves its own production validation order and returns
 * its status, value and iteration counts; they are not interchangeable guards.
 * No PROPOSAL object, random draw, new fit or candidate solver is used here.
 */

namespace corsika::kokkos {
  /** Test the original below-threshold predicate for a bicubic rate column. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool proposal_native_rate_below_threshold(
      NativeDndxColumn const& column, double energy_MeV);

  /**
   * Evaluate the cumulative spline divided by componentWeight_, with native clipping.
   * transformed_loss is the spline's transformed coordinate, not the physical loss.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_cumulative_rate(ProposalNativeDeviceView const& view,
                                        NativeDndxColumn const& column, double energy_MeV,
                                        double transformed_loss);

  /** Evaluate one exported rate column, including its original direct-rate model. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_rate_for_column(ProposalNativeDeviceView const& view,
                                        NativeDndxColumn const& column_value,
                                        double energy_MeV);

  /** Find the canonical PID/process/component column and evaluate its rate. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_rate(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, std::int32_t process_id,
      std::uint64_t component_hash, double energy_MeV);

  /** Evaluate the independent total-rate spline used by interaction-distance sampling. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_total_rate(ProposalNativeDeviceView const& view,
                                   std::int32_t pdg_id, double energy_MeV);

  /** Sum rates in original Interaction::Rates order for process/component selection. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_selection_total_rate(ProposalNativeDeviceView const& view,
                                             std::int32_t pdg_id, double energy_MeV);

  /** Sum continuous stopping-power columns in the exported PROPOSAL accumulation order.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_dedx(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV);

  /**
   * Invert the raw CDF with the original six-point derivative and Newton/bisection.
   * quantile is in [0, 1]. Normally value_ is the stochastic energy-loss fraction.
   * With photon_split, it is the electron fraction for photon pair production,
   * or the positive-muon fraction for BKK muon-pair production. These explicit
   * secondary queries reuse the original rate column, without adding a process.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_loss_fraction(ProposalNativeDeviceView const& view,
                                      std::int32_t pdg_id, std::int32_t process_id,
                                      std::uint64_t component_hash, double energy_MeV,
                                      double quantile, bool photon_split = false);

  /** Evaluate the absolute range relative to the particle's lower-energy limit. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_range(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double energy_MeV);

  /** Invert continuous loss over [lower energy, initial energy], retaining the original
   * fallback. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult
  query_proposal_native_energy_after_continuous_loss(ProposalNativeDeviceView const& view,
                                                     std::int32_t pdg_id,
                                                     double initial_energy_MeV,
                                                     double grammage_g_per_cm2);

  /** Invert the absolute range over the full table using the original logarithmic-axis
   * bisection. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline NativeQueryResult query_proposal_native_energy(
      ProposalNativeDeviceView const& view, std::int32_t pdg_id, double range_g_per_cm2);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/tables/ProposalNativeQueries.inl>
