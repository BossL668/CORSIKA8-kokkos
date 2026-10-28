/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <corsika/framework/core/ParticleProperties.hpp>
#endif
#include <cstdint>

namespace corsika::kokkos {

  // Transport, ParticleCut and the CORSIKA stack use the generated CORSIKA
  // mass table. PROPOSAL's own constants differ slightly and must remain in
  // its cross sections, final-state parametrizations and scattering models.
  // The generated scalar header is host-only (and cannot be included in a
  // CUDA/HIP translation unit). These POD values mirror its seven-significant-
  // digit mass data and are checked in every ordinary host translation unit.
  // Changing CORSIKA's generated mass data therefore requires updating these
  // constants explicitly; never replace them with the PROPOSAL constants.
  inline constexpr double TransportElectronMassGeV = 5.109989e-4;
  inline constexpr double TransportMuonMassGeV = 1.056584e-1;
#if !defined(__CUDACC__) && !defined(__HIPCC__)
  static_assert(get_mass(Code::Electron) / 1_GeV == TransportElectronMassGeV,
                "update device transport electron mass to generated CORSIKA data");
  static_assert(get_mass(Code::MuMinus) / 1_GeV == TransportMuonMassGeV,
                "update device transport muon mass to generated CORSIKA data");
#endif

  /** C8 transport mass in GeV for signed lepton PDG identifiers; zero otherwise.
   * Keep PROPOSAL's separate model/calculator masses in its physical formulas. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline double transport_mass_gev(
      std::int32_t const pid);

  /** Match the native secondary-to-stack adapter: preserve kinetic energy in GeV.
   * native_mass_GeV is the PROPOSAL model mass; transport uses C8's mass table.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline double proposal_secondary_transport_energy_gev(
      std::int32_t const pid, double const native_total_energy_GeV,
      double const native_mass_GeV);

  /** Signed weighted mass-convention correction in GeV, not an energy deposit.
   * Compute from masses directly to retain the correction at high energy.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline double weighted_proposal_mass_correction_gev(
      std::int32_t const pid, double const native_mass_GeV, double const weight);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/TransportMass.inl>
