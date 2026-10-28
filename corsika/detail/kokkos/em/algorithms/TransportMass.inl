/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline double transport_mass_gev(std::int32_t const pid) {
    if (pid == 11 || pid == -11) return TransportElectronMassGeV;
    if (pid == 13 || pid == -13) return TransportMuonMassGeV;
    return 0.;
  }

  CORSIKA_HOST_DEVICE inline double proposal_secondary_transport_energy_gev(
      std::int32_t const pid, double const native_total_energy_GeV,
      double const native_mass_GeV) {
    return (native_total_energy_GeV - native_mass_GeV) + transport_mass_gev(pid);
  }

  CORSIKA_HOST_DEVICE inline double weighted_proposal_mass_correction_gev(
      std::int32_t const pid, double const native_mass_GeV, double const weight) {
    return weight * (transport_mass_gev(pid) - native_mass_GeV);
  }

} // namespace corsika::kokkos
