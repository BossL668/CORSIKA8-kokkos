/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>

namespace corsika::kokkos {

  // PROPOSAL 7.6.2 ME converted from MeV to GeV.
  inline constexpr double ElectronMassGeV = 0.0005109989461;
  inline constexpr double ElectronMassMeV = 1000. * ElectronMassGeV;
  inline constexpr double PhotonPairThresholdMeV = 2. * ElectronMassMeV;

  namespace photon_pair_kinematics_detail {
    CORSIKA_HOST_DEVICE inline bool finite(double value);
  }

  /** Decode the production normalized pair coordinate to electron energy / E_gamma.
   * Energy is GeV. Keep the original endpoint tolerance and failure output.
   * The normalized coordinate is s = (rho - m/E) / (1 - 2m/E), where rho
   * is the electron energy fraction. This avoids storing a collapsing
   * [m/E, 1-m/E] interval near threshold; it does not change the distribution.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool decode_photon_pair_normalized_split(
      double parent_energy_GeV, double normalized_split, double& split_fraction);

  /** Encode the electron energy fraction to (fraction - m/E) / (1 - 2m/E).
   * This is a coordinate conversion, not a new distribution or energy sampler.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline bool encode_photon_pair_normalized_split(
      double parent_energy_GeV, double split_fraction, double& normalized_split);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/models/PhotonPairKinematics.inl>
