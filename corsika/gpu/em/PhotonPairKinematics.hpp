/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#if defined(__CUDACC__)
#define CORSIKA_GPU_PAIR_KINEMATICS_HOST_DEVICE __host__ __device__
#else
#define CORSIKA_GPU_PAIR_KINEMATICS_HOST_DEVICE
#endif

namespace corsika::gpu::em {

  // PROPOSAL 7.6.2 ME converted from MeV to GeV.
  inline constexpr double ElectronMassGeV = 0.0005109989461;
  inline constexpr double ElectronMassMeV =
      1000. * ElectronMassGeV;
  inline constexpr double PhotonPairThresholdMeV =
      2. * ElectronMassMeV;

  namespace photon_pair_kinematics_detail {

    CORSIKA_GPU_PAIR_KINEMATICS_HOST_DEVICE inline bool finite(
        double value) {
      auto constexpr maximum = 1.7976931348623157e308;
      return value == value && value <= maximum &&
             value >= -maximum;
    }

  } // namespace photon_pair_kinematics_detail

  /**
   * Convert the kinematically normalized pair coordinate
   *
   *   s = (rho - m/E) / (1 - 2m/E)
   *
   * back to PROPOSAL's electron-energy fraction rho.  Storing s instead of
   * rho removes the collapsing [m/E, 1-m/E] interval at pair threshold.
   */
  CORSIKA_GPU_PAIR_KINEMATICS_HOST_DEVICE inline bool
  decodePhotonPairNormalizedSplit(
      double parent_energy_GeV, double normalized_split,
      double& split_fraction) {
    if (!photon_pair_kinematics_detail::finite(
            parent_energy_GeV) ||
        !photon_pair_kinematics_detail::finite(
            normalized_split) ||
        !(parent_energy_GeV > 2. * ElectronMassGeV)) {
      return false;
    }
    auto constexpr tolerance =
        64. * 2.2204460492503131e-16;
    if (normalized_split < -tolerance ||
        normalized_split > 1. + tolerance) {
      return false;
    }
    auto const bounded =
        normalized_split < 0.
            ? 0.
            : (normalized_split > 1. ? 1.
                                     : normalized_split);
    auto const lower = ElectronMassGeV / parent_energy_GeV;
    auto const width = 1. - 2. * lower;
    split_fraction = lower + width * bounded;
    return photon_pair_kinematics_detail::finite(
               split_fraction) &&
           split_fraction >= lower &&
           split_fraction <= 1. - lower;
  }

  CORSIKA_GPU_PAIR_KINEMATICS_HOST_DEVICE inline bool
  encodePhotonPairNormalizedSplit(
      double parent_energy_GeV, double split_fraction,
      double& normalized_split) {
    if (!photon_pair_kinematics_detail::finite(
            parent_energy_GeV) ||
        !photon_pair_kinematics_detail::finite(
            split_fraction) ||
        !(parent_energy_GeV > 2. * ElectronMassGeV)) {
      return false;
    }
    auto const lower = ElectronMassGeV / parent_energy_GeV;
    auto const width = 1. - 2. * lower;
    auto const upper = 1. - lower;
    auto constexpr tolerance =
        64. * 2.2204460492503131e-16;
    if (!(width > 0.) ||
        split_fraction < lower - tolerance ||
        split_fraction > upper + tolerance) {
      return false;
    }
    auto const bounded =
        split_fraction < lower
            ? lower
            : (split_fraction > upper ? upper
                                      : split_fraction);
    normalized_split = (bounded - lower) / width;
    return photon_pair_kinematics_detail::finite(
               normalized_split) &&
           normalized_split >= 0. &&
           normalized_split <= 1.;
  }

} // namespace corsika::gpu::em

#undef CORSIKA_GPU_PAIR_KINEMATICS_HOST_DEVICE
