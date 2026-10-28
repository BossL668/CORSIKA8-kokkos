/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  /** Physics-domain gate shared by native CUDA and all Kokkos backends. */
  struct PhysicsRequirements {
    double maximumPrimaryEnergyMeV_{};
    double emTransportCutMeV_{};
    double muonTransportCutMeV_{};
    double stochasticCutMeV_{};
  };

} // namespace corsika::kokkos
