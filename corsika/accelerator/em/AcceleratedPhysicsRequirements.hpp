/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::accelerator::em {

  /** Physics-domain gate shared by native CUDA and all Kokkos backends. */
  struct AcceleratedPhysicsRequirements {
    double maximum_primary_energy_MeV{};
    double em_transport_cut_MeV{};
    double muon_transport_cut_MeV{};
    double stochastic_cut_MeV{};
  };

} // namespace corsika::accelerator::em
