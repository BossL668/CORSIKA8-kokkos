/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline bool is_native_muon_pair_cut(ParticleState const& p) {
    return p.reserved_ == NativeMuonPairCutFlag && is_muon_pid(p.pid_) &&
           p.generation_ > 0 && p.stepId_ == 0 && p.energyGeV_ > 0. &&
           p.energyGeV_ <= transport_mass_gev(p.pid_);
  }

  CORSIKA_HOST_DEVICE inline bool is_native_muon_pair_cut_deposit(std::int32_t pid,
                                                                  std::uint32_t tag,
                                                                  double energy,
                                                                  double deposit) {
    auto const mass = transport_mass_gev(pid);
    auto const residual = deposit - (energy - mass);
    return tag == NativeMuonPairCutFlag && is_muon_pid(pid) && energy > 0. &&
           energy <= mass && deposit <= 0. && residual > -1.e-15 && residual < 1.e-15;
  }

} // namespace corsika::kokkos
