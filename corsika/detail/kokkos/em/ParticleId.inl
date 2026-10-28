/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline constexpr bool is_electron_or_positron_pid(
      std::int32_t const pid) {
    return pid == static_cast<std::int32_t>(EmPid::Electron) ||
           pid == static_cast<std::int32_t>(EmPid::Positron);
  }

  CORSIKA_HOST_DEVICE inline constexpr bool is_muon_pid(std::int32_t const pid) {
    return pid == static_cast<std::int32_t>(EmPid::MuonMinus) ||
           pid == static_cast<std::int32_t>(EmPid::MuonPlus);
  }

  CORSIKA_HOST_DEVICE inline constexpr bool is_charged_lepton_pid(
      std::int32_t const pid) {
    return is_electron_or_positron_pid(pid) || is_muon_pid(pid);
  }

  CORSIKA_HOST_DEVICE inline constexpr bool exceeds_particle_cut_time(
      double const time_s) {
    return time_s > ParticleCutMaximumTimeS;
  }

} // namespace corsika::kokkos
