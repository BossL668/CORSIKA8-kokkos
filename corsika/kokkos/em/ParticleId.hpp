/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/HostDevice.hpp>
#include <cstdint>

namespace corsika::kokkos {

  /** PDG identities supported by production electromagnetic transport. */
  enum class EmPid : std::int32_t {
    Electron = 11,
    Positron = -11,
    MuonMinus = 13,
    MuonPlus = -13,
    Photon = 22,
  };
  /** Original strict time cut, in seconds. */
  inline constexpr double ParticleCutMaximumTimeS = 10.e-3;
  /** Test the electron/positron PDG identifiers. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline constexpr bool is_electron_or_positron_pid(
      std::int32_t const pid);
  /** Test the two muon PDG identifiers. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline constexpr bool is_muon_pid(
      std::int32_t const pid);
  /** Test the charged leptons supported by this transport path. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline constexpr bool is_charged_lepton_pid(
      std::int32_t const pid);
  /** Apply the original strict time comparison without changing nonfinite behavior. */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline constexpr bool exceeds_particle_cut_time(
      double const time_s);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/ParticleId.inl>
