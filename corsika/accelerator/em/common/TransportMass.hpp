/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/accelerator/AcceleratorMacros.hpp>
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <corsika/framework/core/ParticleProperties.hpp>
#endif

#include <cstdint>

namespace corsika::gpu::em {

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

  C8_ACCELERATOR_INLINE_FUNCTION inline double transportMassGeV(
      std::int32_t const pid) {
    if (pid == 11 || pid == -11) return TransportElectronMassGeV;
    if (pid == 13 || pid == -13) return TransportMuonMassGeV;
    return 0.;
  }

  /** Match InteractionModel's PROPOSAL secondary -> kinetic-energy stack adapter. */
  C8_ACCELERATOR_INLINE_FUNCTION inline double proposalSecondaryTransportEnergyGeV(
      std::int32_t const pid, double const native_total_energy_GeV,
      double const native_mass_GeV) {
    return (native_total_energy_GeV - native_mass_GeV) + transportMassGeV(pid);
  }

  // This signed source correction is a mass-table convention, not a deposit.
  // Evaluate from masses directly so high-energy cancellation cannot erase it.
  C8_ACCELERATOR_INLINE_FUNCTION inline double weightedProposalMassCorrectionGeV(
      std::int32_t const pid, double const native_mass_GeV, double const weight) {
    return weight * (transportMassGeV(pid) - native_mass_GeV);
  }

} // namespace corsika::gpu::em
