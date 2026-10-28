/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/ParticleId.hpp>
#include <corsika/kokkos/em/algorithms/LeptonContinuousStep.hpp>
#include <corsika/kokkos/em/models/MoliereScattering.hpp>
#include <corsika/kokkos/em/random/RandomDomains.hpp>
#include <corsika/kokkos/em/random/Philox.hpp>
#include <Kokkos_Macros.hpp>
#include <cstddef>
#include <cstdint>

namespace corsika::kokkos {

  /** Apply production Moliere sampling to one completed charged-lepton step.
   * The caller supplies a previously validated particle snapshot and a borrowed
   * interpolation view accessible to the executing device. An empty view retains
   * the production analytical correction functions; no new input guards are added.
   * ComponentCapacity must cover the material's components and be at most
   * MaxMoliereComponents. Energies in the record are GeV, grammage g/cm^2 and angles
   * radians. Start/tracked directions share the same coordinate system.
   * Draw identities use the START history/step IDs, seed and shower ID, never a
   * queue index. Disabled sampling leaves both output objects untouched. Failure
   * retains the original partial writes and reports the original fallback draw.
   * The record and fallback must be exclusive, non-aliasing output slots.
   * Return true for success/no deflection/disabled sampling, false for fallback.
   */
  template <std::size_t ComponentCapacity>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool apply_moliere_scattering(
      MoliereSnapshot const& snapshot, MoliereInterpolationView const& interpolation,
      bool const enabled, std::uint64_t const random_seed, std::uint64_t const shower_id,
      LeptonTransportRecord& record, FallbackRequest& fallback);

  /** Connect scattering to production transport-state and vertex bookkeeping.
   * State 2 bypasses scattering and becomes 0; other nonzero states pass through
   * unchanged. State 0 selects the electron or muon snapshot by the start PID.
   * A missing muon snapshot requests CPU fallback. On success only interaction
   * and decay candidates copy the updated endpoint into the interaction record.
   * Return 0 for success/bypass, 1 for a new fallback, or the input failure state.
   * This stage does not perform transport, final-state generation or CPU handling.
   * Snapshot/view validity, units and ownership match apply_moliere_scattering.
   */
  template <std::size_t ComponentCapacity>
  [[nodiscard]] KOKKOS_INLINE_FUNCTION std::uint32_t apply_moliere_scattering_stage(
      MoliereSnapshot const& electron_snapshot, MoliereSnapshot const& muon_snapshot,
      MoliereInterpolationView const& interpolation, bool const muon_snapshot_available,
      std::uint64_t const random_seed, std::uint64_t const shower_id,
      LeptonTransportRecord& record, FallbackRequest& fallback,
      std::uint32_t const transport_state);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/MoliereStep.inl>
