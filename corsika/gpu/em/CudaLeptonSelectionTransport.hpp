/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <corsika/gpu/em/BremsLpm.hpp>
#include <corsika/gpu/em/MoliereScattering.hpp>
#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>
#include <corsika/gpu/em/tables/FlatRateTable.hpp>

namespace corsika::gpu::em {

  /**
   * Host-visible validation result of one charged electromagnetic wavefront.
   *
   * Selection, continuous transport, interaction-candidate extraction,
   * vertex-energy process reselection and bremsstrahlung dispatch stay on the
   * device. Intermediate selection and vertex-interaction arrays are
   * represented only by counts and are not copied through the host.
   */
  struct LeptonDevicePipelineBatchResult {
    std::size_t input_particles{};
    std::size_t interaction_candidates{};
    std::size_t vertex_interactions{};
    bool multiple_scattering_enabled{};
    std::vector<LeptonTransportRecord> transport_records{};
    std::vector<ProposalFallbackEvent>
        selection_fallback_events{};
    std::vector<ProposalFallbackEvent>
        transport_fallback_events{};
    std::vector<EmInteractionRecord> vertex_continuations{};
    std::vector<ProposalFallbackEvent>
        vertex_fallback_events{};
    BremsFinalStateBatchResult final_states{};
    std::vector<EmParticleState> next_leptons{};
    std::vector<EmParticleState> generated_photons{};
    std::vector<ObservationRecord> observations{};
    std::vector<EmParticleState> decay_candidates{};
  };

  /**
   * Device-chain one electron/positron wavefront.
   *
   * Versioned Moliere metadata enables multiple scattering inside the
   * transport kernel. A uniform field uses the same bounded leapfrog
   * trajectory and curved spherical intersections as the CPU tracker.
   */
  LeptonDevicePipelineBatchResult
  runLeptonDevicePipelineForValidation(
      tables::FlatRateTableView device_table,
      BremsLpmSnapshot const& lpm_snapshot,
      EmThinningConfig const& thinning,
      MoliereSnapshot const& moliere_snapshot,
      MoliereSnapshot const& muon_moliere_snapshot,
      MoliereInterpolationView const& moliere_interpolation,
      bool apply_moliere, bool muon_moliere_available,
      EnvironmentSnapshot const& environment,
      std::vector<EmParticleState> const& particles,
      std::uint64_t random_seed, std::uint64_t shower_id,
      int device, std::uint64_t first_secondary_history_id,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
