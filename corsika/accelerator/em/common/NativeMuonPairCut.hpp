/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#pragma once
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>

namespace corsika::gpu::em {
  // Native PROPOSAL can return E_mu <= MMU at an interpolated CDF endpoint.
  // C8 stores the signed kinetic energy, then ParticleCut removes the new
  // secondary. This tag authorizes ONLY that zero-length compatibility cut;
  // it must never relax the validity checks for ordinary transport inputs.
  inline constexpr std::uint32_t NativeMuonPairCutFlag = 1U;

  C8_ACCELERATOR_INLINE_FUNCTION inline bool isNativeMuonPairCut(
      EmParticleState const& p) {
    return p.reserved == NativeMuonPairCutFlag && isMuonPid(p.pid) &&
           p.generation > 0 && p.step_id == 0 && p.energy_GeV > 0. &&
           p.energy_GeV <= transportMassGeV(p.pid);
  }

  C8_ACCELERATOR_INLINE_FUNCTION inline bool isNativeMuonPairCutDeposit(
      std::int32_t pid, std::uint32_t tag, double energy, double deposit) {
    auto const mass = transportMassGeV(pid);
    auto const residual = deposit - (energy - mass);
    return tag == NativeMuonPairCutFlag && isMuonPid(pid) && energy > 0. &&
           energy <= mass && deposit <= 0. &&
           residual > -1.e-15 && residual < 1.e-15;
  }

  // Call only for isNativeMuonPairCut(interaction.particle). Shared by the
  // device queue and host queue/spill boundary: no transport or resampling.
  C8_ACCELERATOR_INLINE_FUNCTION inline LeptonTransportRecord nativeMuonPairCutRecord(
      EmInteractionRecord const& interaction) {
    LeptonTransportRecord record{};
    record.interaction = interaction;
    record.interaction.status = EmInteractionStatus::ParticleCut;
    record.interaction.interaction_vertex_reached = 0;
    record.start = record.end = interaction.particle;
    ++record.end.step_id;
    record.input_index = interaction.input_index;
    record.limit = LeptonTransportLimit::ParticleCut;
    record.cut_deposited_energy_GeV =
        record.start.energy_GeV - transportMassGeV(record.start.pid);
    return record;
  }
}
