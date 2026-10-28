/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline LeptonTransportRecord native_muon_pair_cut_record(
      InteractionRecord const& interaction) {
    LeptonTransportRecord record{};
    record.interaction_ = interaction;
    record.interaction_.status_ = InteractionStatus::ParticleCut;
    record.interaction_.interactionVertexReached_ = 0;
    record.start_ = record.end_ = interaction.particle_;
    ++record.end_.stepId_;
    record.inputIndex_ = interaction.inputIndex_;
    record.limit_ = LeptonTransportLimit::ParticleCut;
    record.cutDepositedEnergyGeV_ =
        record.start_.energyGeV_ - transport_mass_gev(record.start_.pid_);
    return record;
  }

} // namespace corsika::kokkos
