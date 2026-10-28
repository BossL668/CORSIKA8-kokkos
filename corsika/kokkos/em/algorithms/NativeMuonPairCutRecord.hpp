/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>

namespace corsika::kokkos {

  /**
   * Construct the original zero-length native muon-pair compatibility cut.
   * The caller must first establish is_native_muon_pair_cut(interaction.particle_).
   * No transport, table query, resampling or queue operation is performed.
   * Preserve the signed E-m deposit and the original increment of the end step;
   * this function does not authorize any other below-rest-mass input.
   */
  [[nodiscard]] CORSIKA_HOST_DEVICE inline LeptonTransportRecord
  native_muon_pair_cut_record(InteractionRecord const& interaction);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/em/algorithms/NativeMuonPairCutRecord.inl>
