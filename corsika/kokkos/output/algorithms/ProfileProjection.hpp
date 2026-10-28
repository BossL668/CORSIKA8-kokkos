/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/output/ProjectedStepRecord.hpp>
#include <corsika/kokkos/output/ProfileProjectionView.hpp>
#include <corsika/kokkos/em/PhotonTransportRecord.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <corsika/kokkos/em/algorithms/NativeMuonPairCut.hpp>

namespace corsika::kokkos {

  /** Project already-computed endpoints onto the production depth support.
   * Preserve interpolation, process tags, deposits and observation/cut overlap.
   * No transport, random draw, boundary search or profile accumulation occurs.
   */
  CORSIKA_HOST_DEVICE inline double project_profile_grammage(
      ProfileProjectionView const& projection, double const position_m[3]);
  CORSIKA_HOST_DEVICE inline ProjectedStepRecord project_photon_step(
      ProfileProjectionView const& projection, PhotonTransportRecord const& record);
  CORSIKA_HOST_DEVICE inline ProjectedStepRecord project_lepton_step(
      ProfileProjectionView const& projection, LeptonTransportRecord const& record);

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/output/algorithms/ProfileProjection.inl>
