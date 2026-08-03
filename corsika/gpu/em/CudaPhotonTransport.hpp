/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <vector>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/em/detail/DeviceWorkspace.hpp>

namespace corsika::gpu::em {

  /**
   * Advance selected photons through one CORSIKA-7 spherical atmosphere layer.
   *
   * Each input produces exactly one stable-order transport record or one
   * explicit CPU fallback. A record stops at the sampled discrete interaction,
   * the next layer boundary, the observation surface, or atmospheric escape.
   */
  PhotonTransportBatchResult transportPhotonsForValidation(
      EnvironmentSnapshot const& environment,
      std::vector<EmInteractionRecord> const& interactions, int device,
      detail::DeviceWorkspace& workspace);

} // namespace corsika::gpu::em
