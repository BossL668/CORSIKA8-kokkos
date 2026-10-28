/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/environment/EnvironmentModel.hpp>
#include <corsika/media/MediumProperties.hpp>
#include <stdexcept>

namespace corsika::kokkos::application::firn {
  /** Require the exported depth-dependent water-ice model for a pure firn domain.
   * Density variation does not change the elemental composition or select a new
   * transport table. Geometry is supplied by the native C8 environment tree.
   */
  template <typename TSnapshot>
  void require_firn(TSnapshot const& snapshot);
} // namespace corsika::kokkos::application::firn
#include <corsika/detail/kokkos/application/cascade/firn/FirnPolicy.inl>
