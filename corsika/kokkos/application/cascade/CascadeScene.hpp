/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika::kokkos::application {
  /** Select the material and radio preparation, never a scheduling policy.
   * Homogeneous ice requires every simulated radio path to stay in the ice.
   * Firn uses a depth-dependent model and its separate radio response adapter.
   */
  enum class CascadeScene { Atmosphere, HomogeneousIce, Firn };
} // namespace corsika::kokkos::application
