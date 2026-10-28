/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/Logging.hpp>

namespace corsika::kokkos::testing {

  // Keep the host logging dependencies out of CUDA translation units.
  void logPhysicsContextPublications(int publications, int showerUpdates) {
    CORSIKA_LOG_INFO("context publications={} shower_updates={}", publications,
                     showerUpdates);
  }

} // namespace corsika::kokkos::testing
