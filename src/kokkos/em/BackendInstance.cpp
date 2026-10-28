/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/execution/BackendInstance.hpp>
#include <corsika/framework/core/Logging.hpp>

namespace corsika::kokkos {

  void BackendInstance::reportUnmatchedTuningCache(std::string const& backend) {
    CORSIKA_LOG_WARN(
        "No exactly matching tuning cache was applied; using "
        "conservative {} defaults. Run c8_kokkos_tune for this "
        "device, build, thread count, and proposal-native table hash.",
        backend);
  }

} // namespace corsika::kokkos
