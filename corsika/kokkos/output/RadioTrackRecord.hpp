/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/data/StepRecord.hpp>

namespace corsika::kokkos {
  /** Completed production transport segment and its two endpoint directions. */
  struct RadioTrackRecord {
    StepRecord step_{};
    double startDirection_[3]{};
    double endDirection_[3]{};
  };
} // namespace corsika::kokkos
