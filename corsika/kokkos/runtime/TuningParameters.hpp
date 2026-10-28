/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>

namespace corsika::kokkos {
  /** Production execution parameters; zero keeps each backend default. */
  struct TuningParameters {
    std::size_t batchSize_{};
    std::size_t chunkSize_{};
    std::size_t teamSize_{};
    std::size_t trackTileSize_{};
    std::size_t observerTileSize_{};
    std::size_t deviceQueues_{1};
  };
} // namespace corsika::kokkos
