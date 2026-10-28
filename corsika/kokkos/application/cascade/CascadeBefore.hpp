/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace corsika::kokkos::application {
  /** Value snapshot of process counters at an event boundary.
   * Later changes to the monitored processes do not modify this snapshot.
   */
  template <typename THigh, typename TLow, typename TPool>
  struct CascadeBefore {
    THigh photoHigh_;
    TLow photoLow_;
    std::uint64_t lowInteractions_;
    std::uint64_t highInteractions_;
    std::size_t lowTimings_;
    std::size_t highTimings_;
    TPool pool_;
  };

} // namespace corsika::kokkos::application
