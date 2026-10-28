/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cmath>
#include <stdexcept>
namespace corsika::kokkos::radio::firn {
  inline void FullWaveDeviceConfig::validate() const {
    grid_.validate();
    if (!std::isfinite(timeOriginS_) || batchSegments_ == 0 || memoryLimitBytes_ == 0)
      throw std::invalid_argument("invalid full-wave device resource configuration");
  }
} // namespace corsika::kokkos::radio::firn
