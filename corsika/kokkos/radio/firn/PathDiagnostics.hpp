/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <corsika/kokkos/radio/firn/FrequencyGrid.hpp>
#include <corsika/kokkos/radio/firn/VectorOperations.hpp>

namespace corsika::kokkos::radio::firn {
  /** PathDiagnostics from the validated stratified-radio implementation. */
  struct PathDiagnostics {
    bool valid_{false};
    std::string topology_{"none"};
    double invariantP_{0.0};
    double sourceRefractiveIndex_{1.0};
    double observerRefractiveIndex_{1.0};
    double geometricLengthM_{0.0};
    double opticalPathM_{0.0};
    double travelTimeS_{0.0};
    double attenuationExponent_{0.0};
    double phaseOffsetRad_{0.0};
    double horizontalResidualM_{0.0};
    double fermatRelativeResidual_{0.0};
    double snellResidual_{0.0};
    double geometricSpreadingPerM_{0.0};
    vector_type emitDirection_{0.0, 0.0, 0.0};
    vector_type receiveDirection_{0.0, 0.0, 0.0};
  };
} // namespace corsika::kokkos::radio::firn
