/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/Types.hpp>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

namespace corsika::kokkos::radio::firn {
  /** MediumSample from the validated stratified-radio implementation. */
  struct MediumSample {
    double refractiveIndex_{1.0};
    double attenuationPerM_{0.0};
  };
} // namespace corsika::kokkos::radio::firn
