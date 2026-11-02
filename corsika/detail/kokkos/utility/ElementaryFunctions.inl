/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline double logarithm_one_plus(double value) {
#if defined(__HIP_DEVICE_COMPILE__)
    // DTK 22.10.1's device log1p(-0) returns +0. Preserve the host/CUDA contract.
    if (value == 0.) { return value; }
#endif
    return ::log1p(value);
  }

  CORSIKA_HOST_DEVICE inline double exponential_minus_one(double value) {
#if defined(__HIP_DEVICE_COMPILE__)
    // The same device-library issue affects expm1(-0).
    if (value == 0.) { return value; }
#endif
    return ::expm1(value);
  }

} // namespace corsika::kokkos
