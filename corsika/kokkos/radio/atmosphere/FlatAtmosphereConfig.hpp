/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/RadioConfig.hpp>
#include <corsika/kokkos/radio/ObserverConversion.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace corsika::kokkos::radio::atmosphere {
  /** Export the production flat-atmosphere radio table and observer values.
   * Both limit points must be expressed in the environment coordinate system.
   * The environment must cover the sampled vertical column, including the lower
   * 1 km margin. This preserves the native discrete integral at all grid steps;
   * it is not a general ray tracer and does not export node ownership to a device.
   */
  template <typename TEnvironment, typename TCoreasDetector, typename TZhsDetector>
  RadioConfig make_radio_config(TEnvironment const& environment, Point const& upper_limit,
                                Point const& lower_limit, LengthType const step,
                                TCoreasDetector const& coreas_detector,
                                TZhsDetector const& zhs_detector,
                                bool coreas_enabled = true, bool zhs_enabled = true);
} // namespace corsika::kokkos::radio::atmosphere

#include <corsika/detail/kokkos/radio/atmosphere/FlatAtmosphereConfig.inl>
