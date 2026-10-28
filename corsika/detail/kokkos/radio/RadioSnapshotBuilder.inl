/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  template <typename TEnvironment, typename TCoreasDetector, typename TZhsDetector>
  RadioConfig make_radio_config(TEnvironment const& environment, Point const& upper_limit,
                                Point const& lower_limit, LengthType const step,
                                TCoreasDetector const& coreas_detector,
                                TZhsDetector const& zhs_detector, bool coreas_enabled,
                                bool zhs_enabled) {
    return atmosphere::make_radio_config(environment, upper_limit, lower_limit, step,
                                         coreas_detector, zhs_detector, coreas_enabled,
                                         zhs_enabled);
  }
} // namespace corsika::kokkos::radio
