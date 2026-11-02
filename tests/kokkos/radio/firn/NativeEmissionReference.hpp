/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/radio/firn/IGeometricalOpticsAccumulator.hpp>

namespace firn_test {
  corsika::kokkos::radio::firn::GeometricalOpticsSpectra native_uniform_emission(
      corsika::kokkos::radio::firn::GeometricalOpticsTrack const& track,
      std::vector<corsika::kokkos::radio::firn::vector_type> const& observers,
      corsika::kokkos::radio::firn::OnlineRadioConfig const& config, double index);
}
