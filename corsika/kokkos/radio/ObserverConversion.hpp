/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/radio/RadioObserverSnapshot.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <vector>
namespace corsika::kokkos::radio {
  namespace snapshot_detail {
    template <typename TDetector>
    std::vector<RadioObserverSnapshot> make_observers(
        TDetector const& detector, CoordinateSystemPtr const& coordinate_system);
  }
} // namespace corsika::kokkos::radio
#include <corsika/detail/kokkos/radio/ObserverConversion.inl>
