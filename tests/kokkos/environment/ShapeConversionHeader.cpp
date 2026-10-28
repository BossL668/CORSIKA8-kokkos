/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/ShapeConversion.hpp>

#include <type_traits>
#include <utility>

static_assert(std::is_same_v<decltype(corsika::kokkos::export_shape(
                                 std::declval<corsika::IVolume const&>(),
                                 std::declval<corsika::CoordinateSystemPtr const&>())),
                             corsika::kokkos::EnvironmentShape>);
