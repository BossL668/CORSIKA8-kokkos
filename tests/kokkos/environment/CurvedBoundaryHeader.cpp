/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/CurvedBoundaryAccess.hpp>

#include <type_traits>
#include <utility>

using tree_type = corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>;
using access_type = corsika::kokkos::CurvedBoundaryAccess<
    corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>>;
static_assert(std::is_copy_constructible_v<access_type>);
static_assert(std::is_same_v<decltype(std::declval<access_type const&>().intersect(0)),
                             corsika::kokkos::GeometryIntersections>);
static_assert(std::is_same_v<corsika::kokkos::straight_intersections_type,
                             corsika::kokkos::GeometryIntersections>);
