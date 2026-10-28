/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/EnvironmentTreeView.hpp>
#include <corsika/kokkos/environment/StraightBoundaryAccess.hpp>

#include <type_traits>
#include <utility>

using access_type = corsika::kokkos::StraightBoundaryAccess<
    corsika::kokkos::EnvironmentTreeView<Kokkos::HostSpace>>;
using intersection_type = corsika::kokkos::straight_intersections_type;
static_assert(std::is_copy_constructible_v<access_type>);
static_assert(std::is_trivially_copyable_v<intersection_type>);
static_assert(std::is_same_v<decltype(std::declval<access_type const&>().intersect(0)),
                             intersection_type>);
static_assert(std::is_same_v<decltype(std::declval<access_type const&>().getParent(0)),
                             std::int32_t>);
