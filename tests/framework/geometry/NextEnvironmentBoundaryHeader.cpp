/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/geometry/NextEnvironmentBoundary.hpp>

#include <type_traits>
#include <utility>

using result_type = corsika::BoundaryResult<double, int>;
static_assert(
    std::is_same_v<decltype(std::declval<result_type const&>().getParameter()), double>);
static_assert(
    std::is_same_v<decltype(std::declval<result_type const&>().getNode()), int>);
