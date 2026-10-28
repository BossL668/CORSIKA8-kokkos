/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/em/EmRunSession.hpp>
#include <type_traits>
static_assert(std::is_nothrow_move_constructible_v<corsika::kokkos::EmRunSession>);
static_assert(!std::is_copy_constructible_v<corsika::kokkos::EmRunSession>);
