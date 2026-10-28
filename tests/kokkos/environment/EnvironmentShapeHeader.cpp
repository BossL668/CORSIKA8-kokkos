/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// No preceding includes: the transfer record must be independently consumable.
#include <corsika/kokkos/environment/EnvironmentShape.hpp>

#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout_v<corsika::kokkos::EnvironmentShape>);
static_assert(std::is_trivially_copyable_v<corsika::kokkos::EnvironmentShape>);
static_assert(std::is_aggregate_v<corsika::kokkos::EnvironmentShape>);
static_assert(
    std::is_same_v<std::underlying_type_t<corsika::kokkos::EnvironmentShape::Kind>,
                   std::uint32_t>);
