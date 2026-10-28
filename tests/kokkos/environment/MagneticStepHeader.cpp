/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/environment/MagneticStep.hpp>

#include <cstdint>
#include <type_traits>

static_assert(std::is_same_v<std::underlying_type_t<corsika::kokkos::MagneticStepStatus>,
                             std::uint32_t>);
static_assert(std::is_standard_layout_v<corsika::kokkos::MagneticStepLimitResult>);
static_assert(std::is_trivially_copyable_v<corsika::kokkos::MagneticStepLimitResult>);
