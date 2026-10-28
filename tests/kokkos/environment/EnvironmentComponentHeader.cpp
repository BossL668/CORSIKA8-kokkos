/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// No preceding includes or native particle/medium dependencies.
#include <corsika/kokkos/environment/EnvironmentComponent.hpp>

#include <cstdint>
#include <type_traits>

using component_type = corsika::kokkos::EnvironmentComponent;
static_assert(std::is_standard_layout_v<component_type>);
static_assert(std::is_trivially_copyable_v<component_type>);
static_assert(std::is_aggregate_v<component_type>);
static_assert(std::is_same_v<decltype(component_type::code_), std::int32_t>);
static_assert(std::is_same_v<decltype(component_type::fraction_), double>);
static_assert(component_type{}.code_ == 0);
static_assert(component_type{}.fraction_ == 0.);
