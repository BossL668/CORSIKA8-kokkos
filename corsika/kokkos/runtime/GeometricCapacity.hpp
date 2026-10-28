/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::kokkos {

  inline std::size_t checked_geometric_capacity(
      std::size_t const current, std::size_t const requested,
      std::size_t const maximum = std::numeric_limits<std::size_t>::max());

} // namespace corsika::kokkos

#include <corsika/detail/kokkos/runtime/GeometricCapacity.inl>
