/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::kokkos {

  enum class BackendKind : std::uint32_t {
    OpenMP = 1,
    Cuda = 2,
    Hip = 3,
    Sycl = 4,
  };

} // namespace corsika::kokkos
