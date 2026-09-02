/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>

namespace corsika::accelerator::em {

  enum class AcceleratorKind : std::uint32_t {
    NativeCuda = 0,
    KokkosOpenMP = 1,
    KokkosCuda = 2,
    KokkosHip = 3,
    KokkosSycl = 4,
  };

} // namespace corsika::accelerator::em
