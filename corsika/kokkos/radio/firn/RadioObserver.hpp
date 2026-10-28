/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/firn/VectorOperations.hpp>
#include <string>

namespace corsika::kokkos::radio::firn {
  /** Named receiver in the same local, metre-valued coordinates as the source tracks. */
  struct RadioObserver {
    std::string name_;
    vector_type positionM_;
  };
} // namespace corsika::kokkos::radio::firn
