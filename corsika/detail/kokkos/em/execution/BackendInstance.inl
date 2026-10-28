/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  inline std::pair<std::size_t, std::size_t>
  BackendInstance::enableCooperativeHostProfileShards() {
    throw std::logic_error("host profile shards unavailable for this endpoint");
  }
} // namespace corsika::kokkos
