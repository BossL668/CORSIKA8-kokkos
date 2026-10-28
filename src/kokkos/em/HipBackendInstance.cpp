/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/detail/kokkos/em/execution/BackendInstanceImplementation.inl>

namespace corsika::kokkos {

  std::unique_ptr<BackendInstance> make_hip_backend_instance(
      RuntimeConfig const& config) {
    return std::make_unique<BackendInstanceImplementation<Kokkos::HIP>>(config);
  }

} // namespace corsika::kokkos
