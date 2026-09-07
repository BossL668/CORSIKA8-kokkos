/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "KokkosBackendInstance.inl"

namespace corsika::accelerator::em::detail {
std::unique_ptr<KokkosBackendInstance> makeCudaBackendInstance(
    KokkosRuntimeConfig const& config) {
  return std::make_unique<KokkosBackendInstanceImpl<Kokkos::Cuda>>(config);
}
} // namespace corsika::accelerator::em::detail
