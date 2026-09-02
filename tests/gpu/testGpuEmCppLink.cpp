/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cstdlib>
#include <stdexcept>

#include <corsika/accelerator/em/NativeCudaBackendAdapter.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>
#include <corsika/gpu/em/detail/CudaEmRunSession.hpp>

int main() {
  corsika::gpu::em::CudaEmBackend backend;
  if (!backend.empty()) { return EXIT_FAILURE; }
  corsika::accelerator::em::NativeCudaBackendAdapter adapter{backend};
  corsika::accelerator::em::IAcceleratedEmBackend* generic_backend =
      &adapter;
  if (generic_backend->capabilities().kind !=
      corsika::accelerator::em::AcceleratorKind::NativeCuda) {
    return EXIT_FAILURE;
  }

  corsika::gpu::em::detail::CudaEmRunSession session{
      corsika::gpu::em::GpuPhysicsSource::ProposalNative, {}, {}};
  if (session.initialized() || session.loadedRateTable() != nullptr) {
    return EXIT_FAILURE;
  }
  try {
    static_cast<void>(session.backend());
  } catch (std::logic_error const&) {
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
