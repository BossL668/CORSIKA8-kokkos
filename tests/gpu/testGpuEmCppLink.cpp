/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <cstdlib>

#include <corsika/gpu/em/CudaEmBackend.hpp>

int main() {
  corsika::gpu::em::CudaEmBackend backend;
  return backend.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
}
