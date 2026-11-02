/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/framework/core/Logging.hpp>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>

// Run in a separate executable so the coordinator never initializes HIP before fork.
int main() {
  // Enumerate ordinals before the per-worker mask; retain any allocation-level mask.
  if (unsetenv("HIP_VISIBLE_DEVICES") != 0 || unsetenv("CUDA_VISIBLE_DEVICES") != 0) {
    CORSIKA_LOG_ERROR("Cannot clear per-worker GPU visibility for device query");
    return EXIT_FAILURE;
  }
  int count = 0;
  auto error = hipGetDeviceCount(&count);
  if (error != hipSuccess) {
    CORSIKA_LOG_ERROR("HIP device query: {}", hipGetErrorString(error));
    return EXIT_FAILURE;
  }
  for (int device = 0; device < count; ++device) {
    char address[32]{};
    error = hipDeviceGetPCIBusId(address, sizeof(address), device);
    if (error != hipSuccess) {
      CORSIKA_LOG_ERROR("HIP device {} PCI query: {}", device, hipGetErrorString(error));
      return EXIT_FAILURE;
    }
    // This is the coordinator's machine-readable protocol, not a log message.
    if (std::printf("%d,%s\n", device, address) < 0) {
      CORSIKA_LOG_ERROR("Cannot write HIP device listing");
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
