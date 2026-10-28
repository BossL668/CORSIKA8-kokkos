/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/kokkos/runtime/BackendRuntime.hpp>
#include <stdexcept>

namespace corsika::kokkos {
  std::string resolve_execution_backend(std::string const& requested) {
#if defined(CORSIKA_KOKKOS_BACKEND_CUDA_OPENMP)
    auto const selected = requested.empty() ? std::string{"cuda"} : requested;
    if (selected != "cuda" && selected != "openmp" && selected != "cuda-openmp" &&
        selected != "openmp-cuda")
      throw std::invalid_argument(
          "This dual Kokkos binary supports cuda, openmp, cuda-openmp or openmp-cuda");
    return selected;
#else
#if defined(CORSIKA_KOKKOS_BACKEND_OPENMP)
    std::string const compiled = "openmp";
#elif defined(CORSIKA_KOKKOS_BACKEND_CUDA)
    std::string const compiled = "cuda";
#elif defined(CORSIKA_KOKKOS_BACKEND_HIP)
    std::string const compiled = "hip";
#elif defined(CORSIKA_KOKKOS_BACKEND_SYCL)
    std::string const compiled = "sycl";
#else
#error "A Kokkos backend must be selected"
#endif
    if (!requested.empty() && requested != compiled)
      throw std::invalid_argument("Requested Kokkos backend '" + requested +
                                  "' is unavailable; this binary contains '" + compiled +
                                  "'");
    return compiled;
#endif
  }
} // namespace corsika::kokkos
