#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <stdexcept>

namespace corsika::accelerator::em {
std::string resolveKokkosExecutionBackend(std::string const& requested) {
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
  auto const selected = requested.empty() ? std::string{"cuda"} : requested;
  if (selected != "cuda" && selected != "openmp")
    throw std::invalid_argument(
        "This dual Kokkos binary supports only cuda or openmp");
  return selected;
#else
#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
  std::string const compiled = "openmp";
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
  std::string const compiled = "cuda";
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
  std::string const compiled = "hip";
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
  std::string const compiled = "sycl";
#else
#error "A Kokkos backend must be selected"
#endif
  if (!requested.empty() && requested != compiled)
    throw std::invalid_argument("Requested Kokkos backend '" + requested +
                                "' is unavailable; this binary contains '" +
                                compiled + "'");
  return compiled;
#endif
}
} // namespace corsika::accelerator::em
