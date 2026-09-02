/*
 * Downstream-package smoke test for an installed Kokkos backend.
 * It deliberately uses only the installed public target and header tree.
 */

#include <corsika/accelerator/em/AcceleratorKind.hpp>
#include <corsika/accelerator/em/KokkosRuntime.hpp>

int main() {
  corsika::accelerator::em::KokkosRuntime runtime;
  return runtime.info().kind ==
                 corsika::accelerator::em::AcceleratorKind::NativeCuda
             ? 1
             : 0;
}
