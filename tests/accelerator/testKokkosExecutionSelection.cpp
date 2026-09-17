/* Host-only selection matrix: no GPU runtime is loaded by this test. */
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
int main() {
  using corsika::accelerator::em::resolveKokkosExecutionBackend;
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
  std::string const expected="cuda";
#elif defined(CORSIKA8_KOKKOS_BACKEND_CUDA)
  std::string const expected="cuda";
#elif defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
  std::string const expected="openmp";
#elif defined(CORSIKA8_KOKKOS_BACKEND_HIP)
  std::string const expected="hip";
#elif defined(CORSIKA8_KOKKOS_BACKEND_SYCL)
  std::string const expected="sycl";
#endif
  if(resolveKokkosExecutionBackend("")!=expected)return 1;
  for(auto const* value:{"cuda","openmp","hip","sycl","cuda-openmp","openmp-cuda","unknown"}) {
    bool accepted=false;
    try {accepted=resolveKokkosExecutionBackend(value)==value;}
    catch(std::invalid_argument const&) {}
#if defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    bool valid=std::string(value)=="cuda" || std::string(value)=="openmp" ||
        std::string(value)=="cuda-openmp" || std::string(value)=="openmp-cuda";
#else
    bool valid=std::string(value)==expected;
#endif
    if(accepted!=valid)throw std::runtime_error(std::string("bad selection gate: ")+value);
  }
  std::cout<<"PASS: compiled selection defaults and all cross-backend rejection gates\n";
}
