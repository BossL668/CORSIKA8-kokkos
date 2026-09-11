#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>

namespace em = corsika::accelerator::em;
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    em::KokkosRuntimeConfig config;
    config.execution_backend = argc > 1 ? argv[1] : "";
    auto const selected = em::resolveKokkosExecutionBackend(config.execution_backend);
    config.threads = selected == "openmp" ? 2 : 1;
#if !defined(CORSIKA8_KOKKOS_BACKEND_CUDA_OPENMP)
    auto invalid = config;
    invalid.cooperative_owner = true;
    bool rejected = false;
    try { em::KokkosRuntime wrong(invalid); }
    catch (std::invalid_argument const&) { rejected = true; }
    require(rejected && !Kokkos::is_initialized(), "independent build initialized cooperative runtime");
#endif
    {
      em::KokkosRuntime runtime(config);
      require(!runtime.info().cooperative_runtime, "ordinary runtime became cooperative");
      require(runtime.info().backend == selected, "ordinary execution space changed");
      require(runtime.info().host_threads == config.threads, "ordinary thread count changed");
      require(runtime.runPrimitiveProbe(1024).scan_valid, "scan failed");
      for (unsigned event = 0; event < 32; ++event) {
        auto const result = runtime.runQueueProbe(256 + event);
        require(result.stable_order && result.roundtrip_exact, "queue/RNG identity failed");
      }
      bool rejected = false;
      try { runtime.shareCooperativeLifetime(); }
      catch (std::logic_error const&) { rejected = true; }
      require(rejected, "ordinary lifetime was shared");
      rejected = false;
      try { em::KokkosRuntime duplicate(config); }
      catch (std::logic_error const&) { rejected = true; }
      require(rejected, "duplicate runtime ownership accepted");
      std::cout << "PASS: exclusive " << selected << " runtime, threads="
                << runtime.info().host_threads << ", 32 queue calls\n";
    }
    require(Kokkos::is_finalized(), "exclusive runtime did not finalize");
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
