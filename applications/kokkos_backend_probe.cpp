/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/accelerator/em/KokkosRuntime.hpp>

#include <CLI/CLI.hpp>

#include <cstddef>
#include <iostream>

int main(int argc, char** argv) {
  CLI::App app{"CORSIKA 8 mutually exclusive Kokkos backend probe"};
  int threads{};
  int device{};
  std::size_t values{1U << 20U};
  app.add_option("--threads", threads);
  app.add_option("--device", device);
  app.add_option("--values", values);
  CLI11_PARSE(app, argc, argv);

  corsika::accelerator::em::KokkosRuntimeConfig runtime_config;
  runtime_config.device = device;
  runtime_config.threads = threads;
  corsika::accelerator::em::KokkosRuntime runtime{runtime_config};
  auto const probe = runtime.runPrimitiveProbe(values);
  auto const queue = runtime.runQueueProbe(4096);
  auto const& info = runtime.info();
  auto const memory_query_valid =
      !info.gpu ||
      (info.device_total_memory_bytes != 0 &&
       info.device_free_memory_bytes_at_initialization != 0 &&
       info.device_free_memory_bytes_at_initialization <=
           info.device_total_memory_bytes);
  std::cout << "backend=" << info.backend << '\n'
            << "kokkos_version=" << info.kokkos_version << '\n'
            << "device_name=" << info.device_name << '\n'
            << "concurrency=" << info.concurrency << '\n'
            << "gpu=" << (info.gpu ? "true" : "false") << '\n'
            << "openmp=" << (info.openmp ? "true" : "false") << '\n'
            << "device_total_memory_bytes="
            << info.device_total_memory_bytes << '\n'
            << "device_free_memory_bytes_at_initialization="
            << info.device_free_memory_bytes_at_initialization << '\n'
            << "memory_query_valid="
            << (memory_query_valid ? "true" : "false") << '\n'
            << "values=" << probe.values << '\n'
            << "selected=" << probe.selected << '\n'
            << "checksum=" << probe.checksum << '\n'
            << "scan_valid=" << (probe.scan_valid ? "true" : "false")
            << '\n'
            << "queue_retained=" << queue.retained_particles << '\n'
            << "queue_secondary_slots=" << queue.secondary_slots << '\n'
            << "queue_stable_order="
            << (queue.stable_order ? "true" : "false") << '\n'
            << "queue_roundtrip_exact="
            << (queue.roundtrip_exact ? "true" : "false")
            << '\n';
  return probe.scan_valid && queue.stable_order && queue.roundtrip_exact &&
                 memory_query_valid
             ? 0
             : 2;
}
