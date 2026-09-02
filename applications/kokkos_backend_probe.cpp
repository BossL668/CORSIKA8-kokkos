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

  corsika::accelerator::em::KokkosRuntime runtime{{device, threads}};
  auto const probe = runtime.runPrimitiveProbe(values);
  auto const queue = runtime.runQueueProbe(4096);
  auto const& info = runtime.info();
  std::cout << "backend=" << info.backend << '\n'
            << "kokkos_version=" << info.kokkos_version << '\n'
            << "device_name=" << info.device_name << '\n'
            << "concurrency=" << info.concurrency << '\n'
            << "gpu=" << (info.gpu ? "true" : "false") << '\n'
            << "openmp=" << (info.openmp ? "true" : "false") << '\n'
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
  return probe.scan_valid && queue.stable_order && queue.roundtrip_exact
             ? 0
             : 2;
}
