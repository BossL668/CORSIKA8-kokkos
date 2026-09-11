/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#include <corsika/accelerator/em/detail/CooperativeDriverAffinity.hpp>
#include <thread>
#include <iostream>
#include <exception>

int main(int argc, char** argv) {
#if defined(__linux__)
  cpu_set_t before, driver, after;
  if (sched_getaffinity(0, sizeof(before), &before)) return 1;
  auto restore = corsika::accelerator::em::detail::captureCooperativeDriverAffinity();
  std::exception_ptr error;
  std::thread worker([&] {
    try {
      restore();
      if (sched_getaffinity(0, sizeof(driver), &driver))
        throw std::runtime_error("Read driver affinity failed");
    } catch (...) { error = std::current_exception(); }
  });
  worker.join();
  if (error) std::rethrow_exception(error);
  if (sched_getaffinity(0, sizeof(after), &after)) return 2;
  if (!CPU_EQUAL(&before, &after)) return 3; // Never rebind the OpenMP master.
  if (argc > 1 && CPU_COUNT(&driver) != std::stoi(argv[1])) return 4;
  if (omp_get_partition_num_places() == 0 && !CPU_EQUAL(&before, &driver)) return 5;
  if (omp_get_partition_num_places() > 0) {
    cpu_set_t expected;
    CPU_ZERO(&expected);
    std::vector<int> places(omp_get_partition_num_places());
    omp_get_partition_place_nums(places.data());
    for (const int place : places) {
      std::vector<int> ids(omp_get_place_num_procs(place));
      omp_get_place_proc_ids(place, ids.data());
      for (const int id : ids) {
        if (id < 0 || id >= CPU_SETSIZE) return 6;
        CPU_SET(id, &expected);
      }
    }
    if (!CPU_EQUAL(&driver, &expected)) return 7;
  }
  std::cout << "master_cpus=" << CPU_COUNT(&before)
            << " driver_cpus=" << CPU_COUNT(&driver) << " ids=";
  for (int id = 0; id < CPU_SETSIZE; ++id)
    if (CPU_ISSET(id, &driver)) std::cout << id << ',';
  std::cout << '\n';
#endif
}
