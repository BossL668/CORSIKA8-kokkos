/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause license. */
#pragma once

#include <functional>
#include <stdexcept>
#if defined(__linux__)
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <system_error>
#include <vector>
#endif

namespace corsika::accelerator::em::detail {

// Used only by the explicit cooperative backend's independent CUDA driver.
// libgomp can bind the initial thread BEFORE main(), not just at Kokkos init.
// Its current OpenMP place partition retains the job's permitted CPU set.
// Unbound runtimes expose no places; preserve sched_getaffinity in that case.
inline std::function<void()> captureCooperativeDriverAffinity() {
#if defined(__linux__)
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    throw std::system_error(errno, std::generic_category(),
                            "Read cooperative CPU affinity");
  const int count = omp_get_partition_num_places();
  if (count > 0) {
    std::vector<int> places(static_cast<std::size_t>(count));
    omp_get_partition_place_nums(places.data());
    cpu_set_t partition;
    CPU_ZERO(&partition);
    for (const int place : places) {
      const int processors = omp_get_place_num_procs(place);
      if (processors <= 0)
        throw std::runtime_error("Empty cooperative OpenMP place");
      std::vector<int> ids(static_cast<std::size_t>(processors));
      omp_get_place_proc_ids(place, ids.data());
      for (const int id : ids) {
        if (id < 0 || id >= CPU_SETSIZE)
          throw std::runtime_error("Cooperative CPU affinity exceeds cpu_set_t");
        CPU_SET(id, &partition);
      }
    }
    if (CPU_COUNT(&partition) == 0)
      throw std::runtime_error("Empty cooperative OpenMP partition");
    allowed = partition;
  }
  return [allowed] {
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(allowed), &allowed);
    if (error != 0)
      throw std::system_error(error, std::generic_category(),
                              "Restore cooperative CUDA driver affinity");
  };
#else
  return [] {};
#endif
}

} // namespace corsika::accelerator::em::detail
