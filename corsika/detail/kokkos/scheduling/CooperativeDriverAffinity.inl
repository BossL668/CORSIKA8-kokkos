/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <stdexcept>
#if defined(__linux__)
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <system_error>
#include <vector>
#endif

namespace corsika::kokkos {

  // Used only by the explicit cooperative backend's independent CUDA driver.
  // libgomp can bind the initial thread BEFORE main(), not just at Kokkos init.
  // Its current OpenMP place partition retains the job's permitted CPU set.
  // Unbound runtimes expose no places; preserve sched_getaffinity in that case.
  inline std::function<void()> capture_cooperative_driver_affinity() {
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "Read cooperative CPU affinity");
    int const count = omp_get_partition_num_places();
    if (count > 0) {
      std::vector<int> places(static_cast<std::size_t>(count));
      omp_get_partition_place_nums(places.data());
      cpu_set_t partition;
      CPU_ZERO(&partition);
      for (int const place : places) {
        int const processors = omp_get_place_num_procs(place);
        if (processors <= 0) throw std::runtime_error("Empty cooperative OpenMP place");
        std::vector<int> ids(static_cast<std::size_t>(processors));
        omp_get_place_proc_ids(place, ids.data());
        for (int const id : ids) {
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
      int const error = pthread_setaffinity_np(pthread_self(), sizeof(allowed), &allowed);
      if (error != 0)
        throw std::system_error(error, std::generic_category(),
                                "Restore cooperative CUDA driver affinity");
    };
#else
    return [] {};
#endif
  }

} // namespace corsika::kokkos
