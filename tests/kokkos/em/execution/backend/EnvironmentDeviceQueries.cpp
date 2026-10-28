/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include "EnvironmentHeaderQueries.hpp"
#include <corsika/kokkos/environment/EnvironmentStorage.hpp>
#include <Kokkos_Core.hpp>

namespace backend_test {
  void query_execution_environment(corsika::kokkos::EnvironmentDataView const& data,
                                   EnvironmentResults (&results)[2]) {
    using execution_space = Kokkos::DefaultExecutionSpace;
    corsika::kokkos::EnvironmentStorage<execution_space> storage(data);
    auto const deviceData = storage.getDataView();
    Kokkos::View<EnvironmentResults*, execution_space> output("header queries", 2);
    Kokkos::parallel_for(
        "header queries", Kokkos::RangePolicy<execution_space>(0, 2),
        KOKKOS_LAMBDA(unsigned index) {
          output(index) = query_header_environment(deviceData, index);
        });
    auto const host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
    for (unsigned i = 0; i != 2; ++i) results[i] = host(i);
  }
} // namespace backend_test
