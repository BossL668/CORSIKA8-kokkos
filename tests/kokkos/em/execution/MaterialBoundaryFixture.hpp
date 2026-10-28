/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/kokkos/environment/EnvironmentDataView.hpp>
#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <vector>

namespace material_boundary_test {
  struct Fixture {
    std::vector<corsika::kokkos::EnvironmentNode> nodes_;
    std::vector<corsika::kokkos::EnvironmentShape> shapes_;
    std::vector<int> children_, exclusions_;
    std::vector<corsika::kokkos::EnvironmentModel> models_;
    std::vector<corsika::kokkos::EnvironmentComponent> composition_;
    std::vector<corsika::kokkos::InteractionRecord> inputs_;
    std::vector<double> distances_, grammages_;
    corsika::kokkos::EnvironmentDataView view() const;
  };
  // Actual C8 firn exporter and native grammage queries in a host-only TU.
  Fixture make_fixture();
} // namespace material_boundary_test
