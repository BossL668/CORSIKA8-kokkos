/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentDataView.hpp>
#include <array>
#include <vector>

namespace firn_test {
  // Values exchanged across the host/device test boundary. FirnProbe.cpp checks
  // that these match the public query status; this header cannot include Kokkos.
  enum class Status : int {
    Success = 0,
    InvalidSnapshot = 1,
    NoForwardIntersection = 5,
    GrammageOutOfRange = 6
  };

  struct Query {
    double positionM_[3]{};
    double direction_[3]{0., 0., -1.};
    double lengthM_{};
    double grammageGPerCm2_{};
  };
  struct Result {
    double values_[7]{};
    int statuses_[4]{};
  };
  char const* execution_name();
  std::vector<Result> probe(corsika::kokkos::EnvironmentDataView data,
                            std::vector<Query> const& queries);
} // namespace firn_test
