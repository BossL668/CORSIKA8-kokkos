/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/accelerator/em/common/EmThinning.hpp>
#include <vector>

namespace corsika::accelerator::testing {
  struct ThinningInput {
    gpu::em::EmThinningConfig config;
    double energy, weight, e1, e2, u1, u2;
    gpu::em::EmThinningResult expected;
  };
  struct ThinningComparison {
    std::size_t decision_errors{}, weight_errors{}, non_bitwise_weights{};
    std::uint64_t maximum_weight_ulp{};
  };
  ThinningComparison compareThinningOnDevice(std::vector<ThinningInput> const& inputs);
}
