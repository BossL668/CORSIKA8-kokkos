/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <corsika/kokkos/em/execution/BackendConfig.hpp>

namespace corsika::kokkos {
  struct ShowerConfig {
    std::uint64_t randomSeed_{};
    std::uint64_t showerId_{};
    EmThinningConfig thinning_{};
    double profileFixedPointWeightLimit_{};
    double profileFixedPointEnergyLimitGeV_{};
  };

  inline ShowerConfig make_shower_config(BackendConfig const& config);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/ShowerConfig.inl>
