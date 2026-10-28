/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/radio/RadioConfig.hpp>
#include <corsika/kokkos/radio/RadioWaveforms.hpp>
#include <corsika/kokkos/em/LeptonTransportRecord.hpp>
#include <string>
#include <vector>
namespace radio_test {
  struct ScalarCase {
    std::string name_;
    corsika::kokkos::radio::RadioConfig config_;
    corsika::kokkos::LeptonTransportRecord track_;
    corsika::kokkos::radio::RadioWaveforms expected_;
    corsika::kokkos::radio::RadioWaveforms release_;
  };
  std::vector<ScalarCase> make_scalar_cases();
} // namespace radio_test
