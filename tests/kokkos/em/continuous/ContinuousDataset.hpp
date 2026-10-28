/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <corsika/kokkos/em/InteractionRecord.hpp>
#include <corsika/kokkos/em/tables/ProposalNativeTable.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace continuous_test {
  enum class Operation : std::uint32_t {
    Dedx,
    Mass,
    TransportMass,
    Cut,
    Minimum,
    Range,
    Energy,
    AfterLoss,
    Prepare,
    Loss,
    Chain,
    Count
  };
  struct Query {
    Operation operation_{};
    corsika::kokkos::InteractionRecord interaction_{};
    double emCutMeV_{.4}, muonCutMeV_{.4};
    double argument_{};
    std::uint32_t continuousLimitWon_{};
    double nativeExpected_{};
    std::uint32_t nativeKind_{}; // 1 stopping power, 2 final E, 3 actual C8 limiter X.
    double fieldT_[3]{0., 0., 5.e-5};
  };
  struct Dataset {
    corsika::kokkos::ProposalNativeTableSet table_;
    std::vector<Query> queries_;
  };
  /** Build small real tables and call actual C8 getMaxStepLength plus its helper.
   * The diagnostic energy inversion uses independently owned PROPOSAL Displacement;
   * it does not execute C8 doContinuous/scattering, a shower, or radio emission.
   */
  Dataset make_dataset();
  std::string cache_directory();
} // namespace continuous_test
