/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <functional>
#include <cstdint>
#include <corsika/kokkos/em/execution/PhotonCascadeResult.hpp>
#include <corsika/kokkos/em/execution/LeptonCascadeResult.hpp>

namespace corsika::kokkos {
  // Borrowed only during host progress; never captured by an endpoint job.
  struct SubshowerCallbacks {
    std::function<std::uint64_t(std::uint64_t)> reserveHistories_;
    std::function<void(PhotonCascadeResult&&, double)> photons_;
    std::function<void(LeptonCascadeResult&&, double)> leptons_;
    std::function<bool()> yieldToScalar_;
  };
} // namespace corsika::kokkos
