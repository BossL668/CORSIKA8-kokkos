/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/kokkos/egs4/Session.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/kokkos/data/RouterParticleConversion.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace corsika::kokkos::egs4::application {
  struct HostStatistics {
    std::uint64_t transportedHadrons_{}, transportedMuons_{}, selectedPhotonuclear_{};
    std::uint64_t generatedSecondaries_{}, retainedSecondaries_{}, arrivalCuts_{};
  };
  /** Consume an already-selected EGS4 nuclear vertex without resampling its target. */
  template <class TStack, class TEnvironment, class TPhotonModel, class TSecondaries,
            class TArrival>
  class HostHandler;
} // namespace corsika::kokkos::egs4::application
#include <corsika/detail/kokkos/egs4/HostHandler.inl>
