/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <corsika/kokkos/em/execution/LeptonCascade.hpp>
#include <corsika/kokkos/em/execution/PhotonCascade.hpp>
#include <corsika/kokkos/radio/RadioAccumulator.hpp>

namespace corsika::kokkos {
  struct ResidentCapacityPlan {
    std::size_t inputParticles_{};
    std::size_t pendingParticlesPerSpecies_{};
    std::size_t retainedBytes_{};
    std::size_t transientPeakBytes_{};
    std::size_t allocatorReserveBytes_{};
  };
  template <class TExecutionSpace>
  MemoryProjection project_resident_arenas(
      std::size_t count, std::size_t retained,
      PhotonFrontWorkspace<TExecutionSpace> const& photons,
      LeptonFrontWorkspace<TExecutionSpace> const& leptons,
      radio::RadioAccumulator<TExecutionSpace> const& radio, bool projectSteps,
      TExecutionSpace const& execution);
  template <class TExecutionSpace>
  void reserve_resident_arenas(std::size_t count,
                               PhotonFrontWorkspace<TExecutionSpace>& photons,
                               LeptonFrontWorkspace<TExecutionSpace>& leptons,
                               radio::RadioAccumulator<TExecutionSpace>& radio,
                               bool projectSteps, TExecutionSpace const& execution);
  template <class TProject>
  ResidentCapacityPlan select_resident_capacity(std::size_t budget,
                                                std::size_t minimumBatch,
                                                std::size_t pendingParticlesPerSpecies,
                                                TProject const& project);
} // namespace corsika::kokkos
#include <corsika/detail/kokkos/em/execution/ResidentCapacity.inl>
