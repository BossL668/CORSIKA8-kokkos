/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace corsika::kokkos {

  struct CpuPrimarySubshowerPolicy {
    // Mirror the legacy primary/auxiliary split: learn ONLY the input share.
    // No time budget, rate-derived input limit, or shrinking resident arena.
    static constexpr double InitialGpuShare = 1. / 16.;
    static double gpuShare(double hostRate, double deviceRate) noexcept;
    // The primary must retain a FULL native arena, not just the minimum launch
    // threshold. Donating from that arena fragments the 130-thread wavefront.
    // This protects deliberately assigned input, not in-flight descendant counts.
    static std::size_t hostReserve(std::size_t capacity,
                                   std::size_t /*minimum*/) noexcept;
    // Start auxiliary work only with a useful new batch. Already-owned tails and
    // explicit bounded memory spills still drain; this is never a particle cut.
    static std::size_t helperMinimum(std::size_t capacity, std::size_t minimum) noexcept;
    // The primary uses the ordinary resident lepton horizon; the auxiliary
    // uses the legacy helper horizon. These are safe checkpoints, not cuts.
    static constexpr std::size_t HostLeptonWaves = 1024;
    static constexpr std::size_t AuxiliaryLeptonWaves = 8;

    // An initially sub-minimum front must not checkpoint after every single
    // wave solely because the threshold exceeds its initial population.
    // A quarter-front threshold is a scheduling boundary, NOT a particle cut:
    // all remaining particles rejoin their owned queues at return. Keep the
    // ordinary native threshold once a front is at least four times as large.
    static std::size_t hostCheckpointMinimum(std::size_t input,
                                             std::size_t minimum) noexcept;

    // Scheduling only: prefer a useful other-species front over a tiny one.
    // Never wait for future arrivals or change the native arena/wave horizon.
    // Count deferrals in completed host epochs, not elapsed time, so a small
    // species receives service after at most eight such choices even if the
    // other front continually replenishes itself. A lone tail drains at once.
    static constexpr unsigned MaximumSpeciesDeferrals = 8;
    static unsigned selectHostSpecies(unsigned preferred,
                                      std::array<std::size_t, 2> const& counts,
                                      std::array<std::size_t, 2> const& capacities,
                                      std::size_t minimum,
                                      std::array<unsigned, 2>& deferrals) noexcept;
  };

} // namespace corsika::kokkos
#include <corsika/detail/kokkos/scheduling/CpuPrimarySubshowerPolicy.inl>
