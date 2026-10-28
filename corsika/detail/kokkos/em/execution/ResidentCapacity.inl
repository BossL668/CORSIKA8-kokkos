/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {
  template <class TExecutionSpace>
  MemoryProjection project_resident_arenas(
      std::size_t count, std::size_t retained,
      PhotonFrontWorkspace<TExecutionSpace> const& photons,
      LeptonFrontWorkspace<TExecutionSpace> const& leptons,
      radio::RadioAccumulator<TExecutionSpace> const& radio, bool projectSteps,
      TExecutionSpace const& execution) {
    auto const chargedLimit = checked_memory_multiply(count, 3);
    auto const photonLimit = checked_memory_multiply(chargedLimit, 2);
    MemoryProjectionBuilder projection(retained);
    projection.appendDisjointGrowth(photons.deviceBytes(),
                                    photons.projectedCapacity(count));
    projection.appendDisjointGrowth(photons.deviceBytes(),
                                    photons.projectedCandidateCapacity(count));
    projection.appendDisjointGrowth(
        photons.bucketing_.deviceBytes(),
        photons.bucketing_.projectedCountRange(count, execution));
    // Queue and source projections both include diagnostic Views if called
    // through the parent workspace. Use the bare queue here to count those
    // shared diagnostics exactly once, in projectedSourceCapacity().
    projection.appendDisjointGrowth(leptons.queue_.deviceBytes(),
                                    leptons.queue_.projectedCapacity(chargedLimit));
    projection.appendDisjointGrowth(leptons.deviceBytes(),
                                    leptons.projectedSourceCapacity(count, chargedLimit));
    projection.appendDisjointGrowth(
        leptons.deviceBytes(),
        leptons.projectedOutputCapacity(count, count, checked_memory_multiply(count, 2),
                                        count, count, count, projectSteps, chargedLimit,
                                        photonLimit));
    projection.appendDisjointGrowth(
        leptons.bucketing_.deviceBytes(),
        leptons.bucketing_.projectedCountRange(count, execution));
    projection.appendDisjointGrowth(radio.deviceBytes(),
                                    radio.projectedTrackCapacity(count));
    return projection.result();
  }

  template <class TExecutionSpace>
  void reserve_resident_arenas(std::size_t count,
                               PhotonFrontWorkspace<TExecutionSpace>& photons,
                               LeptonFrontWorkspace<TExecutionSpace>& leptons,
                               radio::RadioAccumulator<TExecutionSpace>& radio,
                               bool projectSteps, TExecutionSpace const& execution) {
    auto const chargedLimit = checked_memory_multiply(count, 3);
    auto const photonLimit = checked_memory_multiply(chargedLimit, 2);
    photons.ensureCapacity(count, execution);
    photons.ensureCandidateCapacity(count, execution);
    photons.bucketing_.ensureCountRange(count, execution);
    leptons.ensureQueueCapacity(chargedLimit, execution);
    leptons.ensureSourceCapacity(count, chargedLimit, execution);
    leptons.ensureOutputCapacity(count, count, checked_memory_multiply(count, 2), count,
                                 count, count, projectSteps, chargedLimit, photonLimit,
                                 execution);
    leptons.bucketing_.ensureCountRange(count, execution);
    radio.reserveTrackCapacity(count, execution);
  }

  template <class TProject>
  ResidentCapacityPlan select_resident_capacity(std::size_t budget,
                                                std::size_t minimumBatch,
                                                std::size_t pendingParticlesPerSpecies,
                                                TProject const& project) {
    if (minimumBatch == 0 || budget == 0)
      throw std::invalid_argument("Kokkos automatic capacity needs nonzero limits");
    // Check before asking the projection to multiply any particle counts.
    auto const indexLimit =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() / 6U);
    if (minimumBatch > indexLimit)
      throw std::length_error("Kokkos minimum batch exceeds the device index limit");
    auto const pendingBytes =
        checked_memory_multiply(pendingParticlesPerSpecies, sizeof(ParticleState));
    auto const allocatorReserve = std::min(budget / 100, std::size_t{16} << 20);
    // Both pending queues may be full. During compaction/growth, one old
    // queue and its replacement coexist. Other arena Views are pre-reserved.
    auto const reserve =
        checked_memory_add(checked_memory_multiply(pendingBytes, 3), allocatorReserve);
    auto const available = budget > reserve ? budget - reserve : 0;
    auto const fits = [&](std::size_t count) {
      auto const estimate = project(count);
      return std::max(estimate.retainedBytes_, estimate.transientPeakBytes_) <= available;
    };
    if (!fits(minimumBatch))
      throw std::runtime_error(
          "Kokkos automatic resident capacity cannot fit the minimum batch "
          "and both EM/radio arenas in the configured device-memory budget");
    // Shared scans/indices and the 3x charged / 6x photon child bounds must
    // remain representable even on a very large device.
    auto low = minimumBatch;
    auto high = std::max(low, std::min(indexLimit, available / sizeof(ParticleState)));
    while (low < high) {
      auto const mid = low + (high - low + 1) / 2;
      if (fits(mid))
        low = mid;
      else
        high = mid - 1;
    }
    auto const estimate = project(low);
    return {low, pendingParticlesPerSpecies,
            checked_memory_add(estimate.retainedBytes_, 2 * pendingBytes),
            checked_memory_add(estimate.transientPeakBytes_, reserve), allocatorReserve};
  }
} // namespace corsika::kokkos
