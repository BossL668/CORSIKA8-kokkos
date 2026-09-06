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

#include <corsika/accelerator/em/kokkos/KokkosResidentLeptonCascade.hpp>
#include <corsika/accelerator/em/kokkos/KokkosResidentPhotonCascade.hpp>
#include <corsika/accelerator/radio/kokkos/KokkosRadioAccumulator.hpp>

namespace corsika::accelerator::em::kokkos_detail {

  struct KokkosResidentCapacityPlan {
    std::size_t input_particles{};
    std::size_t pending_particles_per_species{};
    std::size_t retained_bytes{};
    std::size_t transient_peak_bytes{};
    std::size_t allocator_reserve_bytes{};
  };

  /**
   * Exact owning-View projection for one reserved photon/lepton arena pair.
   * Each physical source front is limited to count; charged next-front storage
   * holds 3*count children. A larger next front is returned at an existing
   * lossless checkpoint before sampling another interaction, not spilled to
   * scalar transport. Source/output/radio Views therefore never need to grow
   * beyond count while these arenas are used.
   */
  template <class ExecutionSpace>
  KokkosMemoryProjection projectResidentArenas(
      std::size_t count, std::size_t retained,
      KokkosResidentPhotonWorkspace<ExecutionSpace> const& photons,
      KokkosResidentLeptonWorkspace<ExecutionSpace> const& leptons,
      radio::kokkos_detail::KokkosRadioAccumulator<ExecutionSpace> const& radio,
      bool project_steps, ExecutionSpace const& execution) {
    auto const charged_limit = checkedMemoryMultiply(count, 3);
    auto const photon_limit = checkedMemoryMultiply(charged_limit, 2);
    KokkosMemoryProjectionBuilder projection(retained);
    projection.appendDisjointGrowth(
        photons.deviceBytes(), photons.projectedCapacity(count));
    projection.appendDisjointGrowth(
        photons.deviceBytes(), photons.projectedCandidateCapacity(count));
    projection.appendDisjointGrowth(
        photons.bucketing.deviceBytes(),
        photons.bucketing.projectedCountRange(count, execution));
    // Queue and source projections both include diagnostic Views if called
    // through the parent workspace. Use the bare queue here to count those
    // shared diagnostics exactly once, in projectedSourceCapacity().
    projection.appendDisjointGrowth(
        leptons.queue.deviceBytes(),
        leptons.queue.projectedCapacity(charged_limit));
    projection.appendDisjointGrowth(
        leptons.deviceBytes(),
        leptons.projectedSourceCapacity(count, charged_limit));
    projection.appendDisjointGrowth(
        leptons.deviceBytes(), leptons.projectedOutputCapacity(
            count, count, checkedMemoryMultiply(count, 2), count, count,
            count, project_steps, charged_limit, photon_limit));
    projection.appendDisjointGrowth(
        leptons.bucketing.deviceBytes(),
        leptons.bucketing.projectedCountRange(count, execution));
    projection.appendDisjointGrowth(
        radio.deviceBytes(), radio.projectedTrackCapacity(count));
    return projection.result();
  }

  template <class ExecutionSpace>
  void reserveResidentArenas(
      std::size_t count,
      KokkosResidentPhotonWorkspace<ExecutionSpace>& photons,
      KokkosResidentLeptonWorkspace<ExecutionSpace>& leptons,
      radio::kokkos_detail::KokkosRadioAccumulator<ExecutionSpace>& radio,
      bool project_steps, ExecutionSpace const& execution) {
    auto const charged_limit = checkedMemoryMultiply(count, 3);
    auto const photon_limit = checkedMemoryMultiply(charged_limit, 2);
    photons.ensureCapacity(count, execution);
    photons.ensureCandidateCapacity(count, execution);
    photons.bucketing.ensureCountRange(count, execution);
    leptons.ensureQueueCapacity(charged_limit, execution);
    leptons.ensureSourceCapacity(count, charged_limit, execution);
    leptons.ensureOutputCapacity(
        count, count, checkedMemoryMultiply(count, 2), count, count, count,
        project_steps, charged_limit, photon_limit, execution);
    leptons.bucketing.ensureCountRange(count, execution);
    radio.reserveTrackCapacity(count, execution);
  }

  /** Largest representable source front fitting the complete byte budget. */
  template <class Project>
  KokkosResidentCapacityPlan selectResidentCapacity(
      std::size_t budget, std::size_t minimum_batch,
      std::size_t pending_particles_per_species, Project const& project) {
    if (minimum_batch == 0 || budget == 0)
      throw std::invalid_argument("Kokkos automatic capacity needs nonzero limits");
    // Check before asking the projection to multiply any particle counts.
    auto const index_limit = static_cast<std::size_t>(
        std::numeric_limits<std::uint32_t>::max() / 6U);
    if (minimum_batch > index_limit)
      throw std::length_error("Kokkos minimum batch exceeds the device index limit");
    auto const pending_bytes = checkedMemoryMultiply(
        pending_particles_per_species, sizeof(gpu::em::EmParticleState));
    auto const allocator_reserve = std::min(budget / 100, std::size_t{16} << 20);
    // Both pending queues may be full. During compaction/growth, one old
    // queue and its replacement coexist. Other arena Views are pre-reserved.
    auto const reserve = checkedMemoryAdd(
        checkedMemoryMultiply(pending_bytes, 3), allocator_reserve);
    auto const available = budget > reserve ? budget - reserve : 0;
    auto const fits = [&](std::size_t count) {
      auto const estimate = project(count);
      return std::max(estimate.retained_bytes, estimate.transient_peak_bytes)
             <= available;
    };
    if (!fits(minimum_batch))
      throw std::runtime_error(
          "Kokkos automatic resident capacity cannot fit the minimum batch "
          "and both EM/radio arenas in the configured device-memory budget");
    // Shared scans/indices and the 3x charged / 6x photon child bounds must
    // remain representable even on a very large device.
    auto low = minimum_batch;
    auto high = std::max(low, std::min(
        index_limit, available / sizeof(gpu::em::EmParticleState)));
    while (low < high) {
      auto const mid = low + (high - low + 1) / 2;
      if (fits(mid)) low = mid;
      else high = mid - 1;
    }
    auto const estimate = project(low);
    return {low, pending_particles_per_species,
            checkedMemoryAdd(estimate.retained_bytes, 2 * pending_bytes),
            checkedMemoryAdd(estimate.transient_peak_bytes, reserve),
            allocator_reserve};
  }

} // namespace corsika::accelerator::em::kokkos_detail
