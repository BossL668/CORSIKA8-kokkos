/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <corsika/accelerator/em/AcceleratorKind.hpp>
#include <corsika/accelerator/em/common/Types.hpp>
#include <corsika/accelerator/em/detail/SubshowerCallbacks.hpp>

namespace corsika::accelerator::em {

  using AcceleratedEmShowerConfig = gpu::em::GpuEmShowerConfig;
  using AcceleratedEmStatistics = gpu::em::GpuEmStatistics;

  struct BackendCapabilities {
    AcceleratorKind kind{AcceleratorKind::KokkosOpenMP};
    bool photon_transport{};
    bool electron_transport{};
    bool positron_transport{};
    bool muon_transport{};
    bool resident_profile{};
    bool coreas_radio{};
    bool zhs_radio{};
    bool deterministic{};
  };

  /**
   * Host-wavefront boundary shared by native CUDA and Kokkos backends.
   *
   * There are deliberately no virtual calls in particle kernels.  A router
   * invokes this interface only before or after a complete resident
   * wavefront, while the concrete backend launches statically compiled
   * kernels for exactly one execution space.
   */
  class IAcceleratedEmBackend {
  public:
    virtual ~IAcceleratedEmBackend() = default;

    virtual void beginShower(AcceleratedEmShowerConfig const&) = 0;
    virtual bool canTransport(gpu::em::EmParticleState const&) const = 0;
    virtual bool hasProposalTable() const = 0;

    virtual std::size_t minimumBatchSize() const = 0;
    virtual std::size_t maximumResidentPhotonBatchSize() const = 0;
    virtual std::size_t maximumResidentLeptonBatchSize() const = 0;
    virtual std::size_t maximumResidentInputBatchSize() const = 0;
    virtual std::size_t pendingPhotonCount() const noexcept = 0;
    virtual std::size_t pendingLeptonCount() const noexcept = 0;

    // Opt-in air cooperative protocol. Single endpoints retain the synchronous
    // wavefront API unchanged. All callbacks execute on the calling thread.
    virtual bool independentSubshowersEnabled() const noexcept { return false; }
    virtual bool independentSubshowersReady() const noexcept { return false; }
    // Coordinator-only policy, never queried from a device kernel. Adaptive
    // endpoints may coalesce specified CPU completions without joining the
    // other endpoint. Legacy independent/single backends retain their policy.
    virtual bool batchIndependentSpecifiedFallbacks() const noexcept { return false; }
    virtual std::size_t advanceIndependentSubshowers(
        std::vector<gpu::em::EmParticleState> const&,
        detail::SubshowerCallbacks const&) {
      throw std::logic_error("Independent subshowers are unavailable on this backend");
    }

    virtual gpu::em::ResidentPhotonCascadeResult runPhotonWavefront(
        std::vector<gpu::em::EmParticleState> const&,
        std::uint64_t first_secondary_history_id,
        std::size_t maximum_wavefronts,
        std::size_t minimum_resident_batch_size) = 0;

    virtual gpu::em::ResidentLeptonCascadeResult runLeptonWavefront(
        std::vector<gpu::em::EmParticleState> const&,
        std::uint64_t first_secondary_history_id,
        std::size_t maximum_wavefronts,
        std::uint64_t secondary_history_id_limit_exclusive,
        std::size_t minimum_resident_batch_size) = 0;

    // Compatibility spelling used by the existing native-CUDA router.  It
    // remains a host-side forwarding call and introduces no device dispatch.
    gpu::em::ResidentPhotonCascadeResult
    runResidentPhotonCascadeForValidation(
        std::vector<gpu::em::EmParticleState> const& particles,
        std::uint64_t const first_secondary_history_id,
        std::size_t const maximum_wavefronts,
        std::size_t const minimum_resident_batch_size) {
      return runPhotonWavefront(
          particles, first_secondary_history_id, maximum_wavefronts,
          minimum_resident_batch_size);
    }

    gpu::em::ResidentLeptonCascadeResult
    runResidentLeptonCascadeForValidation(
        std::vector<gpu::em::EmParticleState> const& particles,
        std::uint64_t const first_secondary_history_id,
        std::size_t const maximum_wavefronts,
        std::uint64_t const secondary_history_id_limit_exclusive,
        std::size_t const minimum_resident_batch_size) {
      return runLeptonWavefront(
          particles, first_secondary_history_id, maximum_wavefronts,
          secondary_history_id_limit_exclusive,
          minimum_resident_batch_size);
    }

    virtual BackendCapabilities capabilities() const = 0;
    virtual AcceleratedEmStatistics const& statistics() const = 0;

    virtual bool gpuProfileEnabled() const noexcept = 0;
    virtual gpu::em::GpuProfileResult downloadProfile() = 0;
    virtual bool gpuRadioEnabled() const noexcept = 0;
    virtual gpu::radio::GpuRadioWaveforms downloadRadioWaveforms() = 0;
    virtual std::optional<gpu::em::GpuFirstInteractionSnapshot>
    downloadFirstInteractionSnapshot() = 0;
  };

} // namespace corsika::accelerator::em
