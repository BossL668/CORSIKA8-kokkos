/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/accelerator/em/IAcceleratedEmBackend.hpp>
#include <corsika/gpu/em/CudaEmBackend.hpp>

namespace corsika::accelerator::em {

  /** Non-owning adapter; native CUDA allocation and session timing are unchanged. */
  class NativeCudaBackendAdapter final : public IAcceleratedEmBackend {
  public:
    explicit NativeCudaBackendAdapter(gpu::em::CudaEmBackend& backend) noexcept
        : backend_(backend) {}

    void beginShower(AcceleratedEmShowerConfig const& config) override {
      backend_.beginShower(config);
    }

    bool canTransport(gpu::em::EmParticleState const& state) const override {
      return backend_.canTransport(state);
    }

    bool hasProposalTable() const override {
      return backend_.hasProposalTable();
    }

    std::size_t minimumBatchSize() const override {
      return backend_.minimumBatchSize();
    }

    std::size_t maximumResidentPhotonBatchSize() const override {
      return backend_.maximumResidentPhotonBatchSize();
    }

    std::size_t maximumResidentLeptonBatchSize() const override {
      return backend_.maximumResidentLeptonBatchSize();
    }

    std::size_t maximumResidentInputBatchSize() const override {
      return backend_.maximumResidentInputBatchSize();
    }

    std::size_t pendingPhotonCount() const noexcept override {
      return backend_.pendingPhotonCount();
    }

    std::size_t pendingLeptonCount() const noexcept override {
      return backend_.pendingLeptonCount();
    }

    gpu::em::ResidentPhotonCascadeResult runPhotonWavefront(
        std::vector<gpu::em::EmParticleState> const& particles,
        std::uint64_t const first_secondary_history_id,
        std::size_t const maximum_wavefronts,
        std::size_t const minimum_resident_batch_size) override {
      return backend_.runResidentPhotonCascadeForValidation(
          particles, first_secondary_history_id, maximum_wavefronts,
          minimum_resident_batch_size);
    }

    gpu::em::ResidentLeptonCascadeResult runLeptonWavefront(
        std::vector<gpu::em::EmParticleState> const& particles,
        std::uint64_t const first_secondary_history_id,
        std::size_t const maximum_wavefronts,
        std::uint64_t const secondary_history_id_limit_exclusive,
        std::size_t const minimum_resident_batch_size) override {
      return backend_.runResidentLeptonCascadeForValidation(
          particles, first_secondary_history_id, maximum_wavefronts,
          secondary_history_id_limit_exclusive,
          minimum_resident_batch_size);
    }

    BackendCapabilities capabilities() const override {
      return {
          AcceleratorKind::NativeCuda, true, true, true,
          true, backend_.gpuProfileEnabled(),
          backend_.gpuRadioEnabled(), backend_.gpuRadioEnabled(), true};
    }

    AcceleratedEmStatistics const& statistics() const override {
      return backend_.statistics();
    }

    bool gpuProfileEnabled() const noexcept override {
      return backend_.gpuProfileEnabled();
    }

    gpu::em::GpuProfileResult downloadProfile() override {
      return backend_.downloadProfile();
    }

    bool gpuRadioEnabled() const noexcept override {
      return backend_.gpuRadioEnabled();
    }

    gpu::radio::GpuRadioWaveforms downloadRadioWaveforms() override {
      return backend_.downloadRadioWaveforms();
    }

    std::optional<gpu::em::GpuFirstInteractionSnapshot>
    downloadFirstInteractionSnapshot() override {
      return backend_.downloadFirstInteractionSnapshot();
    }

    gpu::em::CudaEmBackend& nativeBackend() noexcept { return backend_; }

  private:
    gpu::em::CudaEmBackend& backend_;
  };

} // namespace corsika::accelerator::em
