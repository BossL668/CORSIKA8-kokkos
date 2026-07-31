/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <corsika/gpu/em/Types.hpp>
#include <corsika/gpu/radio/Types.hpp>

namespace corsika::gpu::radio {

  /**
   * Persistent device-side CoREAS/ZHS waveform accumulator.
   *
   * The input pointer passed to accumulateLeptonTracksOnDevice must address
   * LeptonTransportRecord objects already resident on the configured CUDA
   * device. No track data is copied through the host.
   */
  class CudaRadioAccumulator {
  public:
    CudaRadioAccumulator();
    ~CudaRadioAccumulator();

    CudaRadioAccumulator(CudaRadioAccumulator&&) noexcept;
    CudaRadioAccumulator& operator=(CudaRadioAccumulator&&) noexcept;

    CudaRadioAccumulator(CudaRadioAccumulator const&) = delete;
    CudaRadioAccumulator& operator=(CudaRadioAccumulator const&) = delete;

    void initialize(GpuRadioConfig const&, int device,
                    std::size_t memory_budget_bytes);
    bool enabled() const noexcept;
    void accumulateLeptonTracksOnDevice(
        em::LeptonTransportRecord const* device_records,
        std::size_t count, std::size_t input_slot = 0);
    /**
     * Validation/replay entry point for scalar tracks captured on the host.
     *
     * Production CUDA EM transport uses the device-pointer overload and
     * avoids this transfer.  This overload deliberately performs an exact
     * host-to-device copy so a legacy CPU decision tape can be projected by
     * the CUDA CoREAS/ZHS kernels without resampling the shower.
     */
    void accumulateLeptonTracksFromHost(
        std::vector<em::LeptonTransportRecord> const& records);
    /**
     * Wait only for the radio batch that still reads the selected EM
     * double-buffer slot. This allows the other slot to remain in flight.
     */
    void waitForInputSlot(std::size_t input_slot);
    void drain();
    GpuRadioWaveforms downloadWaveforms();
    /**
     * Clear waveforms, device counters and per-shower statistics while
     * preserving propagation tables, observers and device allocations.
     */
    void reset();
    void release() noexcept;
    std::size_t deviceBytes() const noexcept;
    GpuRadioStatistics const& statistics() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
  };

} // namespace corsika::gpu::radio
