/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/radio/IRadioBackend.hpp>

namespace corsika::kokkos::radio {
  /** Adapt the existing integrated backend without changing kernels or batching.
   * The caller owns the configured backend and must outlive this adapter.
   */
  template <typename TBackend>
  class BorrowedRadioBackend final : public IRadioBackend {
  public:
    /** Borrow an already configured backend, which must outlive this adapter. */
    explicit BorrowedRadioBackend(TBackend&);
    /** Forward the configured backend's radio capability. */
    bool gpuRadioEnabled() const noexcept override;
    /** Forward a borrowed range; the backend copies it before returning. */
    void enqueueRadioTracks(boost::span<LeptonTransportRecord const>) override;
    /** Flush the configured backend and retrieve its waveform output. */
    RadioWaveforms downloadRadioWaveforms() override;

  private:
    TBackend& backend_;
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/BorrowedRadioBackend.inl>
