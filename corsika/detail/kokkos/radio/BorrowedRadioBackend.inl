/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  template <typename TBackend>
  BorrowedRadioBackend<TBackend>::BorrowedRadioBackend(TBackend& backend)
      : backend_(backend) {}

  template <typename TBackend>
  bool BorrowedRadioBackend<TBackend>::gpuRadioEnabled() const noexcept {
    return backend_.gpuRadioEnabled();
  }

  template <typename TBackend>
  void BorrowedRadioBackend<TBackend>::enqueueRadioTracks(
      boost::span<LeptonTransportRecord const> records) {
    backend_.enqueueRadioTracks(records);
  }

  template <typename TBackend>
  RadioWaveforms BorrowedRadioBackend<TBackend>::downloadRadioWaveforms() {
    return backend_.downloadRadioWaveforms();
  }
} // namespace corsika::kokkos::radio
