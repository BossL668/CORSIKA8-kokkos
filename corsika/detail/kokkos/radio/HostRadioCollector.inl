/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  template <typename TBackend>
  inline HostRadioCollector<TBackend>::HostRadioCollector(TBackend& backend)
      : backend_(backend) {
    if (!backend_.gpuRadioEnabled()) {
      throw std::logic_error("cannot redirect scalar radio to a disabled backend");
    }
  }

  template <typename TBackend>
  inline void HostRadioCollector<TBackend>::append(
      corsika::radio::RadioTrackSegment const& segment) {
    auto const record = make_host_radio_record(segment);
    backend_.enqueueRadioTracks({&record, 1});
  }
} // namespace corsika::kokkos::radio
