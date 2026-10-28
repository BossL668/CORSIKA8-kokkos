/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <boost/core/span.hpp>
#include <corsika/modules/radio/RadioTrackSink.hpp>
#include <corsika/kokkos/radio/HostRadioRecord.hpp>

namespace corsika::kokkos::radio {
  template <typename TBackend>
  class HostRadioCollector final : public corsika::radio::RadioTrackSink {
  public:
    explicit HostRadioCollector(TBackend& backend);

    void append(corsika::radio::RadioTrackSegment const& segment) override;

  private:
    TBackend& backend_;
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/HostRadioCollector.inl>
