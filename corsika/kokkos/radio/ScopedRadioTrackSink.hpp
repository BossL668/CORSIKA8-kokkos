/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/radio/RadioTrackSink.hpp>
#include <stdexcept>

namespace corsika::kokkos::radio {
  template <typename TCoreas, typename TZhs>
  class ScopedRadioTrackSink {
  public:
    template <typename TCoordinates>
    ScopedRadioTrackSink(TCoreas& coreas, TZhs& zhs, corsika::radio::RadioTrackSink& sink,
                         TCoordinates const& coordinates);

    ~ScopedRadioTrackSink();

    ScopedRadioTrackSink(ScopedRadioTrackSink const&) = delete;
    ScopedRadioTrackSink& operator=(ScopedRadioTrackSink const&) = delete;

  private:
    TCoreas& coreas_;
    TZhs& zhs_;
  };
} // namespace corsika::kokkos::radio

#include <corsika/detail/kokkos/radio/ScopedRadioTrackSink.inl>
