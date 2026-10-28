/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  template <typename TCoreas, typename TZhs>
  template <typename TCoordinates>
  inline ScopedRadioTrackSink<TCoreas, TZhs>::ScopedRadioTrackSink(
      TCoreas& coreas, TZhs& zhs, corsika::radio::RadioTrackSink& sink,
      TCoordinates const& coordinates)
      : coreas_(coreas)
      , zhs_(zhs) {
    if (coreas_.hasTrackSink() || zhs_.hasTrackSink()) {
      throw std::logic_error("scalar radio already has a track sink");
    }
    // The common accelerated projector handles both enabled models. Even
    // though both processes visit the step, capture its geometry only once.
    coreas_.setTrackSink(&sink, coordinates, true);
    try {
      zhs_.setTrackSink(&sink, coordinates, coreas_.observerCollection().size() == 0);
    } catch (...) {
      coreas_.setTrackSink(nullptr, {});
      throw;
    }
  }

  template <typename TCoreas, typename TZhs>
  inline ScopedRadioTrackSink<TCoreas, TZhs>::~ScopedRadioTrackSink() {
    coreas_.setTrackSink(nullptr, {});
    zhs_.setTrackSink(nullptr, {});
  }
} // namespace corsika::kokkos::radio
