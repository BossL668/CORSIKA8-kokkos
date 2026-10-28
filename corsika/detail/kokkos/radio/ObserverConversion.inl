/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  namespace snapshot_detail {
    template <typename TDetector>
    std::vector<RadioObserverSnapshot> make_observers(
        TDetector const& detector, CoordinateSystemPtr const& coordinate_system) {
      std::vector<RadioObserverSnapshot> result;
      result.reserve(static_cast<std::size_t>(detector.size()));
      for (auto const& observer : detector.getObservers()) {
        RadioObserverSnapshot snapshot{};
        auto const position = observer.getLocation().getCoordinates(coordinate_system);
        for (int axis = 0; axis < 3; ++axis) {
          snapshot.positionM_[axis] = position[axis] / 1_m;
        }
        snapshot.startTimeS_ = observer.getStartTime() / 1_s;
        snapshot.durationS_ = observer.getDuration() / 1_s;
        snapshot.sampleRateHz_ = observer.getSampleRate() / 1_Hz;
        snapshot.numberOfBins_ = observer.getWaveformX().size();
        result.push_back(snapshot);
      }
      return result;
    }
  } // namespace snapshot_detail
} // namespace corsika::kokkos::radio
