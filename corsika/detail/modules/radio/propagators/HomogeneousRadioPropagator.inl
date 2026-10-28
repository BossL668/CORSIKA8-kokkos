/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

namespace corsika {
  template <typename TE>
  HomogeneousRadioPropagator<TE>::HomogeneousRadioPropagator(TE const& environment)
      : base_type(environment) {}

  template <typename TE>
  template <typename TP>
  typename HomogeneousRadioPropagator<TE>::signal_path_collection_type
  HomogeneousRadioPropagator<TE>::propagate(TP const&, Point const& source,
                                            Point const& destination) {
    auto const* universe = base_type::env_.getUniverse().get();
    auto const* sourceNode = universe->getContainingNode(source);
    auto const* destinationNode = universe->getContainingNode(destination);
    double const sourceIndex =
        sourceNode->getModelProperties().getRefractiveIndex(source);
    double const destinationIndex =
        destinationNode->getModelProperties().getRefractiveIndex(destination);
    double const scale =
        std::max({1., std::abs(sourceIndex), std::abs(destinationIndex)});
    if (std::abs(sourceIndex - destinationIndex) > 1.e-12 * scale) {
      throw std::runtime_error("radio path left the homogeneous medium");
    }
    auto const emit = (destination - source).normalized();
    auto const receive = -emit;
    auto const distance = (destination - source).getNorm();
    double const averageIndex = 0.5 * (sourceIndex + destinationIndex);
    TimeType const time = averageIndex * distance / constants::c;
    std::deque<Point> points{source, destination};
    return {SignalPath(time, averageIndex, sourceIndex, destinationIndex, emit, receive,
                       distance, points)};
  }
} // namespace corsika
