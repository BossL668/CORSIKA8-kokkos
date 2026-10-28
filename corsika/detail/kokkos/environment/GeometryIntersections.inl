/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos {

  CORSIKA_HOST_DEVICE inline GeometryIntersections::GeometryIntersections() {}

  CORSIKA_HOST_DEVICE inline GeometryIntersections::GeometryIntersections(double entry,
                                                                          double exit)
      : valid_(true)
      , entry_(entry)
      , exit_(exit) {}

  CORSIKA_HOST_DEVICE inline bool GeometryIntersections::hasIntersections() const {
    return valid_;
  }

  CORSIKA_HOST_DEVICE inline double GeometryIntersections::getEntry() const {
    return entry_;
  }

  CORSIKA_HOST_DEVICE inline double GeometryIntersections::getExit() const {
    return exit_;
  }

} // namespace corsika::kokkos
