/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/framework/geometry/Point.hpp>

namespace corsika {

  /**
   * Read-only parameter export of a supported mass-density model.
   *
   * This host-side value retains C8 units and coordinate systems; it is not a
   * device-memory representation. Parameters are copied from the model, not fitted
   * to density samples. Export again after changing the source model.
   */
  struct DensityModelParameters {
    /// Supported density laws; this does not specify a volume or material identity.
    enum class Kind {
      Homogeneous,
      SlidingPlanarExponential,
      FlatExponential,
      ExponentialFirn
    };

    Kind kind_;                        ///< The density law described by these parameters.
    MassDensityType referenceDensity_; ///< Constant or reference mass density.
    LengthType scaleLength_; ///< Signed exponential scale length; zero if homogeneous.
    LengthType referenceHeight_; ///< Reference height for the exponential density.
    Point anchor_;               ///< Origin for the density model's height coordinate.
    DirectionVector axis_;       ///< Stored flat-exponential axis, without normalization.
    /// Firn: surface density deficit subtracted from referenceDensity_; otherwise zero.
    MassDensityType densityDeficit_{};
  };

} // namespace corsika
