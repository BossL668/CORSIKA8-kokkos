/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>

namespace corsika {

  /** Parameters of the previously validated planar firn model.
   *
   * rho(z) = rho_deep - delta_rho exp((z-z_surface)/H), with
   * delta_rho = rho_deep delta_n/(n_deep-1). The refractive index uses
   * n(z) = n_deep - delta_n exp((z-z_surface)/H). This is one configurable
   * model, not a universal fit for firn. Heights belong to the caller's
   * coordinate system; the model is only defined below its surface.
   */
  struct ExponentialFirnProfile {
    MassDensityType deepDensity_{0.917_g / cube(1_cm)};
    double deepRefractiveIndex_{1.78};
    double refractiveIndexDeficit_{0.423};
    LengthType scaleDepth_{77_m};
    LengthType surfaceHeight_{0_m};

    /// Check the original profile parameter constraints.
    void validate() const;
    /// Density deficit at the surface, retaining the original operation order.
    MassDensityType getDensityDeficit() const;
    /// Positive lower density bound inside the firn.
    MassDensityType getSurfaceDensity() const;
    /// Density at a finite height no more than 1e-12 m above the surface.
    MassDensityType getDensity(LengthType height) const;
    /// Refractive index at the same height; not an optical path integral.
    double getRefractiveIndex(LengthType height) const;
    /// Reject a non-finite height or a point outside the model's domain.
    void validateHeight(LengthType height) const;
  };

} // namespace corsika

#include <corsika/detail/media/ExponentialFirnProfile.inl>
