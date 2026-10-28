/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>

namespace corsika {

  /**
   * Read-only host-side parameters of a supported refractive-index model.
   *
   * For Uniform, value_ is the refractive index and inverseReferenceDensity_ is
   * zero. For GladstoneDale, value_ is the reference refractivity (n - 1), and
   * inverseReferenceDensity_ is the stored inverse reference mass density. Keep
   * those factors separate to preserve the original model's evaluation order.
   */
  struct RefractivityModelParameters {
    /// Supported refractive-index laws.
    enum class Kind { Uniform, GladstoneDale, ExponentialFirn };

    Kind kind_;    ///< The refractive-index law described by these parameters.
    double value_; ///< Uniform index or reference refractivity, depending on kind_.
    InverseMassDensityType inverseReferenceDensity_; ///< Stored density scale factor.
    /// Firn only: surface index deficit; value_ is the deep index. Uses density's height
    /// and scale.
    double refractiveIndexDeficit_{};
  };

} // namespace corsika
