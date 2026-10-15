/*
 * (c) Copyright 2023 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/EnvironmentModelExport.hpp>

namespace corsika {

  /**
   * A tabulated refractive index.
   *
   * This class returns the value of refractive index
   * for a specific height bin.
   */
  template <typename T>
  class GladstoneDaleRefractiveIndex : public T, public IRefractivityExport {

    double const
        referenceRefractivity_; ///< The constant reference refractive index minus one.
    InverseMassDensityType const
        referenceInvDensity_; ///< The constant inverse mass density at reference point.

  public:
    /**
     * Construct a GladstoneDaleRefractiveIndex.
     *
     * This is initialized with a fixed refractive index
     * at sea level and uses the Gladstone-Dale law to
     * calculate the refractive index at a given point.
     *
     * @param seaLevelRefractiveIndex  The refractive index at sea level.
     * @param point AA point at earth's surface.
     */
    template <typename... Args>
    GladstoneDaleRefractiveIndex(double const seaLevelRefractiveIndex, Point const point,
                                 Args&&... args);

    /**
     * Evaluate the refractive index at a given location.
     *
     * @param  point    The location to evaluate at.
     * @returns    The refractive index at this point.
     */
    double getRefractiveIndex(Point const& point) const override;

    ExportRefractivity exportRefractivity() const override {
      return {ExportRefractivity::Kind::GladstoneDale,referenceRefractivity_,referenceInvDensity_};
    }

  }; // END: class GladstoneDaleRefractiveIndex

} // namespace corsika

#include <corsika/detail/media/GladstoneDaleRefractiveIndex.inl>
