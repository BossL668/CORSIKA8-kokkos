/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/IRefractiveIndexModel.hpp>
#include <corsika/media/IRefractivityModelExport.hpp>

namespace corsika {

  /**
   * A uniform refractive index.
   *
   * This class returns the same refractive index
   * for all evaluated locations.
   */
  template <typename T>
  class UniformRefractiveIndex : public T, public IRefractivityModelExport {

    double n_; ///< The constant refractive index that we use.

  public:
    /**
     * Construct a UniformRefractiveIndex.
     *
     * This is initialized with a fixed refractive index
     * and returns this refractive index at all locations.
     *
     * @param n  The refractive index to return everywhere.
     */
    template <typename... Args>
    UniformRefractiveIndex(double const n, Args&&... args);

    /**
     * Evaluate the refractive index at a given location. Note: since this
     * is *uniform* model, it has no position-dependence.
     *
     * @param  point    The location to evaluate at (not used internally).
     * @returns    The refractive index at this point.
     */
    double getRefractiveIndex(Point const& point) const override;

    /**
     * Set the refractive index returned by this instance.
     *
     * @param  n  The global refractive index.
     */
    void setRefractiveIndex(double const n);

    /// Copy the current uniform index without querying a spatial point.
    [[nodiscard]] RefractivityModelParameters exportRefractivity() const override;

  }; // END: class RefractiveIndex

} // namespace corsika

#include <corsika/detail/media/UniformRefractiveIndex.inl>
