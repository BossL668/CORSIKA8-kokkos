/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/BaseTrajectory.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/media/ExponentialFirnProfile.hpp>
#include <corsika/media/IDensityModelExport.hpp>
#include <corsika/media/IRefractivityModelExport.hpp>
#include <corsika/media/NuclearComposition.hpp>

#include <utility>

namespace corsika {

  /** Planar firn density and refractive index with exact straight-line grammage.
   *
   * The model retains the validated firn formula, surface checks, horizontal
   * branch, and 80-step bracketed inverse. Material identity and magnetic field
   * are supplied by the usual C8 mixins; the caller supplies the composition.
   * The coordinate system's z coordinate defines height. Optical ray bending
   * and propagation across the surface are not implemented by this medium.
   */
  template <typename TModel>
  class ExponentialFirn : public TModel,
                          public IDensityModelExport,
                          public IRefractivityModelExport {
  public:
    ExponentialFirn(CoordinateSystemPtr coordinateSystem,
                    NuclearComposition const& composition,
                    ExponentialFirnProfile const& profile = {});

    MassDensityType getMassDensity(Point const& point) const override;
    NuclearComposition const& getNuclearComposition() const override;
    double getRefractiveIndex(Point const& point) const override;
    GrammageType getIntegratedGrammage(BaseTrajectory const& trajectory) const override;
    LengthType getArclengthFromGrammage(BaseTrajectory const& trajectory,
                                        GrammageType grammage) const override;

    /// Export exact parameters, not a fit to density samples.
    [[nodiscard]] DensityModelParameters exportDensity() const override;
    [[nodiscard]] RefractivityModelParameters exportRefractivity() const override;
    ExponentialFirnProfile const& getProfile() const;

  private:
    std::pair<double, double> getStartAndDirectionZ(
        BaseTrajectory const& trajectory) const;
    double integratedDensity(double startHeightM, double directionZ,
                             double lengthM) const;

    CoordinateSystemPtr coordinateSystem_;
    NuclearComposition composition_;
    ExponentialFirnProfile profile_;
  };

} // namespace corsika

#include <corsika/detail/media/ExponentialFirn.inl>
