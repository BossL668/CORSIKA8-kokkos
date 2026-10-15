/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Line.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/media/NuclearComposition.hpp>
#include <corsika/framework/geometry/BaseTrajectory.hpp>
#include <corsika/media/EnvironmentModelExport.hpp>

namespace corsika {

  /**
   * a homogeneous medium
   */

  template <typename T>
  class HomogeneousMedium : public T, public IDensityExport {

  public:
    HomogeneousMedium(MassDensityType density, NuclearComposition const& nuclComp);

    ExportDensity exportDensity() const override {
      auto cs=get_root_CoordinateSystem();
      return {ExportDensityKind::Homogeneous,density_,0_m,0_m,
              Point(cs,0_m,0_m,0_m),DirectionVector(cs,{0.,0.,1.})};
    }

    MassDensityType getMassDensity(Point const&) const override;

    NuclearComposition const& getNuclearComposition() const override;

    GrammageType getIntegratedGrammage(BaseTrajectory const&) const override;

    LengthType getArclengthFromGrammage(BaseTrajectory const&,
                                        GrammageType grammage) const override;

  private:
    MassDensityType const density_;
    NuclearComposition const nuclComp_;
  };

} // namespace corsika

#include <corsika/detail/media/HomogeneousMedium.inl>
