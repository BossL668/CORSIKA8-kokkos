/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {

  template <typename TDerived>
  inline SlidingPlanarExponential<TDerived>::SlidingPlanarExponential(
      Point const& p0, MassDensityType const rho0, LengthType const lambda,
      NuclearComposition const& nuclComp, LengthType const referenceHeight)
      : BaseExponential<SlidingPlanarExponential<TDerived>>(p0, referenceHeight, rho0,
                                                            lambda)
      , nuclComp_(nuclComp) {}

  template <typename TDerived>
  inline MassDensityType SlidingPlanarExponential<TDerived>::getMassDensity(
      Point const& point) const {
    auto const heightFull =
        (point - BaseExponential<SlidingPlanarExponential<TDerived>>::getAnchorPoint())
            .getNorm();
    return BaseExponential<SlidingPlanarExponential<TDerived>>::getMassDensity(
        heightFull);
  }

  template <typename TDerived>
  inline NuclearComposition const&
  SlidingPlanarExponential<TDerived>::getNuclearComposition() const {
    return nuclComp_;
  }

  template <typename TDerived>
  inline GrammageType SlidingPlanarExponential<TDerived>::getIntegratedGrammage(
      BaseTrajectory const& traj) const {
    auto const axis =
        (traj.getPosition(0) -
         BaseExponential<SlidingPlanarExponential<TDerived>>::getAnchorPoint())
            .normalized();
    return BaseExponential<SlidingPlanarExponential<TDerived>>::getIntegratedGrammage(
        traj, axis);
  }

  template <typename TDerived>
  inline LengthType SlidingPlanarExponential<TDerived>::getArclengthFromGrammage(
      BaseTrajectory const& traj, GrammageType const grammage) const {
    auto const axis =
        (traj.getPosition(0) -
         BaseExponential<SlidingPlanarExponential<TDerived>>::getAnchorPoint())
            .normalized();
    return BaseExponential<SlidingPlanarExponential<TDerived>>::getArclengthFromGrammage(
        traj, grammage, axis);
  }

  template <typename TDerived>
  inline DensityModelParameters SlidingPlanarExponential<TDerived>::exportDensity()
      const {
    return {DensityModelParameters::Kind::SlidingPlanarExponential,
            Base::getReferenceDensity(),
            Base::getScaleLength(),
            Base::getReferenceHeight(),
            Base::getAnchorPoint(),
            DirectionVector(Base::getAnchorPoint().getCoordinateSystem(), {0., 0., 1.})};
  }

} // namespace corsika
