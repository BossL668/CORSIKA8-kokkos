/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/ParticleProperties.hpp>
#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>

namespace corsika {

  template <typename TDerived>
  inline BaseExponential<TDerived>::BaseExponential(Point const& point,
                                                    LengthType const referenceHeight,
                                                    MassDensityType const rho0,
                                                    LengthType const lambda)
      : rho0_(rho0)
      , lambda_(lambda)
      , invLambda_(1 / lambda)
      , point_(point)
      , referenceHeight_(referenceHeight) {}

  template <typename TDerived>
  inline auto const& BaseExponential<TDerived>::getImplementation() const {
    return *static_cast<TDerived const*>(this);
  }

  template <typename TDerived>
  inline MassDensityType BaseExponential<TDerived>::getMassDensity(
      LengthType const height) const {
    return rho0_ * exp(invLambda_ * (height - referenceHeight_));
  }

  template <typename TDerived>
  inline GrammageType BaseExponential<TDerived>::getIntegratedGrammage(
      BaseTrajectory const& traj, DirectionVector const& axis) const {
    LengthType const length = traj.getLength();
    if (length == LengthType::zero()) { return GrammageType::zero(); }

    // this corresponds to height:
    double const uDotA = traj.getDirection(0).dot(axis);
    MassDensityType const rhoStart =
        getImplementation().getMassDensity(traj.getPosition(0));

    if (uDotA == 0) {
      return length * rhoStart;
    } else {
      return rhoStart * (lambda_ / uDotA) * expm1(uDotA * length * invLambda_);
    }
  }

  template <typename TDerived>
  inline LengthType BaseExponential<TDerived>::getArclengthFromGrammage(
      BaseTrajectory const& traj, GrammageType const grammage,
      DirectionVector const& axis) const {
    // this corresponds to height:
    double const uDotA = traj.getDirection(0).dot(axis);
    MassDensityType const rhoStart =
        getImplementation().getMassDensity(traj.getPosition(0));

    if (uDotA == 0) {
      return grammage / rhoStart;
    } else {
      auto const logArg = grammage * invLambda_ * uDotA / rhoStart;
      if (logArg > -1) {
        return lambda_ / uDotA * log1p(logArg);
      } else {
        return std::numeric_limits<typename decltype(grammage)::value_type>::infinity() *
               meter;
      }
    }
  }

  template <typename TDerived>
  inline MassDensityType BaseExponential<TDerived>::getReferenceDensity() const {
    return rho0_;
  }

  template <typename TDerived>
  inline LengthType BaseExponential<TDerived>::getScaleLength() const {
    return lambda_;
  }

  template <typename TDerived>
  inline LengthType BaseExponential<TDerived>::getReferenceHeight() const {
    return referenceHeight_;
  }

} // namespace corsika
