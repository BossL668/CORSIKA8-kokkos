/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/StraightTrajectory.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace corsika {

  template <typename TModel>
  inline ExponentialFirn<TModel>::ExponentialFirn(CoordinateSystemPtr coordinateSystem,
                                                  NuclearComposition const& composition,
                                                  ExponentialFirnProfile const& profile)
      : coordinateSystem_(std::move(coordinateSystem))
      , composition_(composition)
      , profile_(profile) {
    if (!coordinateSystem_) {
      throw std::runtime_error("firn needs a coordinate system");
    }
    profile_.validate();
    if (!(profile_.getSurfaceDensity() > 0_kg / cube(1_m))) {
      throw std::runtime_error("firn surface density must remain positive");
    }
  }

  template <typename TModel>
  inline MassDensityType ExponentialFirn<TModel>::getMassDensity(
      Point const& point) const {
    auto const height = point.getZ(coordinateSystem_);
    if (!std::isfinite(height / 1_m)) { return profile_.deepDensity_; }
    return profile_.getDensity(height);
  }

  template <typename TModel>
  inline NuclearComposition const& ExponentialFirn<TModel>::getNuclearComposition()
      const {
    return composition_;
  }

  template <typename TModel>
  inline double ExponentialFirn<TModel>::getRefractiveIndex(Point const& point) const {
    auto const height = point.getZ(coordinateSystem_);
    if (!std::isfinite(height / 1_m)) { return profile_.deepRefractiveIndex_; }
    return profile_.getRefractiveIndex(height);
  }

  template <typename TModel>
  inline GrammageType ExponentialFirn<TModel>::getIntegratedGrammage(
      BaseTrajectory const& trajectory) const {
    double const lengthM = trajectory.getLength() / 1_m;
    if (std::isinf(lengthM)) {
      return std::numeric_limits<double>::infinity() * (1_g / square(1_cm));
    }
    if (std::isnan(lengthM) || lengthM < 0.) {
      throw std::runtime_error("firn grammage received a negative trajectory");
    }
    auto const [z0M, directionZ] = getStartAndDirectionZ(trajectory);
    if (!std::isfinite(z0M) || !std::isfinite(directionZ)) {
      return std::numeric_limits<double>::infinity() * (1_g / square(1_cm));
    }
    return integratedDensity(z0M, directionZ, lengthM) * (1_g / cube(1_cm)) * 1_m;
  }

  template <typename TModel>
  inline LengthType ExponentialFirn<TModel>::getArclengthFromGrammage(
      BaseTrajectory const& trajectory, GrammageType grammage) const {
    double const signedTarget = grammage / ((1_g / cube(1_cm)) * 1_m);
    if (std::isinf(signedTarget)) { return signedTarget * 1_m; }
    if (std::isnan(signedTarget)) {
      throw std::runtime_error("firn arclength received invalid grammage");
    }
    // PROPOSAL may return a negative range at its propagation threshold. Like
    // the native homogeneous/exponential media, preserve the sign so that the
    // calling continuous process can apply its zero-step policy.
    double const orientation = signedTarget < 0. ? -1. : 1.;
    double const target = std::abs(signedTarget);
    if (target == 0.) { return 0_m; }
    auto const [z0M, trajectoryDirectionZ] = getStartAndDirectionZ(trajectory);
    double const directionZ = orientation * trajectoryDirectionZ;
    if (!std::isfinite(z0M) || !std::isfinite(directionZ)) {
      return orientation * std::numeric_limits<double>::infinity() * 1_m;
    }
    double upperM = target / (profile_.getSurfaceDensity() / (1_g / cube(1_cm)));
    if (directionZ > 0.) {
      double const toSurfaceM = (profile_.surfaceHeight_ / 1_m - z0M) / directionZ;
      if (upperM > toSurfaceM) {
        double const available = integratedDensity(z0M, directionZ, toSurfaceM);
        if (available + 1.e-13 * std::max(available, target) < target) {
          return orientation * std::numeric_limits<double>::infinity() * 1_m;
        }
        upperM = toSurfaceM;
      }
    }
    double const upperIntegral = integratedDensity(z0M, directionZ, upperM);
    if (upperIntegral < target) {
      double const shortfall = target - upperIntegral;
      if (shortfall <= 1.e-12 * std::max(target, std::numeric_limits<double>::min())) {
        return orientation * upperM * 1_m;
      }
      throw std::runtime_error("failed to bracket firn grammage inverse");
    }
    double lowerM = 0.;
    for (int iteration = 0; iteration < 80; ++iteration) {
      double const middleM = 0.5 * (lowerM + upperM);
      if (integratedDensity(z0M, directionZ, middleM) < target) {
        lowerM = middleM;
      } else {
        upperM = middleM;
      }
    }
    return orientation * 0.5 * (lowerM + upperM) * 1_m;
  }

  template <typename TModel>
  inline double ExponentialFirn<TModel>::integratedDensity(double z0M, double directionZ,
                                                           double lengthM) const {
    profile_.validateHeight(z0M * 1_m);
    if (!(lengthM >= 0.) || std::isnan(lengthM) || !std::isfinite(directionZ)) {
      throw std::runtime_error("invalid straight line in firn grammage integral");
    }
    if (lengthM == 0.) { return 0.; }
    if (directionZ > 0. &&
        z0M + directionZ * lengthM > profile_.surfaceHeight_ / 1_m + 1.e-10) {
      throw std::runtime_error("firn grammage line extends above the ice surface");
    }
    double const scaleM = profile_.scaleDepth_ / 1_m;
    double const exponential0 = std::exp((z0M - profile_.surfaceHeight_ / 1_m) / scaleM);
    double exponentialIntegralM = 0.;
    if (std::abs(directionZ) < 1.e-14) {
      exponentialIntegralM = exponential0 * lengthM;
    } else {
      double const exponent = directionZ * lengthM / scaleM;
      exponentialIntegralM = exponential0 * scaleM / directionZ * std::expm1(exponent);
    }
    return (profile_.deepDensity_ / (1_g / cube(1_cm))) * lengthM -
           (profile_.getDensityDeficit() / (1_g / cube(1_cm))) * exponentialIntegralM;
  }

  template <typename TModel>
  inline std::pair<double, double> ExponentialFirn<TModel>::getStartAndDirectionZ(
      BaseTrajectory const& trajectory) const {
    // The stored line remains finite when position(0) would evaluate 0 * infinity.
    if (auto const* straight = dynamic_cast<StraightTrajectory const*>(&trajectory)) {
      return {straight->getLine().getStartPoint().getZ(coordinateSystem_) / 1_m,
              straight->getLine().getDirection().getZ(coordinateSystem_).magnitude()};
    }
    return {trajectory.getPosition(0.).getZ(coordinateSystem_) / 1_m,
            trajectory.getDirection(0.).getZ(coordinateSystem_).magnitude()};
  }

  template <typename TModel>
  inline DensityModelParameters ExponentialFirn<TModel>::exportDensity() const {
    return {DensityModelParameters::Kind::ExponentialFirn,
            profile_.deepDensity_,
            profile_.scaleDepth_,
            profile_.surfaceHeight_,
            Point(coordinateSystem_, 0_m, 0_m, 0_m),
            DirectionVector(coordinateSystem_, {0., 0., 1.}),
            profile_.getDensityDeficit()};
  }

  template <typename TModel>
  inline RefractivityModelParameters ExponentialFirn<TModel>::exportRefractivity() const {
    return {RefractivityModelParameters::Kind::ExponentialFirn,
            profile_.deepRefractiveIndex_, InverseMassDensityType{},
            profile_.refractiveIndexDeficit_};
  }

  template <typename TModel>
  inline ExponentialFirnProfile const& ExponentialFirn<TModel>::getProfile() const {
    return profile_;
  }

} // namespace corsika
