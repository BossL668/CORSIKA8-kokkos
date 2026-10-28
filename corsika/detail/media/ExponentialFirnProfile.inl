/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <stdexcept>

namespace corsika {

  inline void ExponentialFirnProfile::validate() const {
    if (!(deepDensity_ > 0_kg / cube(1_m)) || !(deepRefractiveIndex_ > 1.) ||
        !(refractiveIndexDeficit_ >= 0.) ||
        !(refractiveIndexDeficit_ < deepRefractiveIndex_ - 1.) || !(scaleDepth_ > 0_m) ||
        !std::isfinite(surfaceHeight_ / 1_m)) {
      throw std::runtime_error("invalid exponential-firn profile");
    }
  }

  inline MassDensityType ExponentialFirnProfile::getDensityDeficit() const {
    validate();
    double const deep = deepDensity_ / (1_g / cube(1_cm));
    return (deep * refractiveIndexDeficit_ / (deepRefractiveIndex_ - 1.)) *
           (1_g / cube(1_cm));
  }

  inline MassDensityType ExponentialFirnProfile::getSurfaceDensity() const {
    return deepDensity_ - getDensityDeficit();
  }

  inline MassDensityType ExponentialFirnProfile::getDensity(LengthType height) const {
    validateHeight(height);
    double const deep = deepDensity_ / (1_g / cube(1_cm));
    double const deficit = getDensityDeficit() / (1_g / cube(1_cm));
    return (deep - deficit * std::exp((height / 1_m - surfaceHeight_ / 1_m) /
                                      (scaleDepth_ / 1_m))) *
           (1_g / cube(1_cm));
  }

  inline double ExponentialFirnProfile::getRefractiveIndex(LengthType height) const {
    validateHeight(height);
    return deepRefractiveIndex_ -
           refractiveIndexDeficit_ *
               std::exp((height / 1_m - surfaceHeight_ / 1_m) / (scaleDepth_ / 1_m));
  }

  inline void ExponentialFirnProfile::validateHeight(LengthType height) const {
    validate();
    double const z = height / 1_m;
    if (!std::isfinite(z) || z > surfaceHeight_ / 1_m + 1.e-12) {
      throw std::runtime_error("exponential-firn point is outside its ice surface");
    }
  }

} // namespace corsika
