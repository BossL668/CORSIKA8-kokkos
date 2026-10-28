/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace corsika::kokkos {

  namespace firn_model_detail {
    double constexpr minimum_normal = std::numeric_limits<double>::min();
    CORSIKA_HOST_DEVICE inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
      return ::isfinite(value);
#else
      return std::isfinite(value);
#endif
    }
    CORSIKA_HOST_DEVICE inline double infinity() { return HUGE_VAL; }
    CORSIKA_HOST_DEVICE inline double invalid() {
#if defined(__CUDA_ARCH__)
      return CUDART_NAN;
#else
      return std::numeric_limits<double>::quiet_NaN();
#endif
    }
    CORSIKA_HOST_DEVICE inline double dot(double const (&left)[3],
                                          double const (&right)[3]) {
#if defined(__CUDA_ARCH__)
      return __dadd_rn(
          __dadd_rn(__dmul_rn(left[0], right[0]), __dmul_rn(left[1], right[1])),
          __dmul_rn(left[2], right[2]));
#else
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
#endif
    }
    CORSIKA_HOST_DEVICE inline double height(EnvironmentModel const& model,
                                             double const (&positionM)[3]) {
      double const delta[3]{positionM[0] - model.anchorM_[0],
                            positionM[1] - model.anchorM_[1],
                            positionM[2] - model.anchorM_[2]};
      return dot(delta, model.axis_);
    }
    CORSIKA_HOST_DEVICE inline bool valid_height(EnvironmentModel const& model,
                                                 double zM) {
      return finite(zM) && zM <= model.referenceHeightM_ + 1.e-12;
    }

    // Numeric integral in (g/cm^3)*m, before the public g/cm^2 conversion.
    CORSIKA_HOST_DEVICE inline double integral(EnvironmentModel const& model, double z0M,
                                               double directionZ, double lengthM) {
      if (!valid_height(model, z0M) || !(lengthM >= 0.) || !finite(directionZ)) {
        return invalid();
      }
      if (lengthM == 0.) { return 0.; }
      if (directionZ > 0. &&
          z0M + directionZ * lengthM > model.referenceHeightM_ + 1.e-10) {
        return invalid();
      }
      double const exponential0 =
          ::exp((z0M - model.referenceHeightM_) / model.scaleLengthM_);
      double exponentialIntegralM = 0.;
      if (::fabs(directionZ) < 1.e-14) {
        exponentialIntegralM = exponential0 * lengthM;
      } else {
        double const exponent = directionZ * lengthM / model.scaleLengthM_;
        exponentialIntegralM =
            exponential0 * model.scaleLengthM_ / directionZ * ::expm1(exponent);
      }
      return model.referenceDensityGPerCm3_ * lengthM -
             model.densityDeficitGPerCm3_ * exponentialIntegralM;
    }
  } // namespace firn_model_detail

  CORSIKA_HOST_DEVICE inline double firn_density(EnvironmentModel const& model,
                                                 double const (&positionM)[3]) {
    using namespace firn_model_detail;
    auto const zM = height(model, positionM);
    if (!finite(zM)) { return model.referenceDensityGPerCm3_; }
    if (!valid_height(model, zM)) { return invalid(); }
    return model.referenceDensityGPerCm3_ -
           model.densityDeficitGPerCm3_ *
               ::exp((zM - model.referenceHeightM_) / model.scaleLengthM_);
  }

  CORSIKA_HOST_DEVICE inline double firn_refractive_index(EnvironmentModel const& model,
                                                          double const (&positionM)[3]) {
    using namespace firn_model_detail;
    auto const zM = height(model, positionM);
    if (!finite(zM)) { return model.refractivityValue_; }
    if (!valid_height(model, zM)) { return invalid(); }
    return model.refractivityValue_ -
           model.refractiveIndexDeficit_ *
               ::exp((zM - model.referenceHeightM_) / model.scaleLengthM_);
  }

  CORSIKA_HOST_DEVICE inline double firn_grammage(EnvironmentModel const& model,
                                                  double const (&positionM)[3],
                                                  double const (&direction)[3],
                                                  double lengthM) {
    using namespace firn_model_detail;
    if (lengthM == infinity() || lengthM == -infinity()) { return infinity(); }
    if (!(lengthM >= 0.)) { return invalid(); }
    double const z0M = height(model, positionM);
    double const directionZ = dot(direction, model.axis_);
    if (!finite(z0M) || !finite(directionZ)) { return infinity(); }
    return integral(model, z0M, directionZ, lengthM) * 100.;
  }

  CORSIKA_HOST_DEVICE inline double firn_arclength(EnvironmentModel const& model,
                                                   double const (&positionM)[3],
                                                   double const (&direction)[3],
                                                   double grammageGPerCm2) {
    using namespace firn_model_detail;
    double const target = grammageGPerCm2 / 100.;
    if (target == infinity() || target == -infinity()) { return infinity(); }
    if (!(target >= 0.)) { return invalid(); }
    if (target == 0.) { return 0.; }
    double const z0M = height(model, positionM);
    double const directionZ = dot(direction, model.axis_);
    if (!finite(z0M) || !finite(directionZ)) { return infinity(); }
    double upperM =
        target / (model.referenceDensityGPerCm3_ - model.densityDeficitGPerCm3_);
    if (directionZ > 0.) {
      double const toSurfaceM = (model.referenceHeightM_ - z0M) / directionZ;
      if (upperM > toSurfaceM) {
        double const available = integral(model, z0M, directionZ, toSurfaceM);
        if (!finite(available)) { return invalid(); }
        if (available + 1.e-13 * (available > target ? available : target) < target) {
          return infinity();
        }
        upperM = toSurfaceM;
      }
    }
    double const upperIntegral = integral(model, z0M, directionZ, upperM);
    if (!finite(upperIntegral)) { return invalid(); }
    if (upperIntegral < target) {
      double const shortfall = target - upperIntegral;
      double const minimum = firn_model_detail::minimum_normal;
      if (shortfall <= 1.e-12 * (target > minimum ? target : minimum)) { return upperM; }
      return invalid();
    }
    double lowerM = 0.;
    for (int iteration = 0; iteration < 80; ++iteration) {
      double const middleM = 0.5 * (lowerM + upperM);
      if (integral(model, z0M, directionZ, middleM) < target) {
        lowerM = middleM;
      } else {
        upperM = middleM;
      }
    }
    return 0.5 * (lowerM + upperM);
  }

} // namespace corsika::kokkos
