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

  namespace model_grammage_detail {

    CORSIKA_HOST_DEVICE inline double projected_axis(EnvironmentModel const& model,
                                                     double const (&positionM)[3],
                                                     double const (&direction)[3]) {
      if (model.densityKind_ == EnvironmentModel::Density::FlatExponential) {
        return model_evaluation_detail::dot(direction, model.axis_);
      }
      double delta[3]{positionM[0] - model.anchorM_[0], positionM[1] - model.anchorM_[1],
                      positionM[2] - model.anchorM_[2]};
      return model_evaluation_detail::dot(delta, direction) /
             ::sqrt(model_evaluation_detail::dot(delta, delta));
    }

    CORSIKA_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
      return CUDART_INF;
#else
      return std::numeric_limits<double>::infinity();
#endif
    }

  } // namespace model_grammage_detail

  CORSIKA_HOST_DEVICE inline double grammage(EnvironmentModel const& model,
                                             double const (&positionM)[3],
                                             double const (&direction)[3],
                                             double lengthM) {
    if (model.densityKind_ == EnvironmentModel::Density::ExponentialFirn) {
      return firn_grammage(model, positionM, direction, lengthM);
    }
    if (lengthM == 0.) { return 0.; }
    auto const localDensity = density(model, positionM);
    if (model.densityKind_ == EnvironmentModel::Density::Homogeneous) {
      return lengthM * localDensity * 100.;
    }
    auto const projection =
        model_grammage_detail::projected_axis(model, positionM, direction);
    if (projection == 0.) { return lengthM * localDensity * 100.; }
    return localDensity * (model.scaleLengthM_ / projection) *
           ::expm1(projection * lengthM * model.inverseScaleLengthPerM_) * 100.;
  }

  CORSIKA_HOST_DEVICE inline double arclength(EnvironmentModel const& model,
                                              double const (&positionM)[3],
                                              double const (&direction)[3],
                                              double grammageGPerCm2) {
    if (model.densityKind_ == EnvironmentModel::Density::ExponentialFirn) {
      return firn_arclength(model, positionM, direction, grammageGPerCm2);
    }
    auto const localDensity = density(model, positionM);
    if (model.densityKind_ == EnvironmentModel::Density::Homogeneous) {
      return grammageGPerCm2 / (localDensity * 100.);
    }
    auto const projection =
        model_grammage_detail::projected_axis(model, positionM, direction);
    if (projection == 0.) { return grammageGPerCm2 / (localDensity * 100.); }
    auto const logArgument =
        grammageGPerCm2 * .01 * model.inverseScaleLengthPerM_ * projection / localDensity;
    return logArgument > -1. ? model.scaleLengthM_ / projection * ::log1p(logArgument)
                             : model_grammage_detail::infinity();
  }

} // namespace corsika::kokkos
