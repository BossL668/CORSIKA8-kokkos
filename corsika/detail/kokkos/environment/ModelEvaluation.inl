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

  namespace model_evaluation_detail {

    // Keep the production dot product's rounding on CUDA. Do not replace these
    // operations with a fused multiply-add or alter global compiler flags.
    CORSIKA_HOST_DEVICE inline double product(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dmul_rn(first, second);
#else
      return first * second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double sum(double first, double second) {
#if defined(__CUDA_ARCH__)
      return __dadd_rn(first, second);
#else
      return first + second;
#endif
    }

    CORSIKA_HOST_DEVICE inline double dot(double const (&first)[3],
                                          double const (&second)[3]) {
      return sum(sum(product(first[0], second[0]), product(first[1], second[1])),
                 product(first[2], second[2]));
    }

    CORSIKA_HOST_DEVICE inline double not_a_number() {
#if defined(__CUDA_ARCH__)
      return CUDART_NAN;
#else
      return std::numeric_limits<double>::quiet_NaN();
#endif
    }

  } // namespace model_evaluation_detail

  CORSIKA_HOST_DEVICE inline double density(EnvironmentModel const& model,
                                            double const (&positionM)[3]) {
    if (model.densityKind_ == EnvironmentModel::Density::ExponentialFirn) {
      return firn_density(model, positionM);
    }
    if (model.densityKind_ == EnvironmentModel::Density::Homogeneous) {
      return model.referenceDensityGPerCm3_;
    }
    double delta[3]{positionM[0] - model.anchorM_[0], positionM[1] - model.anchorM_[1],
                    positionM[2] - model.anchorM_[2]};
    auto height =
        model.densityKind_ == EnvironmentModel::Density::SlidingPlanarExponential
            ? ::sqrt(model_evaluation_detail::dot(delta, delta))
            : model_evaluation_detail::dot(delta, model.axis_);
    return model.referenceDensityGPerCm3_ *
           ::exp(model.inverseScaleLengthPerM_ * (height - model.referenceHeightM_));
  }

  CORSIKA_HOST_DEVICE inline double refractive_index(EnvironmentModel const& model,
                                                     double const (&positionM)[3]) {
    if (model.refractivityKind_ == EnvironmentModel::Refractivity::ExponentialFirn) {
      return firn_refractive_index(model, positionM);
    }
    if (model.refractivityKind_ == EnvironmentModel::Refractivity::Uniform) {
      return model.refractivityValue_;
    }
    if (model.refractivityKind_ == EnvironmentModel::Refractivity::GladstoneDale) {
      return model.refractivityValue_ *
                 (density(model, positionM) * model.inverseReferenceDensityCm3PerG_) +
             1.;
    }
    return model_evaluation_detail::not_a_number();
  }

} // namespace corsika::kokkos
