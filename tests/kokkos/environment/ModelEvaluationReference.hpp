/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/environment/EnvironmentModel.hpp>

#include <corsika/framework/core/HostDevice.hpp>

#include <cmath>
#include <limits>
#include <vector>

#if defined(__CUDACC__)
#include <math_constants.h>
#endif

namespace model_evaluation_reference {

  using model_type = corsika::kokkos::EnvironmentModel;

  // Frozen expressions from production 7c24233bead442c136361dfb02b23450b17efbe6:
  // EnvironmentTreeSnapshot.hpp product/sum/dot, and
  // EnvironmentModelSnapshot.hpp density/refractiveIndex. Only the record names
  // are adapted. Do not call any new evaluator or its internal helpers here.
  CORSIKA_HOST_DEVICE inline double product(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
  }

  CORSIKA_HOST_DEVICE inline double sum(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
  }

  CORSIKA_HOST_DEVICE inline double dot(double const* a, double const* b) {
    return sum(sum(product(a[0], b[0]), product(a[1], b[1])), product(a[2], b[2]));
  }

  CORSIKA_HOST_DEVICE inline double not_a_number() {
#if defined(__CUDA_ARCH__)
    return CUDART_NAN;
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
  }

  CORSIKA_HOST_DEVICE inline double density(model_type const& m, double const* p) {
    if (m.densityKind_ == model_type::Density::Homogeneous) {
      return m.referenceDensityGPerCm3_;
    }
    double delta[3]{p[0] - m.anchorM_[0], p[1] - m.anchorM_[1], p[2] - m.anchorM_[2]};
    auto h = m.densityKind_ == model_type::Density::SlidingPlanarExponential
                 ? ::sqrt(dot(delta, delta))
                 : dot(delta, m.axis_);
    return m.referenceDensityGPerCm3_ *
           ::exp(m.inverseScaleLengthPerM_ * (h - m.referenceHeightM_));
  }

  CORSIKA_HOST_DEVICE inline double refractive_index(model_type const& m,
                                                     double const* p) {
    if (m.refractivityKind_ == model_type::Refractivity::Uniform) {
      return m.refractivityValue_;
    }
    if (m.refractivityKind_ == model_type::Refractivity::GladstoneDale) {
      return m.refractivityValue_ * (density(m, p) * m.inverseReferenceDensityCm3PerG_) +
             1.;
    }
    return not_a_number();
  }

  struct Query {
    model_type model_;
    double positionM_[3]{};
  };

  // Shared host-created inputs keep the CUDA executable independent of native
  // C8 models. Degenerate records test IEEE arithmetic, not physical media.
  inline std::vector<Query> queries() {
    std::vector<Query> result;
    for (auto const kind :
         {model_type::Density::Homogeneous, model_type::Density::FlatExponential,
          model_type::Density::SlidingPlanarExponential}) {
      for (auto const refractivity :
           {model_type::Refractivity::Unavailable, model_type::Refractivity::Uniform,
            model_type::Refractivity::GladstoneDale}) {
        for (double const scale : {-8000., 8000.}) {
          for (int index = 0; index < 97; ++index) {
            Query query;
            auto& model = query.model_;
            model.densityKind_ = kind;
            model.referenceDensityGPerCm3_ = 0.001225;
            model.scaleLengthM_ = scale;
            model.inverseScaleLengthPerM_ = 1. / scale;
            model.referenceHeightM_ = 63.;
            model.anchorM_[0] = 6371000.;
            model.anchorM_[1] = -1234.5;
            model.anchorM_[2] = 3456.75;
            model.axis_[0] = 0.25;
            model.axis_[1] = -0.5;
            model.axis_[2] = 2.5; // deliberately not normalized
            model.refractivityKind_ = refractivity;
            model.refractivityValue_ =
                refractivity == model_type::Refractivity::Uniform ? 1.000312 : 0.000312;
            model.inverseReferenceDensityCm3PerG_ = 1. / 0.001225;
            for (int component = 0; component < 3; ++component) {
              model.magneticFieldT_[component] = (component - 1) * 12.5e-6;
              query.positionM_[component] =
                  model.anchorM_[component] + (index - 48) * (component + 1) * 11.25;
            }
            result.push_back(query);
          }
        }
      }
    }
    // Ordinary nonbinary inputs exercise rounding in dot products without
    // amplifying cancellation into an ill-conditioned cross-device comparison.
    for (auto const kind : {model_type::Density::FlatExponential,
                            model_type::Density::SlidingPlanarExponential}) {
      for (int index = 0; index < 12; ++index) {
        Query query;
        auto& model = query.model_;
        model.densityKind_ = kind;
        model.referenceDensityGPerCm3_ = 0.0123456789;
        model.scaleLengthM_ = -7.919123;
        model.inverseScaleLengthPerM_ = 1. / model.scaleLengthM_;
        model.referenceHeightM_ = 0.142857;
        model.axis_[0] = 0.123456789;
        model.axis_[1] = -0.567891234;
        model.axis_[2] = 2.718281828;
        model.refractivityKind_ = model_type::Refractivity::GladstoneDale;
        model.refractivityValue_ = 0.000271828;
        model.inverseReferenceDensityCm3PerG_ = 1013.271828;
        for (int component = 0; component < 3; ++component) {
          model.anchorM_[component] = (component - 1) * 3.123456789;
          query.positionM_[component] =
              model.anchorM_[component] +
              (index + 1) * (component + 1) * 0.3141592653589793 +
              0.2718281828459045 / (component + 1);
        }
        result.push_back(query);
      }
    }
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const infinity = std::numeric_limits<double>::infinity();
    for (double const rho : {0., -0., -0.001, infinity, -infinity, nan}) {
      Query query;
      query.model_.densityKind_ = model_type::Density::Homogeneous;
      query.model_.referenceDensityGPerCm3_ = rho;
      query.model_.refractivityKind_ = model_type::Refractivity::GladstoneDale;
      query.model_.refractivityValue_ = 0.000312;
      query.model_.inverseReferenceDensityCm3PerG_ = 1000.;
      query.positionM_[0] = nan;
      query.positionM_[1] = infinity;
      query.positionM_[2] = -infinity;
      result.push_back(query);
    }
    for (double const exponent : {-1000., -745., -700., 0., 700., 709., 1000.}) {
      Query query;
      query.model_.densityKind_ = model_type::Density::FlatExponential;
      query.model_.referenceDensityGPerCm3_ = 1.;
      query.model_.axis_[2] = 1.;
      query.model_.scaleLengthM_ = 1.;
      query.model_.inverseScaleLengthPerM_ = 1.;
      query.model_.refractivityKind_ = model_type::Refractivity::GladstoneDale;
      query.model_.refractivityValue_ = 0.001;
      query.model_.inverseReferenceDensityCm3PerG_ = -3.;
      query.positionM_[2] = exponent;
      result.push_back(query);
      query.model_.referenceDensityGPerCm3_ = -0.;
      result.push_back(query);
    }
    for (auto const kind : {model_type::Density::FlatExponential,
                            model_type::Density::SlidingPlanarExponential}) {
      for (double const inverseScale : {0., -0., infinity, -infinity, nan}) {
        Query query;
        query.model_.densityKind_ = kind;
        query.model_.referenceDensityGPerCm3_ = -0.001;
        query.model_.scaleLengthM_ = 0.;
        query.model_.inverseScaleLengthPerM_ = inverseScale;
        query.model_.axis_[2] = 2.5;
        query.model_.refractivityKind_ = model_type::Refractivity::Uniform;
        query.model_.refractivityValue_ = -2.;
        result.push_back(query);     // radial center or zero flat projection
        query.positionM_[2] = 1e200; // retain original squared-norm overflow
        result.push_back(query);
        query.positionM_[2] = nan;
        result.push_back(query);
      }
    }
    return result;
  }

} // namespace model_evaluation_reference
