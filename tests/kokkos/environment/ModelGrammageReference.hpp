/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "ModelEvaluationReference.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace model_grammage_reference {

  using model_type = corsika::kokkos::EnvironmentModel;

  // Frozen EnvironmentModelSnapshot.hpp expressions at
  // 41cdc543429b82a216857ad458f83deb0f210a5f. The independently transcribed
  // density/dot expressions in ModelEvaluationReference.hpp are unchanged between
  // its pinned 7c24233b and this commit (checked in the immutable Git objects).
  // Never call new evaluators or their implementation helpers in this oracle.
  CORSIKA_HOST_DEVICE inline double infinity() {
#if defined(__CUDA_ARCH__)
    return CUDART_INF;
#else
    return std::numeric_limits<double>::infinity();
#endif
  }

  CORSIKA_HOST_DEVICE inline double projected_axis(model_type const& m, double const* p,
                                                   double const* direction) {
    if (m.densityKind_ == model_type::Density::FlatExponential) {
      return model_evaluation_reference::dot(direction, m.axis_);
    }
    double delta[3]{p[0] - m.anchorM_[0], p[1] - m.anchorM_[1], p[2] - m.anchorM_[2]};
    return model_evaluation_reference::dot(delta, direction) /
           ::sqrt(model_evaluation_reference::dot(delta, delta));
  }

  CORSIKA_HOST_DEVICE inline double grammage(model_type const& m, double const* p,
                                             double const* direction, double lengthM) {
    if (lengthM == 0.) { return 0.; }
    auto const rho = model_evaluation_reference::density(m, p);
    if (m.densityKind_ == model_type::Density::Homogeneous) {
      return lengthM * rho * 100.;
    }
    auto const projection = projected_axis(m, p, direction);
    if (projection == 0.) { return lengthM * rho * 100.; }
    return rho * (m.scaleLengthM_ / projection) *
           ::expm1(projection * lengthM * m.inverseScaleLengthPerM_) * 100.;
  }

  CORSIKA_HOST_DEVICE inline double arclength(model_type const& m, double const* p,
                                              double const* direction,
                                              double grammageGPerCm2) {
    auto const rho = model_evaluation_reference::density(m, p);
    if (m.densityKind_ == model_type::Density::Homogeneous) {
      return grammageGPerCm2 / (rho * 100.);
    }
    auto const projection = projected_axis(m, p, direction);
    if (projection == 0.) { return grammageGPerCm2 / (rho * 100.); }
    auto const logArgument =
        grammageGPerCm2 * .01 * m.inverseScaleLengthPerM_ * projection / rho;
    return logArgument > -1. ? m.scaleLengthM_ / projection * ::log1p(logArgument)
                             : infinity();
  }

  struct Query {
    model_type model_;
    double positionM_[3]{};
    double direction_[3]{};
    double lengthM_{1.};
    double grammageGPerCm2_{0.1};
  };

  inline Query make_query(model_type::Density kind, double scale, double projection) {
    Query query;
    query.model_.densityKind_ = kind;
    query.model_.referenceDensityGPerCm3_ = 0.001225;
    query.model_.scaleLengthM_ = scale;
    query.model_.inverseScaleLengthPerM_ = 1. / scale;
    query.model_.referenceHeightM_ = 100.;
    query.model_.axis_[2] = 1.;
    query.positionM_[2] = 100.;
    query.direction_[0] = std::sqrt(1. - projection * projection);
    query.direction_[2] = projection;
    return query;
  }

  inline std::vector<Query> queries() {
    std::vector<Query> result;
    for (auto const kind :
         {model_type::Density::Homogeneous, model_type::Density::FlatExponential,
          model_type::Density::SlidingPlanarExponential}) {
      for (double const scale : {-8000., 8000.}) {
        for (double const projection :
             {-1., -0.123456789, -1e-16, 0., 1e-16, 0.3456789, 1.}) {
          for (double const length : {0., 1e-12, 1e-6, 0.25, 10., 1000.}) {
            auto query = make_query(kind, scale, projection);
            query.lengthM_ = length;
            query.grammageGPerCm2_ = length * 0.1;
            result.push_back(query);
          }
        }
      }
    }
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const inf = std::numeric_limits<double>::infinity();
    for (auto const kind : {model_type::Density::FlatExponential,
                            model_type::Density::SlidingPlanarExponential}) {
      // rho=1 and projection=1 give logArgument=-X*.01. The adjacent
      // representable X values straddle the finite escape grammage X=100.
      for (double const grammage :
           {std::nextafter(100., 0.), 100., std::nextafter(100., inf), 0., nan}) {
        auto query = make_query(kind, -1., 1.);
        query.model_.referenceDensityGPerCm3_ = 1.;
        query.grammageGPerCm2_ = grammage;
        result.push_back(query);
      }
      // Ordinary nonbinary inputs exercise dot rounding without making the
      // inverse ill-conditioned close to logArgument=-1.
      for (int index = 0; index < 6; ++index) {
        auto query = make_query(kind, -79.19123, 0.3456789);
        query.model_.referenceHeightM_ = 0.;
        query.model_.axis_[0] = 0.123456789;
        query.model_.axis_[1] = -0.567891234;
        query.model_.axis_[2] = 2.718281828;
        for (int component = 0; component < 3; ++component) {
          query.positionM_[component] =
              (index + 1) * (component + 1) * 0.3141592653589793;
        }
        query.lengthM_ = 13.7;
        query.grammageGPerCm2_ = 0.1;
        result.push_back(query);
      }
      // Input values are not silently normalized, although a nonunit direction
      // is not a physical arclength parametrization.
      auto rawDirection = make_query(kind, 1., 1.);
      rawDirection.direction_[2] = 2.;
      result.push_back(rawDirection);
    }
    for (double const exponent : {-1000., -700., -1e-16, 1e-16, 700., 1000.}) {
      auto query = make_query(model_type::Density::FlatExponential, 1.,
                              std::copysign(1., exponent));
      query.lengthM_ = std::abs(exponent);
      query.grammageGPerCm2_ = 1.;
      result.push_back(query);
    }
    // IEEE cases retain the original arithmetic, not a repaired physical model.
    for (auto const kind :
         {model_type::Density::Homogeneous, model_type::Density::FlatExponential,
          model_type::Density::SlidingPlanarExponential}) {
      // Negative requests retain the original raw arithmetic; they are not
      // asserted to describe a valid forward physical transport step.
      auto negative = make_query(kind, -8000., 0.3456789);
      negative.lengthM_ = -0.25;
      negative.grammageGPerCm2_ = -0.1;
      result.push_back(negative);
      for (double const rho : {0., -0., -1., inf, -inf, nan}) {
        for (double const projection : {0., 1.}) {
          for (double const grammage : {0., 1.}) {
            auto query = make_query(kind, -8000., projection);
            query.model_.referenceDensityGPerCm3_ = rho;
            query.lengthM_ = 5.;
            query.grammageGPerCm2_ = grammage;
            result.push_back(query);
          }
        }
      }
      for (double const length : {0., 1.}) {
        auto query = make_query(kind, -8000., 1.);
        query.positionM_[0] = nan;
        query.direction_[0] = nan;
        query.lengthM_ = length;
        result.push_back(query);
      }
    }
    for (double const length : {0., 1.}) {
      for (double const grammage : {0., 1.}) {
        auto query =
            make_query(model_type::Density::SlidingPlanarExponential, -8000., 1.);
        query.positionM_[2] = 0.; // singular radial-axis origin
        query.lengthM_ = length;
        query.grammageGPerCm2_ = grammage;
        result.push_back(query);
      }
    }
    return result;
  }

} // namespace model_grammage_reference
