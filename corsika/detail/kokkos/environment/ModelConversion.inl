/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/IDensityModelExport.hpp>
#include <corsika/media/IRefractivityModelExport.hpp>
#include <corsika/media/IUniformMagneticFieldExport.hpp>

#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace corsika::kokkos {

  template <typename TModel>
  inline EnvironmentModel export_model(TModel const& model,
                                       CoordinateSystemPtr const& coordinateSystem,
                                       std::vector<EnvironmentComponent>& composition) {
    if (!coordinateSystem) {
      throw std::invalid_argument("model export requires a common coordinate system");
    }
    auto const* densityExport = dynamic_cast<IDensityModelExport const*>(&model);
    auto const* fieldExport = dynamic_cast<IUniformMagneticFieldExport const*>(&model);
    auto const* refractivityExport =
        dynamic_cast<IRefractivityModelExport const*>(&model);
    if (!densityExport || !fieldExport || !refractivityExport) {
      throw std::invalid_argument("model has no exact parameter exporter");
    }
    auto density = densityExport->exportDensity();
    auto field = fieldExport->exportUniformField();
    auto refractivity = refractivityExport->exportRefractivity();
    if (!density.anchor_.getCoordinateSystem() || !density.axis_.getCoordinateSystem() ||
        !field.getCoordinateSystem()) {
      throw std::invalid_argument("model export requires source coordinate systems");
    }

    EnvironmentModel result;
    switch (density.kind_) {
      case DensityModelParameters::Kind::Homogeneous:
        result.densityKind_ = EnvironmentModel::Density::Homogeneous;
        break;
      case DensityModelParameters::Kind::SlidingPlanarExponential:
        result.densityKind_ = EnvironmentModel::Density::SlidingPlanarExponential;
        break;
      case DensityModelParameters::Kind::FlatExponential:
        result.densityKind_ = EnvironmentModel::Density::FlatExponential;
        break;
      case DensityModelParameters::Kind::ExponentialFirn:
        result.densityKind_ = EnvironmentModel::Density::ExponentialFirn;
        break;
      default:
        throw std::invalid_argument("unsupported exported density kind");
    }
    result.referenceDensityGPerCm3_ =
        density.referenceDensity_ / (1_g / (1_cm * 1_cm * 1_cm));
    result.scaleLengthM_ = density.scaleLength_ / 1_m;
    result.inverseScaleLengthPerM_ =
        result.scaleLengthM_ != 0. ? 1. / result.scaleLengthM_ : 0.;
    result.referenceHeightM_ = density.referenceHeight_ / 1_m;
    result.densityDeficitGPerCm3_ = density.densityDeficit_ / (1_g / cube(1_cm));
    auto anchor = density.anchor_.getCoordinates(coordinateSystem);
    auto axis = density.axis_.getComponents(coordinateSystem);
    auto magnetic = field.getComponents(coordinateSystem);
    for (int component = 0; component < 3; ++component) {
      result.anchorM_[component] = anchor[component] / 1_m;
      result.axis_[component] = axis[component].magnitude();
      result.magneticFieldT_[component] = magnetic[component] / 1_T;
    }
    switch (refractivity.kind_) {
      case RefractivityModelParameters::Kind::Uniform:
        result.refractivityKind_ = EnvironmentModel::Refractivity::Uniform;
        break;
      case RefractivityModelParameters::Kind::GladstoneDale:
        result.refractivityKind_ = EnvironmentModel::Refractivity::GladstoneDale;
        break;
      case RefractivityModelParameters::Kind::ExponentialFirn:
        if (result.densityKind_ != EnvironmentModel::Density::ExponentialFirn) {
          throw std::invalid_argument(
              "firn index requires the matching firn density coordinates");
        }
        result.refractivityKind_ = EnvironmentModel::Refractivity::ExponentialFirn;
        break;
      default:
        throw std::invalid_argument("unsupported exported refractivity kind");
    }
    result.refractivityValue_ = refractivity.value_;
    result.refractiveIndexDeficit_ = refractivity.refractiveIndexDeficit_;
    result.inverseReferenceDensityCm3PerG_ =
        refractivity.inverseReferenceDensity_ * (1_g / (1_cm * 1_cm * 1_cm));
    result.mediumId_ = static_cast<std::int32_t>(model.getMedium());

    // New firn records must have usable parameters before any device upload.
    // Existing density branches retain their original export behavior.
    if (result.densityKind_ == EnvironmentModel::Density::ExponentialFirn) {
      if (!std::isfinite(result.referenceDensityGPerCm3_) ||
          !std::isfinite(result.densityDeficitGPerCm3_) ||
          !(result.referenceDensityGPerCm3_ > result.densityDeficitGPerCm3_) ||
          !(result.densityDeficitGPerCm3_ >= 0.) ||
          !std::isfinite(result.scaleLengthM_) || !(result.scaleLengthM_ > 0.) ||
          !std::isfinite(result.referenceHeightM_)) {
        throw std::invalid_argument("invalid exported firn density parameters");
      }
      if (result.refractivityKind_ == EnvironmentModel::Refractivity::ExponentialFirn &&
          (!std::isfinite(result.refractivityValue_) ||
           !std::isfinite(result.refractiveIndexDeficit_) ||
           !(result.refractivityValue_ > 1.) || !(result.refractiveIndexDeficit_ >= 0.) ||
           !(result.refractiveIndexDeficit_ < result.refractivityValue_ - 1.))) {
        throw std::invalid_argument("invalid exported firn refractive index parameters");
      }
    }

    auto const& nativeComposition = model.getNuclearComposition();
    auto const begin = composition.size();
    auto const count = nativeComposition.getSize();
    auto const maximumIndex = std::numeric_limits<std::uint32_t>::max();
    if (begin > maximumIndex || count > maximumIndex - begin ||
        count > composition.max_size() - begin) {
      throw std::length_error("model composition exceeds the supported array size");
    }
    auto const& codes = nativeComposition.getComponents();
    auto const& fractions = nativeComposition.getFractions();
    result.compositionBegin_ = static_cast<std::uint32_t>(begin);
    result.compositionCount_ = static_cast<std::uint32_t>(count);
    composition.reserve(begin + count);
    for (std::size_t component = 0; component < count; ++component) {
      composition.push_back(
          {static_cast<std::int32_t>(codes[component]), fractions[component]});
    }
    return result;
  }

} // namespace corsika::kokkos
