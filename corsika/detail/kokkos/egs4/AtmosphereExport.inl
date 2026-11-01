/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::egs4 {
  inline AtmosphereData export_atmosphere(EnvironmentDataView const& tree,
                                          TransportGeometryConfig const& geometry,
                                          double earthRadiusM) {
    auto require = [](bool valid) {
      if (!valid)
        throw std::invalid_argument(
            "EGS4 AIR-NTP requires five nested atmospheric spheres, four radial "
            "exponential layers, a homogeneous top layer and a constant field");
    };
    require(tree.nodeCount_ == 6 && tree.nodes_ && tree.shapes_ && tree.models_ &&
            tree.children_ && tree.exclusionCount_ == 0 &&
            tree.shapes_[0].kind_ == EnvironmentShape::Kind::Universe &&
            tree.nodes_[0].modelIndex_ == -1 && std::isfinite(earthRadiusM) &&
            earthRadiusM > 0.);
    std::array<int, 5> nodes{};
    int parent = 0;
    for (int i = 4; i >= 0; --i) {
      auto const& node = tree.nodes_[parent];
      require(node.childCount_ == 1 && node.childOffset_ < tree.childCount_);
      int child = tree.children_[node.childOffset_];
      require(child > 0 && static_cast<unsigned>(child) < tree.nodeCount_);
      nodes[i] = child;
      require(tree.nodes_[child].parentIndex_ == parent);
      parent = child;
    }
    require(tree.nodes_[parent].childCount_ == 0);
    AtmosphereData result;
    result.numberOfLayers_ = 5;
    for (int i = 0; i < 5; ++i) {
      auto const& shape = tree.shapes_[nodes[i]];
      auto modelIndex = tree.nodes_[nodes[i]].modelIndex_;
      require(shape.kind_ == EnvironmentShape::Kind::Sphere && modelIndex >= 0 &&
              static_cast<unsigned>(modelIndex) < tree.modelCount_);
      auto const& model = tree.models_[modelIndex];
      require(model.mediumId_ == static_cast<int>(Medium::AirDry1Atm));
      require(tree.composition_ && model.compositionCount_ == 3 &&
              model.compositionBegin_ <= tree.componentCount_ &&
              model.compositionCount_ <= tree.componentCount_ - model.compositionBegin_);
      // Accept the standard C8 and historical C7 AIR-NTP number fractions.
      // This is a compatibility check, not a replacement material model.
      std::array<double, 3> fractions{};
      std::array<Code, 3> const elements{Code::Nitrogen, Code::Oxygen, Code::Argon};
      for (unsigned component = 0; component < model.compositionCount_; ++component) {
        auto const& entry = tree.composition_[model.compositionBegin_ + component];
        auto element =
            std::find(elements.begin(), elements.end(), static_cast<Code>(entry.code_));
        require(element != elements.end() && std::isfinite(entry.fraction_) &&
                entry.fraction_ > 0.);
        fractions[element - elements.begin()] += entry.fraction_;
      }
      auto matches = [&](std::array<double, 3> const& reference) {
        for (unsigned element = 0; element < fractions.size(); ++element)
          if (std::abs(fractions[element] - reference[element]) > 1.e-10) return false;
        return true;
      };
      if (!matches({.78479, .21052, .00469}) && !matches({.7847, .2105, .0048}))
        throw std::invalid_argument(
            "EGS4 AIR-NTP tables require the standard air composition");
      auto& layer = result.atmosphereLayers_[i];
      layer.innerRadiusM_ =
          i ? result.atmosphereLayers_[i - 1].outerRadiusM_ : earthRadiusM;
      layer.outerRadiusM_ = shape.radiusM_;
      layer.mediumId_ = model.mediumId_;
      require(layer.outerRadiusM_ > layer.innerRadiusM_);
      for (int axis = 0; axis < 3; ++axis) {
        if (i == 0) {
          result.earthCenterM_[axis] = shape.centerM_[axis];
          result.magneticFieldT_[axis] = model.magneticFieldT_[axis];
          result.observationPlanePointM_[axis] = geometry.observation_.pointM_[axis];
          result.observationPlaneNormal_[axis] = geometry.observation_.normal_[axis];
        }
        require(shape.centerM_[axis] == result.earthCenterM_[axis] &&
                model.magneticFieldT_[axis] == result.magneticFieldT_[axis]);
        if (i < 4) require(model.anchorM_[axis] == shape.centerM_[axis]);
      }
      layer.densityParameterA_ = model.referenceDensityGPerCm3_;
      layer.densityParameterB_ = model.referenceHeightM_;
      layer.densityParameterC_ = model.scaleLengthM_;
      if (i < 4) {
        require(model.densityKind_ ==
                    EnvironmentModel::Density::SlidingPlanarExponential &&
                model.scaleLengthM_ < 0.);
      } else {
        require(model.densityKind_ == EnvironmentModel::Density::Homogeneous);
        layer.densityModel_ = DensityModel::Homogeneous;
      }
    }
    result.maximumMagneticDeflectionRad_ = geometry.query_.maximumMagneticDeflectionRad_;
    return result;
  }
} // namespace corsika::kokkos::egs4
