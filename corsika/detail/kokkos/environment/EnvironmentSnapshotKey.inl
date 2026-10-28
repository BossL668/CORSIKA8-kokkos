/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cstring>
namespace corsika::kokkos {
  inline std::vector<double> make_environment_snapshot_key(
      EnvironmentDataView const& tree) {
    std::vector<double> key;
    auto add = [&](auto value) { key.push_back(static_cast<double>(value)); };
    add(tree.nodeCount_);
    add(tree.childCount_);
    add(tree.exclusionCount_);
    add(tree.modelCount_);
    add(tree.componentCount_);
    for (unsigned i = 0; i < tree.nodeCount_; ++i) {
      auto const& node = tree.nodes_[i];
      auto const& shape = tree.shapes_[i];
      add(node.parentIndex_);
      add(node.childOffset_);
      add(node.childCount_);
      add(node.excludedOffset_);
      add(node.excludedCount_);
      // The new enum reserves zero for Invalid; production starts at Universe.
      add(static_cast<int>(shape.kind_) - 1);
      add(shape.radiusM_);
      add(node.modelIndex_);
      for (int a = 0; a < 3; ++a) {
        add(shape.centerM_[a]);
        add(shape.halfExtentM_[a]);
        for (int b = 0; b < 3; ++b)
          add(shape.kind_ == EnvironmentShape::Kind::SeparationPlane && a == 0
                  ? shape.normal_[b]
                  : shape.axes_[a][b]);
      }
    }
    for (unsigned i = 0; i < tree.childCount_; ++i) add(tree.children_[i]);
    for (unsigned i = 0; i < tree.exclusionCount_; ++i) add(tree.exclusions_[i]);
    for (unsigned i = 0; i < tree.modelCount_; ++i) {
      auto const& model = tree.models_[i];
      add(static_cast<int>(model.densityKind_) - 1);
      add(model.referenceDensityGPerCm3_);
      add(model.scaleLengthM_);
      add(model.inverseScaleLengthPerM_);
      add(model.referenceHeightM_);
      add(model.refractivityKind_);
      add(model.refractivityValue_);
      add(model.inverseReferenceDensityCm3PerG_);
      add(model.mediumId_);
      add(model.compositionBegin_);
      add(model.compositionCount_);
      for (int a = 0; a < 3; ++a) {
        add(model.anchorM_[a]);
        add(model.axis_[a]);
        add(model.magneticFieldT_[a]);
      }
    }
    for (unsigned i = 0; i < tree.componentCount_; ++i) {
      add(tree.composition_[i].code_);
      add(tree.composition_[i].fraction_);
    }
    // Existing media keep their original key/fingerprint. Firn's added model
    // kinds must also distinguish the new exported parameters on reuse.
    for (unsigned i = 0; i < tree.modelCount_; ++i) {
      auto const& model = tree.models_[i];
      if (model.densityKind_ == EnvironmentModel::Density::ExponentialFirn ||
          model.refractivityKind_ == EnvironmentModel::Refractivity::ExponentialFirn) {
        add(model.densityDeficitGPerCm3_);
        add(model.refractiveIndexDeficit_);
      }
    }
    return key;
  }
  inline std::uint64_t environment_snapshot_fingerprint(EnvironmentDataView const& tree) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (double value : make_environment_snapshot_key(tree)) {
      if (value == 0.) value = 0.;
      std::uint64_t bits{};
      static_assert(sizeof(bits) == sizeof(value));
      std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned i = 0; i < 8; ++i) {
        hash ^= (bits >> (8 * i)) & 255;
        hash *= UINT64_C(1099511628211);
      }
    }
    return hash;
  }
} // namespace corsika::kokkos
