/*
 * (c) Copyright 2020, 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cmath>

namespace corsika::kokkos {

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION CurvedBoundaryAccess<TTreeView>::CurvedBoundaryAccess(
      TTreeView const& tree, double const (&positionM)[3],
      double const (&velocityMPerS)[3], double const (&direction)[3],
      double const (&fieldT)[3], double speedMPerS, double momentumKgMPerS,
      double chargeC, double gyroradiusM, bool linear)
      : tree_(tree)
      , positionM_{positionM[0], positionM[1], positionM[2]}
      , velocityMPerS_{velocityMPerS[0], velocityMPerS[1], velocityMPerS[2]}
      , direction_{direction[0], direction[1], direction[2]}
      , fieldT_{fieldT[0], fieldT[1], fieldT[2]}
      , speedMPerS_(speedMPerS)
      , momentumKgMPerS_(momentumKgMPerS)
      , chargeC_(chargeC)
      , gyroradiusM_(gyroradiusM)
      , linear_(linear) {}

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION GeometryIntersections
  CurvedBoundaryAccess<TTreeView>::intersect(node_index_type node) const {
    return intersectShape(tree_.getShape(node));
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION GeometryIntersections
  CurvedBoundaryAccess<TTreeView>::intersectShape(EnvironmentShape const& shape) const {
    using namespace straight_intersection_detail;
    if (linear_) { return straight_intersect(shape, positionM_, velocityMPerS_); }
    if (shape.kind_ == EnvironmentShape::Kind::Universe) { return {}; }
    double projected[3]{difference(product(direction_[1], fieldT_[2]),
                                   product(direction_[2], fieldT_[1])),
                        difference(product(direction_[2], fieldT_[0]),
                                   product(direction_[0], fieldT_[2])),
                        difference(product(direction_[0], fieldT_[1]),
                                   product(direction_[1], fieldT_[0]))};
    double delta[3]{positionM_[0] - shape.centerM_[0], positionM_[1] - shape.centerM_[1],
                    positionM_[2] - shape.centerM_[2]};
    auto const inf = infinity();
    if (shape.kind_ == EnvironmentShape::Kind::SeparationPlane) {
      if (chargeC_ == 0.) {
        return straight_intersect(shape, positionM_, velocityMPerS_);
      }
      double const denominator = chargeC_ * dot(shape.normal_, projected);
      double const linearCoefficient =
          2 * momentumKgMPerS_ * dot(direction_, shape.normal_);
      double const constantCoefficient = 2 * momentumKgMPerS_ * dot(shape.normal_, delta);
      auto const roots =
          quadratic_roots(denominator, linearCoefficient, constantCoefficient);
      double first = inf;
      for (unsigned int index = 0; index < roots.getSize(); ++index) {
        if (roots[index] >= 0. && roots[index] < first) { first = roots[index]; }
      }
      return {first / speedMPerS_, inf};
    }
    if (shape.kind_ != EnvironmentShape::Kind::Sphere) {
      unsupportedShape_ = true;
      return {};
    }
    auto const projectedSquared = dot(projected, projected);
    if (projectedSquared == 0.) {
      return straight_intersect(shape, positionM_, velocityMPerS_);
    }
    double const deltaNorm = ::sqrt(dot(delta, delta));
    double const distance = ::fabs(deltaNorm - shape.radiusM_);
    if (distance > gyroradiusM_) { return {}; }
    if (gyroradiusM_ > 1000 * distance) {
      return straight_intersect(shape, positionM_, velocityMPerS_);
    }
    double const k = chargeC_ / momentumKgMPerS_;
    double const denominator = 4. / (projectedSquared * k * k);
    double const quadraticCoefficient =
        product(sum(product(dot(projected, delta), k), 1), denominator);
    double const linearCoefficient = dot(direction_, delta) * 2 * denominator;
    double const constantCoefficient =
        (deltaNorm + shape.radiusM_) * (deltaNorm - shape.radiusM_) * denominator;
    auto const roots = stable_quartic_roots(quadraticCoefficient, linearCoefficient,
                                            constantCoefficient);
    bool const inside = contains(shape, positionM_);
    double entry = 0.;
    double exit = 0.;
    unsigned int count = 0;
    unsigned int entryCount = 0;
    unsigned int exitCount = 0;
    for (unsigned int index = 0; index < roots.getSize(); ++index) {
      double const rootDistance = roots[index];
      if (!extended_finite(rootDistance)) {
        incompleteRoots_ = true;
        return {};
      }
      if (inside) {
        if (rootDistance < 1.e-4) {
          if (!entryCount || entry < rootDistance) { entry = rootDistance; }
          ++entryCount;
        } else {
          if (!exitCount || rootDistance < exit) { exit = rootDistance; }
          ++exitCount;
        }
        count = (entryCount > 0) + (exitCount > 0);
      } else {
        if (rootDistance < -1.e-4) { continue; }
        if (!count) {
          entry = rootDistance;
        } else if (rootDistance < entry) {
          exit = entry;
          entry = rootDistance;
        } else {
          exit = rootDistance;
        }
        ++count;
      }
    }
    if (!count) { return {}; }
    // Preserve production error propagation instead of inventing a missing
    // endpoint or silently dropping an unresolved boundary.
    if ((inside && (!entryCount || !exitCount)) || (!inside && count < 2)) {
      incompleteRoots_ = true;
      return {};
    }
    return {entry / speedMPerS_, exit / speedMPerS_};
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION double CurvedBoundaryAccess<TTreeView>::getSpeedMPerS() const {
    return speedMPerS_;
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION bool CurvedBoundaryAccess<TTreeView>::hasUnsupportedShape()
      const {
    return unsupportedShape_;
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION bool CurvedBoundaryAccess<TTreeView>::hasIncompleteRoots()
      const {
    return incompleteRoots_;
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION auto CurvedBoundaryAccess<TTreeView>::getParent(
      node_index_type node) const -> node_index_type {
    return tree_.getParent(node);
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION std::uint32_t CurvedBoundaryAccess<TTreeView>::getChildCount(
      node_index_type node) const {
    return tree_.getChildCount(node);
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION auto CurvedBoundaryAccess<TTreeView>::getChild(
      node_index_type node, std::uint32_t child) const -> node_index_type {
    return tree_.getChild(node, child);
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION std::uint32_t CurvedBoundaryAccess<TTreeView>::getExcludedCount(
      node_index_type node) const {
    return tree_.getExcludedCount(node);
  }

  template <typename TTreeView>
  KOKKOS_INLINE_FUNCTION auto CurvedBoundaryAccess<TTreeView>::getExcluded(
      node_index_type node, std::uint32_t excluded) const -> node_index_type {
    return tree_.getExcluded(node, excluded);
  }

} // namespace corsika::kokkos
