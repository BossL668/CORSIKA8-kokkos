/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include "StraightBoundaryReference.hpp"
#include "../utility/PolynomialRootsReference.hpp"

#include <Kokkos_Macros.hpp>

#include <cmath>
#include <cstdint>

namespace curved_boundary_reference {

  using shape_type = straight_boundary_reference::shape_type;
  using interval_type = straight_boundary_reference::interval_type;
  using straight_boundary_reference::difference;
  using straight_boundary_reference::dot;
  using straight_boundary_reference::infinity;
  using straight_boundary_reference::product;
  using straight_boundary_reference::sum;

  // Independent EnvironmentTreeSnapshot.hpp expressions from production
  // 41cdc543429b82a216857ad458f83deb0f210a5f. Only numeric record names change:
  // a plane's original axes[0] is the separate normal_ in the exported record.
  KOKKOS_INLINE_FUNCTION bool contains(shape_type const& shape, double const* position) {
    if (shape.kind_ == shape_type::Kind::Universe) { return true; }
    double delta[3]{position[0] - shape.centerM_[0], position[1] - shape.centerM_[1],
                    position[2] - shape.centerM_[2]};
    if (shape.kind_ == shape_type::Kind::Sphere) {
      return dot(delta, delta) < shape.radiusM_ * shape.radiusM_;
    }
    if (shape.kind_ == shape_type::Kind::SeparationPlane) {
      return dot(delta, shape.normal_) <= 0.;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (!(::fabs(dot(delta, shape.axes_[axis])) < shape.halfExtentM_[axis])) {
        return false;
      }
    }
    return true;
  }

  // Independent EnvironmentCurvedSnapshot.hpp expressions from the same fixed
  // production object. The base supplies only raw topology and straight-ray
  // data; ordered selection is tested with boundary_reference::next. Arithmetic,
  // containment and root solving below never call the newly migrated helpers.
  struct Access : straight_boundary_reference::Tree {
    double direction_[3]{};
    double field_[3]{};
    double speed_{};
    double momentumSI_{};
    double chargeSI_{};
    double gyroradiusM_{};
    bool linear_{};
    mutable bool unsupportedShape_{};
    mutable bool incompleteRoots_{};

    KOKKOS_INLINE_FUNCTION interval_type intersect(std::int32_t node) const {
      return intersectShape(shapes_[node]);
    }

    KOKKOS_INLINE_FUNCTION interval_type intersectShape(shape_type const& shape) const {
      if (linear_) {
        return straight_boundary_reference::intersect(shape, position_, velocity_);
      }
      if (shape.kind_ == shape_type::Kind::Universe) { return {}; }
      double projected[3]{difference(product(direction_[1], field_[2]),
                                     product(direction_[2], field_[1])),
                          difference(product(direction_[2], field_[0]),
                                     product(direction_[0], field_[2])),
                          difference(product(direction_[0], field_[1]),
                                     product(direction_[1], field_[0]))};
      double delta[3]{position_[0] - shape.centerM_[0], position_[1] - shape.centerM_[1],
                      position_[2] - shape.centerM_[2]};
      auto const inf = infinity();
      if (shape.kind_ == shape_type::Kind::SeparationPlane) {
        if (chargeSI_ == 0.) {
          return straight_boundary_reference::intersect(shape, position_, velocity_);
        }
        double const denom = chargeSI_ * dot(shape.normal_, projected);
        double const p = 2 * momentumSI_ * dot(direction_, shape.normal_);
        double const q = 2 * momentumSI_ * dot(shape.normal_, delta);
        auto const roots = polynomial_roots_reference::quadratic(denom, p, q);
        double first = inf;
        for (unsigned index = 0; index < roots.size; ++index) {
          if (roots.values[index] >= 0. && roots.values[index] < first) {
            first = roots.values[index];
          }
        }
        return {true, first / speed_, inf};
      }
      if (shape.kind_ != shape_type::Kind::Sphere) {
        unsupportedShape_ = true;
        return {};
      }
      auto const projected2 = dot(projected, projected);
      if (projected2 == 0.) {
        return straight_boundary_reference::intersect(shape, position_, velocity_);
      }
      double const deltaNorm = ::sqrt(dot(delta, delta));
      double const distance = ::fabs(deltaNorm - shape.radiusM_);
      if (distance > gyroradiusM_) { return {}; }
      if (gyroradiusM_ > 1000 * distance) {
        return straight_boundary_reference::intersect(shape, position_, velocity_);
      }
      double const k = chargeSI_ / momentumSI_, denom = 4. / (projected2 * k * k);
      double const b = product(sum(product(dot(projected, delta), k), 1), denom);
      double const c = dot(direction_, delta) * 2 * denom;
      double const d =
          (deltaNorm + shape.radiusM_) * (deltaNorm - shape.radiusM_) * denom;
      auto const roots = polynomial_roots_reference::stableQuartic(b, c, d);
      bool const inside = curved_boundary_reference::contains(shape, position_);
      double entry = 0., exit = 0.;
      unsigned count = 0, entryCount = 0, exitCount = 0;
      for (unsigned index = 0; index < roots.size; ++index) {
        double const distance = roots.values[index];
        if (!polynomial_roots_reference::extendedFinite(distance)) {
          incompleteRoots_ = true;
          return {};
        }
        if (inside) {
          if (distance < 1.e-4) {
            if (!entryCount || entry < distance) { entry = distance; }
            ++entryCount;
          } else {
            if (!exitCount || distance < exit) { exit = distance; }
            ++exitCount;
          }
          count = (entryCount > 0) + (exitCount > 0);
        } else {
          if (distance < -1.e-4) { continue; }
          if (!count) {
            entry = distance;
          } else if (distance < entry) {
            exit = entry;
            entry = distance;
          } else {
            exit = distance;
          }
          ++count;
        }
      }
      if (!count) { return {}; }
      if ((inside && (!entryCount || !exitCount)) || (!inside && count < 2)) {
        incompleteRoots_ = true;
        return {};
      }
      return {true, entry / speed_, exit / speed_};
    }
  };

} // namespace curved_boundary_reference
