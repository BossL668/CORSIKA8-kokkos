/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 * Distributed under the 3-clause BSD license; see LICENSE.
 */

#pragma once

#include <corsika/accelerator/AcceleratorMacros.hpp>

#include <cstddef>
#include <cfloat>
#include <cmath>
#include <type_traits>

namespace corsika::geometry_detail {

  /// An outward unit normal and plane offset in metres, in one explicit frame.
  /// The closed half-space is nx*x + ny*y + nz*z <= offset_m.
  struct ConvexPlane {
    double nx;
    double ny;
    double nz;
    double offset_m;
  };

  struct ConvexRayInterval {
    bool intersects = false;
    double entry_m = 0.;
    double exit_m = 0.;
    std::size_t entry_face = static_cast<std::size_t>(-1);
    std::size_t exit_face = static_cast<std::size_t>(-1);
  };

  static_assert(std::is_trivially_copyable_v<ConvexPlane>);
  static_assert(std::is_standard_layout_v<ConvexPlane>);
  static_assert(std::is_trivially_copyable_v<ConvexRayInterval>);

  C8_ACCELERATOR_INLINE_FUNCTION inline bool finiteConvexCoordinate(double value) {
    return value == value && value <= DBL_MAX && value >= -DBL_MAX;
  }

  /// Shared host/device containment. Coordinates/tolerance are metres.
  /// Derived from cma21_askaryan.geometry.MeshGeometry.contains(): outward
  /// half-spaces avoid parity ambiguity at a triangle's shared edges/vertices.
  /// The caller owns a previously validated, non-empty convex plane array.
  C8_ACCELERATOR_INLINE_FUNCTION inline bool containsConvex(
      ConvexPlane const* planes, std::size_t count, double x, double y, double z,
      double tolerance_m = 1e-8) {
    if (planes == nullptr || count < 4 || !finiteConvexCoordinate(x) ||
        !finiteConvexCoordinate(y) || !finiteConvexCoordinate(z) ||
        !finiteConvexCoordinate(tolerance_m) || tolerance_m < 0.) {
      return false;
    }
    for (std::size_t face = 0; face < count; ++face) {
      auto const& plane = planes[face];
      if (plane.nx * x + plane.ny * y + plane.nz * z - plane.offset_m >
          tolerance_m) {
        return false;
      }
    }
    return true;
  }

  /// Intersect a forward ray with all outward half-spaces. Direction must be
  /// unit length; the returned entry/exit distances are physical metres.
  /// Tolerance classifies near-surface points only: the physical planes are
  /// never shifted to make an artificially thick layer. A surface-outward ray
  /// returns a zero-length interval, not a fresh positive entry.
  C8_ACCELERATOR_INLINE_FUNCTION inline ConvexRayInterval clipConvexRay(
      ConvexPlane const* planes, std::size_t count, double x, double y, double z,
      double dx, double dy, double dz, double tolerance_m = 1e-8) {
    ConvexRayInterval result;
    if (planes == nullptr || count < 4 || !finiteConvexCoordinate(x) ||
        !finiteConvexCoordinate(y) || !finiteConvexCoordinate(z) ||
        !finiteConvexCoordinate(dx) || !finiteConvexCoordinate(dy) ||
        !finiteConvexCoordinate(dz) || !finiteConvexCoordinate(tolerance_m) ||
        tolerance_m < 0.) {
      return result;
    }
    double const norm2 = dx * dx + dy * dy + dz * dz;
    if (!(norm2 > 1. - 1e-10 && norm2 < 1. + 1e-10)) return result;
    double entry = 0.;
    double exit = HUGE_VAL;
    for (std::size_t face = 0; face < count; ++face) {
      auto const& plane = planes[face];
      double const signedDistance =
          plane.nx * x + plane.ny * y + plane.nz * z - plane.offset_m;
      double const direction = plane.nx * dx + plane.ny * dy + plane.nz * dz;
      // Exact zero, rather than an angular epsilon: a very shallow but real
      // ray must retain its (possibly very long) physical intersection.
      if (direction == 0.) {
        if (signedDistance > tolerance_m) return ConvexRayInterval{};
        continue;
      }
      double const intersection = -signedDistance / direction;
      if (direction < 0.) {
        if (intersection > entry) {
          entry = intersection;
          result.entry_face = face;
        }
      } else if (intersection < exit) {
        exit = intersection;
        result.exit_face = face;
      }
      if (entry > exit + tolerance_m) return ConvexRayInterval{};
    }
    if (!finiteConvexCoordinate(exit) || exit < -tolerance_m) {
      return ConvexRayInterval{};
    }
    result.intersects = true;
    result.entry_m = entry;
    result.exit_m = exit < entry ? entry : exit;
    return result;
  }

} // namespace corsika::geometry_detail
