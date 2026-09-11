/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <sstream>

namespace corsika::terrain {

  inline AABB::AABB()
      : min_(get_root_CoordinateSystem(), {std::numeric_limits<double>::max() * 1_m,
                                           std::numeric_limits<double>::max() * 1_m,
                                           std::numeric_limits<double>::max() * 1_m})
      , max_(get_root_CoordinateSystem(), {std::numeric_limits<double>::lowest() * 1_m,
                                           std::numeric_limits<double>::lowest() * 1_m,
                                           std::numeric_limits<double>::lowest() * 1_m})
      , valid_(false) {}

  inline AABB::AABB(Point const& min, Point const& max)
      : min_(min)
      , max_(max)
      , valid_(true) {}

  inline bool AABB::contains(Point const& p) const {
    if (!valid_) return false;

    CoordinateSystemPtr cs = min_.getCoordinateSystem();
    LengthType const px = p.getX(cs);
    LengthType const py = p.getY(cs);
    LengthType const pz = p.getZ(cs);

    LengthType const minX = min_.getX(cs);
    LengthType const minY = min_.getY(cs);
    LengthType const minZ = min_.getZ(cs);

    LengthType const maxX = max_.getX(cs);
    LengthType const maxY = max_.getY(cs);
    LengthType const maxZ = max_.getZ(cs);

    return (px >= minX && px <= maxX && py >= minY && py <= maxY && pz >= minZ &&
            pz <= maxZ);
  }

  inline bool AABB::intersectsRay(Point const& origin,
                                  Vector<dimensionless_d> const& invDir, LengthType& tMin,
                                  LengthType& tMax) const {
    if (!valid_) return false;

    CoordinateSystemPtr cs = min_.getCoordinateSystem();

    LengthType const ox = origin.getX(cs);
    LengthType const oy = origin.getY(cs);
    LengthType const oz = origin.getZ(cs);

    double const idx = invDir.getX(cs);
    double const idy = invDir.getY(cs);
    double const idz = invDir.getZ(cs);

    LengthType const minX = min_.getX(cs);
    LengthType const minY = min_.getY(cs);
    LengthType const minZ = min_.getZ(cs);

    LengthType const maxX = max_.getX(cs);
    LengthType const maxY = max_.getY(cs);
    LengthType const maxZ = max_.getZ(cs);

    // Compute intersection intervals for each axis
    LengthType t1x = (minX - ox) * idx;
    LengthType t2x = (maxX - ox) * idx;
    if (t1x > t2x) std::swap(t1x, t2x);

    LengthType t1y = (minY - oy) * idy;
    LengthType t2y = (maxY - oy) * idy;
    if (t1y > t2y) std::swap(t1y, t2y);

    LengthType t1z = (minZ - oz) * idz;
    LengthType t2z = (maxZ - oz) * idz;
    if (t1z > t2z) std::swap(t1z, t2z);

    // Find the intersection of all intervals
    tMin = std::max({t1x, t1y, t1z});
    tMax = std::min({t2x, t2y, t2z});

    return tMax >= tMin && tMax >= 0_m;
  }

  inline AABB AABB::merge(AABB const& a, AABB const& b) {
    if (!a.valid_) return b;
    if (!b.valid_) return a;

    CoordinateSystemPtr cs = a.min_.getCoordinateSystem();

    LengthType const minX = std::min(a.min_.getX(cs), b.min_.getX(cs));
    LengthType const minY = std::min(a.min_.getY(cs), b.min_.getY(cs));
    LengthType const minZ = std::min(a.min_.getZ(cs), b.min_.getZ(cs));

    LengthType const maxX = std::max(a.max_.getX(cs), b.max_.getX(cs));
    LengthType const maxY = std::max(a.max_.getY(cs), b.max_.getY(cs));
    LengthType const maxZ = std::max(a.max_.getZ(cs), b.max_.getZ(cs));

    return AABB(Point(cs, minX, minY, minZ), Point(cs, maxX, maxY, maxZ));
  }

  inline void AABB::expand(Point const& p) {
    CoordinateSystemPtr cs = min_.getCoordinateSystem();

    LengthType const px = p.getX(cs);
    LengthType const py = p.getY(cs);
    LengthType const pz = p.getZ(cs);

    if (!valid_) {
      min_ = Point(cs, px, py, pz);
      max_ = Point(cs, px, py, pz);
      valid_ = true;
    } else {
      LengthType const minX = std::min(min_.getX(cs), px);
      LengthType const minY = std::min(min_.getY(cs), py);
      LengthType const minZ = std::min(min_.getZ(cs), pz);

      LengthType const maxX = std::max(max_.getX(cs), px);
      LengthType const maxY = std::max(max_.getY(cs), py);
      LengthType const maxZ = std::max(max_.getZ(cs), pz);

      min_ = Point(cs, minX, minY, minZ);
      max_ = Point(cs, maxX, maxY, maxZ);
    }
  }

  inline Point const& AABB::getMin() const { return min_; }

  inline Point const& AABB::getMax() const { return max_; }

  inline CoordinateSystemPtr AABB::getCoordinateSystem() const {
    return min_.getCoordinateSystem();
  }

  inline decltype(1_m * 1_m) AABB::getSurfaceArea() const {
    if (!valid_) return 0_m * 0_m;

    CoordinateSystemPtr cs = min_.getCoordinateSystem();
    LengthType const dx = max_.getX(cs) - min_.getX(cs);
    LengthType const dy = max_.getY(cs) - min_.getY(cs);
    LengthType const dz = max_.getZ(cs) - min_.getZ(cs);

    return 2 * (dx * dy + dy * dz + dz * dx);
  }

  inline Point AABB::getCenter() const {
    CoordinateSystemPtr cs = min_.getCoordinateSystem();
    return Point(cs, (min_.getX(cs) + max_.getX(cs)) / 2,
                 (min_.getY(cs) + max_.getY(cs)) / 2,
                 (min_.getZ(cs) + max_.getZ(cs)) / 2);
  }

  inline bool AABB::isValid() const { return valid_; }

  inline std::string AABB::asString() const {
    std::ostringstream txt;
    txt << "AABB(min=" << min_ << ", max=" << max_ << ", valid=" << valid_ << ")";
    return txt.str();
  }

} // namespace corsika

