/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/Point.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>

#include <algorithm>
#include <limits>
#include <string>

namespace corsika::terrain {

  /**
   * Axis-Aligned Bounding Box (AABB) for spatial acceleration structures.
   *
   * The AABB is defined by its minimum and maximum corner points in a
   * given coordinate system. It provides methods for containment tests
   * and ray intersection tests.
   */
  class AABB {

  public:
    /**
     * Construct an empty (invalid) AABB.
     */
    AABB();

    /**
     * Construct an AABB from minimum and maximum corner points.
     *
     * @param min The minimum corner point (smallest x, y, z values).
     * @param max The maximum corner point (largest x, y, z values).
     */
    AABB(Point const& min, Point const& max);

    /**
     * Check if a point is inside the AABB.
     *
     * @param p The point to test.
     * @return true if the point is inside (or on the boundary of) the AABB.
     */
    bool contains(Point const& p) const;

    /**
     * Test if a ray intersects this AABB.
     *
     * Uses the slab method for ray-box intersection.
     *
     * @param origin The ray origin point.
     * @param invDir The inverse of the ray direction (1/dx, 1/dy, 1/dz).
     * @param tMin Output: the entry time (if intersection occurs).
     * @param tMax Output: the exit time (if intersection occurs).
     * @return true if the ray intersects the AABB.
     */
    bool intersectsRay(Point const& origin, Vector<dimensionless_d> const& invDir,
                       LengthType& tMin, LengthType& tMax) const;

    /**
     * Merge two AABBs into a single AABB that contains both.
     *
     * @param a First AABB.
     * @param b Second AABB.
     * @return A new AABB that contains both input AABBs.
     */
    static AABB merge(AABB const& a, AABB const& b);

    /**
     * Expand this AABB to include a point.
     *
     * @param p The point to include.
     */
    void expand(Point const& p);

    /**
     * Get the minimum corner point.
     */
    Point const& getMin() const;

    /**
     * Get the maximum corner point.
     */
    Point const& getMax() const;

    /**
     * Get the coordinate system of this AABB.
     */
    CoordinateSystemPtr getCoordinateSystem() const;

    /**
     * Calculate the surface area of this AABB.
     *
     * Used for Surface Area Heuristic (SAH) in BVH construction.
     *
     * @return The surface area.
     */
    decltype(1_m * 1_m) getSurfaceArea() const;

    /**
     * Get the center point of the AABB.
     */
    Point getCenter() const;

    /**
     * Check if this AABB is valid (non-empty).
     */
    bool isValid() const;

    std::string asString() const;

  private:
    Point min_;
    Point max_;
    bool valid_;
  };

} // namespace corsika

#include <corsika/geometry/terrain/detail/AABB.inl>

