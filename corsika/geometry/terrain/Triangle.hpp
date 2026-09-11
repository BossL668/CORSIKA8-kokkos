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
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/geometry/terrain/AABB.hpp>

#include <array>
#include <vector>

namespace corsika::terrain {

  /**
   * A triangle primitive for mesh representation.
   *
   * Stores vertex indices and precomputed data for efficient ray intersection
   * using the Möller-Trumbore algorithm.
   */
  class Triangle {

  public:
    /**
     * Construct a triangle from vertex indices.
     *
     * @param v0 Index of first vertex.
     * @param v1 Index of second vertex.
     * @param v2 Index of third vertex.
     * @param vertices Reference to the vertex array for computing edges.
     */
    Triangle(size_t v0, size_t v1, size_t v2, std::vector<Point> const& vertices);

    /**
     * Get the vertex indices.
     */
    std::array<size_t, 3> const& getVertexIndices() const;

    /**
     * Get the precomputed edge from v0 to v1.
     */
    Vector<length_d> const& getEdge1() const;

    /**
     * Get the precomputed edge from v0 to v2.
     */
    Vector<length_d> const& getEdge2() const;

    /**
     * Get the normal vector of the triangle.
     *
     * The normal is computed as (edge1 x edge2).normalized().
     */
    DirectionVector const& getNormal() const;

    /**
     * Perform Möller-Trumbore ray-triangle intersection test.
     *
     * @param origin Ray origin point.
     * @param direction Ray direction (normalized).
     * @param vertices Reference to vertex array.
     * @param t Output: distance along ray to intersection point.
     * @param epsilon Tolerance for parallel ray detection.
     * @return true if the ray intersects the triangle.
     */
    /// Tolerance for the Möller-Trumbore determinant parallel-ray test.
    /// The determinant has units of m²; this value is chosen well below any
    /// physically meaningful triangle area while remaining safely above
    /// double-precision rounding noise.
    static constexpr double kParallelEpsilon = 1e-9;

    bool intersectRay(Point const& origin, DirectionVector const& direction,
                      std::vector<Point> const& vertices, LengthType& t,
                      double epsilon = kParallelEpsilon) const;

    /**
     * Compute the AABB for this triangle.
     *
     * @param vertices Reference to vertex array.
     * @return The axis-aligned bounding box containing this triangle.
     */
    AABB computeAABB(std::vector<Point> const& vertices) const;

    /**
     * Compute the centroid of this triangle.
     *
     * @param vertices Reference to vertex array.
     * @return The center point of the triangle.
     */
    Point computeCentroid(std::vector<Point> const& vertices) const;

  private:
    std::array<size_t, 3> vertexIndices_;
    Vector<length_d> edge1_;
    Vector<length_d> edge2_;
    DirectionVector normal_;
  };

} // namespace corsika

#include <corsika/geometry/terrain/detail/Triangle.inl>

