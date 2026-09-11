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
#include <corsika/geometry/terrain/Triangle.hpp>

#include <memory>
#include <vector>
#include <functional>

namespace corsika::terrain {

  /**
   * A node in the Bounding Volume Hierarchy (BVH) tree.
   *
   * Each node contains an AABB and either:
   * - Two child nodes (internal node), or
   * - A list of triangle indices (leaf node)
   */
  struct BVHNode {
    AABB bounds;

    // For internal nodes
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;

    // For leaf nodes (empty for internal nodes)
    std::vector<size_t> triangleIndices;

    /**
     * Check if this node is a leaf node.
     */
    bool isLeaf() const { return !left && !right; }
  };

  /**
   * Bounding Volume Hierarchy for efficient ray-triangle intersection.
   *
   * The BVH is a binary tree where each node has an axis-aligned bounding
   * box. Internal nodes have two children, while leaf nodes contain triangle
   * indices. This structure allows O(log n) average-case ray intersection
   * queries instead of O(n) brute-force.
   */
  class BVH {

  public:
    /**
     * Build a BVH from a list of triangles.
     *
     * Uses the Surface Area Heuristic (SAH) to determine optimal split
     * positions during tree construction.
     *
     * @param triangles The triangles to build the BVH from.
     * @param vertices The vertices referenced by the triangles.
     * @param maxLeafSize Maximum number of triangles in a leaf node.
     */
    BVH(std::vector<Triangle> const& triangles, std::vector<Point> const& vertices,
        size_t maxLeafSize = 4);

    /**
     * Traverse the BVH and find all triangles that potentially intersect a ray.
     *
     * @param origin Ray origin point.
     * @param direction Ray direction (normalized).
     * @param callback Function called with each potentially intersecting triangle index.
     */
    void traverse(Point const& origin, DirectionVector const& direction,
                  std::function<void(size_t)> const& callback) const;

    /**
     * Find the closest ray-triangle intersection.
     *
     * @param origin Ray origin point.
     * @param direction Ray direction (normalized).
     * @param triangles The triangle array.
     * @param vertices The vertex array.
     * @param hitTriangle Output: index of the hit triangle (-1 if no hit).
     * @param hitDistance Output: distance to the hit point.
     * @return true if an intersection was found.
     */
    bool findClosestIntersection(Point const& origin, DirectionVector const& direction,
                                 std::vector<Triangle> const& triangles,
                                 std::vector<Point> const& vertices, size_t& hitTriangle,
                                 LengthType& hitDistance) const;

    /// Threshold below which a direction component is treated as zero during
    /// BVH slab intersection.  Applied to normalised direction components
    /// (dimensionless, range [0,1]); the reciprocal fallback ±1/kDirEpsilon
    /// is used instead of ±∞ to keep arithmetic well-defined.
    static constexpr double kDirEpsilon = 1e-10;

    /**
     * Find all ray-triangle intersections.
     *
     * Hits closer together than @p deduplicationTolerance are collapsed into
     * one entry.  Set this to the mesh's boundaryPadding so that coplanar
     * triangle-edge duplicates (which differ only by floating-point noise) are
     * removed while genuinely distinct surfaces remain separate.
     *
     * @param origin Ray origin point.
     * @param direction Ray direction (normalized).
     * @param triangles The triangle array.
     * @param vertices The vertex array.
     * @param deduplicationTolerance Minimum separation between distinct hits.
     * @return Vector of pairs (triangle index, distance).
     */
    std::vector<std::pair<size_t, LengthType>> findAllIntersections(
        Point const& origin, DirectionVector const& direction,
        std::vector<Triangle> const& triangles, std::vector<Point> const& vertices,
        LengthType deduplicationTolerance) const;

    /**
     * Get the root node of the BVH.
     */
    BVHNode const* getRoot() const;

    /**
     * Get the overall bounding box of the BVH.
     */
    AABB const& getBounds() const;

  private:
    std::unique_ptr<BVHNode> root_;

    /**
     * Recursively build the BVH tree using SAH.
     */
    std::unique_ptr<BVHNode> buildRecursive(std::vector<size_t>& indices,
                                            std::vector<Triangle> const& triangles,
                                            std::vector<Point> const& vertices,
                                            size_t maxLeafSize);

    /**
     * Recursive traversal helper.
     */
    void traverseRecursive(BVHNode const* node, Point const& origin,
                           Vector<dimensionless_d> const& invDir,
                           std::function<void(size_t)> const& callback) const;
  };

} // namespace corsika

#include <corsika/geometry/terrain/detail/BVH.inl>

