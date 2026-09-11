/*
 * (c) Copyright 2020 CORSIKA Project, corsika8@kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <limits>
#include <map>

namespace corsika::terrain {

  inline BVH::BVH(std::vector<Triangle> const& triangles,
                  std::vector<Point> const& vertices, size_t maxLeafSize) {
    if (triangles.empty()) {
      root_ = nullptr;
      return;
    }

    // Create initial index list
    std::vector<size_t> indices(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) { indices[i] = i; }

    // Build the tree recursively
    root_ = buildRecursive(indices, triangles, vertices, maxLeafSize);
  }

  inline std::unique_ptr<BVHNode> BVH::buildRecursive(
      std::vector<size_t>& indices, std::vector<Triangle> const& triangles,
      std::vector<Point> const& vertices, size_t maxLeafSize) {
    auto node = std::make_unique<BVHNode>();

    // Compute bounds for all triangles in this node
    for (size_t idx : indices) {
      AABB triAABB = triangles[idx].computeAABB(vertices);
      node->bounds = AABB::merge(node->bounds, triAABB);
    }

    // If few enough triangles, create a leaf
    if (indices.size() <= maxLeafSize) {
      node->triangleIndices = indices;
      return node;
    }

    // Otherwise, split using SAH
    CoordinateSystemPtr cs = node->bounds.getCoordinateSystem();

    // Compute centroids for triangles and map triangle index to centroid
    std::map<size_t, Point> centroidMap;
    for (size_t idx : indices) {
      centroidMap.emplace(idx, triangles[idx].computeCentroid(vertices));
    }

    // Find the best split axis and position using SAH
    int bestAxis = -1;
    double bestCost = std::numeric_limits<double>::max();
    size_t bestSplit = indices.size() / 2;
    std::vector<size_t> bestSortedIndices = indices;

    // Surface area of parent
    double const parentArea = node->bounds.getSurfaceArea() / (1_m * 1_m);

    for (int axis = 0; axis < 3; ++axis) {
      // Sort indices by centroid on this axis
      std::vector<size_t> sortedIndices = indices;
      std::sort(sortedIndices.begin(), sortedIndices.end(),
                [&centroidMap, cs, axis](size_t a, size_t b) {
                  LengthType ca, cb;
                  if (axis == 0) {
                    ca = centroidMap.at(a).getX(cs);
                    cb = centroidMap.at(b).getX(cs);
                  } else if (axis == 1) {
                    ca = centroidMap.at(a).getY(cs);
                    cb = centroidMap.at(b).getY(cs);
                  } else {
                    ca = centroidMap.at(a).getZ(cs);
                    cb = centroidMap.at(b).getZ(cs);
                  }
                  return ca < cb;
                });

      // Try different split positions
      // Precompute prefix AABBs from left
      std::vector<AABB> leftAABBs(sortedIndices.size());
      AABB leftAccum;
      for (size_t i = 0; i < sortedIndices.size(); ++i) {
        leftAccum =
            AABB::merge(leftAccum, triangles[sortedIndices[i]].computeAABB(vertices));
        leftAABBs[i] = leftAccum;
      }

      // Precompute suffix AABBs from right
      std::vector<AABB> rightAABBs(sortedIndices.size());
      AABB rightAccum;
      for (size_t i = sortedIndices.size(); i > 0; --i) {
        rightAccum = AABB::merge(rightAccum,
                                 triangles[sortedIndices[i - 1]].computeAABB(vertices));
        rightAABBs[i - 1] = rightAccum;
      }

      // Evaluate SAH cost for each split position
      for (size_t split = 1; split < sortedIndices.size(); ++split) {
        double const leftArea = leftAABBs[split - 1].getSurfaceArea() / (1_m * 1_m);
        double const rightArea = rightAABBs[split].getSurfaceArea() / (1_m * 1_m);
        size_t const leftCount = split;
        size_t const rightCount = sortedIndices.size() - split;

        // SAH cost: C = C_trav + (SA_left/SA_parent)*N_left*C_isect +
        //                       (SA_right/SA_parent)*N_right*C_isect
        // Simplified: C = 1 + (SA_left/SA_parent)*N_left + (SA_right/SA_parent)*N_right
        double cost = 1.0;
        if (parentArea > 0) {
          cost +=
              (leftArea / parentArea) * leftCount + (rightArea / parentArea) * rightCount;
        } else {
          cost += leftCount + rightCount;
        }

        if (cost < bestCost) {
          bestCost = cost;
          bestAxis = axis;
          bestSplit = split;
        }
      }
      // Sorting is unchanged. Copy the winning order once per axis, not at
      // every improving split (which can copy O(N^2) indices).
      if (bestAxis == axis) bestSortedIndices = sortedIndices;
    }

    // If no good split found, create a leaf
    if (bestAxis < 0 || bestSplit == 0 || bestSplit == indices.size()) {
      node->triangleIndices = indices;
      return node;
    }

    // Use the best sorted order
    indices = bestSortedIndices;

    // Split the indices
    std::vector<size_t> leftIndices(indices.begin(), indices.begin() + bestSplit);
    std::vector<size_t> rightIndices(indices.begin() + bestSplit, indices.end());

    // Recursively build children
    node->left = buildRecursive(leftIndices, triangles, vertices, maxLeafSize);
    node->right = buildRecursive(rightIndices, triangles, vertices, maxLeafSize);

    return node;
  }

  inline void BVH::traverse(Point const& origin, DirectionVector const& direction,
                            std::function<void(size_t)> const& callback) const {
    if (!root_) return;

    // Compute inverse direction for slab-based AABB intersection
    CoordinateSystemPtr cs = root_->bounds.getCoordinateSystem();
    auto const dirComps = direction.getComponents(cs);

    double const dx = dirComps.getX();
    double const dy = dirComps.getY();
    double const dz = dirComps.getZ();

    double const invX = std::abs(dx) > kDirEpsilon
                            ? 1.0 / dx
                            : (dx >= 0 ? 1.0 / kDirEpsilon : -1.0 / kDirEpsilon);
    double const invY = std::abs(dy) > kDirEpsilon
                            ? 1.0 / dy
                            : (dy >= 0 ? 1.0 / kDirEpsilon : -1.0 / kDirEpsilon);
    double const invZ = std::abs(dz) > kDirEpsilon
                            ? 1.0 / dz
                            : (dz >= 0 ? 1.0 / kDirEpsilon : -1.0 / kDirEpsilon);

    Vector<dimensionless_d> invDir(cs, {invX, invY, invZ});
    traverseRecursive(root_.get(), origin, invDir, callback);
  }

  inline void BVH::traverseRecursive(BVHNode const* node, Point const& origin,
                                     Vector<dimensionless_d> const& invDir,
                                     std::function<void(size_t)> const& callback) const {
    if (!node) return;

    LengthType tMin, tMax;
    if (!node->bounds.intersectsRay(origin, invDir, tMin, tMax)) { return; }

    if (node->isLeaf()) {
      for (size_t idx : node->triangleIndices) { callback(idx); }
    } else {
      traverseRecursive(node->left.get(), origin, invDir, callback);
      traverseRecursive(node->right.get(), origin, invDir, callback);
    }
  }

  inline bool BVH::findClosestIntersection(Point const& origin,
                                           DirectionVector const& direction,
                                           std::vector<Triangle> const& triangles,
                                           std::vector<Point> const& vertices,
                                           size_t& hitTriangle,
                                           LengthType& hitDistance) const {
    hitTriangle = static_cast<size_t>(-1);
    hitDistance = std::numeric_limits<double>::infinity() * 1_m;

    traverse(origin, direction, [&](size_t idx) {
      LengthType t;
      if (triangles[idx].intersectRay(origin, direction, vertices, t)) {
        if (t < hitDistance) {
          hitDistance = t;
          hitTriangle = idx;
        }
      }
    });

    return hitTriangle != static_cast<size_t>(-1);
  }

  inline std::vector<std::pair<size_t, LengthType>> BVH::findAllIntersections(
      Point const& origin, DirectionVector const& direction,
      std::vector<Triangle> const& triangles, std::vector<Point> const& vertices,
      LengthType deduplicationTolerance) const {
    std::vector<std::pair<size_t, LengthType>> results;

    traverse(origin, direction, [&](size_t idx) {
      LengthType t;
      if (triangles[idx].intersectRay(origin, direction, vertices, t)) {
        results.emplace_back(idx, t);
      }
    });

    // Sort by distance
    std::sort(results.begin(), results.end(),
              [](auto const& a, auto const& b) { return a.second < b.second; });

    // Remove duplicate hits closer together than deduplicationTolerance.
    // This collapses coplanar triangle-edge hits that differ only by
    // floating-point noise while leaving genuinely distinct surfaces intact.
    if (!results.empty()) {
      std::vector<std::pair<size_t, LengthType>> unique;
      unique.push_back(results[0]);

      for (size_t i = 1; i < results.size(); ++i) {
        if (results[i].second - unique.back().second > deduplicationTolerance) {
          unique.push_back(results[i]);
        }
      }

      return unique;
    }

    return results;
  }

  inline BVHNode const* BVH::getRoot() const { return root_.get(); }

  inline AABB const& BVH::getBounds() const {
    static AABB emptyAABB;
    if (!root_) return emptyAABB;
    return root_->bounds;
  }

} // namespace corsika

